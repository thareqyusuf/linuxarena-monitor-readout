#!/usr/bin/env python3
"""Track G runner (TASK-ver3 / G4'): gpt-oss-20b live monitor, 7-position residual-stream capture, two readings per action.

Per action (traj_id, action_idx) in a run-order stage (or a --subset), in --render mode:

  1. render the user text with monitor_prompt.render_user_text (original is sha256-checked against the
     manifest), apply the pinned harmony template (date + reasoning effort are literals, asserted);
  2. ONE prefill forward pass (chunked_eager, fixed --attn-chunk-size, logits_to_keep=1) with forward hooks on
     the embedding output and every decoder layer; the residual stream at the four PROMPT-side positions
     (instruction_end, transcript_end, action_end, prompt_end == old `final`) is kept as fp16;
  3. forced readout on a deep copy of the prefill cache (discarded afterwards): `<|channel|>final<|message|>`
     then `<score>\n` -> `expected_score`, `digit_dist`, `digit_token_mass` (unchanged from TASK-ver2);
  4. optional (--n-samples k): k temperature samples, each from its own deep copy of the prefill cache;
  5. greedy unforced generation on the original cache, budget --max-new-tokens, stop on <|return|>/<|call|>;
     `score` is parsed from `<score>N</score>` in the final channel. NEW: the same hooks record the residual
     stream of every decode step into a GPU buffer; after parsing, the three GENERATION-side positions are
     located and kept: analysis_end (token before the analysis channel's <|end|>), reasoning_end (last token
     before `</reasoning>`), pre_score (the `>\n` of `<score>\n`, whose next-token distribution is the digit).
     At pre_score the digit distribution of that step is recorded as expected_score_post / digit_dist_post /
     digit_token_mass_post. Missing positions -> NaN rows, capture_ok=false; the action does not fail.
  6. append the record to <out>/scores.jsonl; activations [7, 25, 2880] fp16 are buffered and flushed to
     <out>/acts_XXXX.pt every --checkpoint-every actions (dict: "_meta" + "traj|idx|render" -> tensor), with
     <out>/acts_index.json. SIGTERM/SIGINT/SIGHUP flush the buffer after the current action, then exit 130.

Resume: keys in acts_index.json are done. scores.jsonl lines whose activations were never flushed (a kill
between checkpoints) or whose status is `failed` are moved to scores.jsonl.bak_<ts> (nothing is deleted) and
the action is recomputed, so scores.jsonl never holds a duplicate key. `skipped` (too long) lines are kept.

Rolling stats (frac score>=3, sd E, frac hit budget, frac capture_ok, s/action, ETA) are logged every 25
actions and a score histogram every 200. There is NO halt gate (TASK-ver3 decision).

Usage (chain: g4_chain.sh; single stage):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf \
  python experiments/local-monitor-activations/run_monitor.py --attn chunked_eager --attn-chunk-size 256 \
      --max-prompt-tokens 32000 --reasoning-effort low --template-date 2026-09-10 \
      --run-order experiments/local-monitor-activations/run_order.json --stage main \
      --out checkpoints/local-monitor-activations/out/main
"""
from __future__ import annotations

import argparse
import bisect
import collections
import copy
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
SLUG = HERE.parent.name
REPO = HERE.parents[2]
RAW = REPO / "results" / "raw" / SLUG
sys.path.insert(0, str(HERE.parent))
import chunked_attention  # noqa: E402
import monitor_prompt as mp  # noqa: E402

FORCED_FINAL = "<|channel|>final<|message|>"
SCORE_TAG = "<score>\n"
ANALYSIS_MARK = "<|channel|>analysis<|message|>"
STOP_UNFORCED = [200002, 200012]          # <|return|>, <|call|>  (NOT <|end|>: it closes the analysis message)
END_ID, MESSAGE_ID = 200007, 200008       # <|end|>, <|message|>
SCORE_RE = re.compile(r"<score>\s*(\d+)\s*</score>", re.S | re.I)
SCORE_LOOSE_RE = re.compile(r"<[^>]*\bscore\b[^>]*>\s*(\d+)\s*</[^>]*\bscore\b[^>]*>", re.S | re.I)
REASONING_CLOSE = "</reasoning>"

# capture positions, in the order they are stored along axis 0 of every activation tensor (TASK-ver3)
PROMPT_POSITIONS = ("instruction_end", "transcript_end", "action_end", "prompt_end")
GEN_POSITIONS = ("analysis_end", "reasoning_end", "pre_score")
POSITIONS = PROMPT_POSITIONS + GEN_POSITIONS
POSITION_ALIASES = {"prompt_end": "final"}                 # old name (TASK-ver2 shards) -> written in every record too
PROMPT_POS_SOURCE = {"instruction_end": "instruction_end", "transcript_end": "transcript_end",
                     "action_end": "action_end", "prompt_end": "final"}   # name in monitor_prompt.tokenize_with_positions
N_LAYERS, D_MODEL = 24, 2880


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PositionCapture:
    """Forward hooks on embed_tokens + every decoder layer.
    Prefill (input length > 1, armed via arm_prefill): keeps output[0, positions, :] -> [n_pos, n_layers+1, d] fp16 on CPU.
    Decode (input length == 1, armed by setting gen_step): writes output[0, 0, :] into gen_buf[gen_step, layer] on the GPU,
    so the greedy loop pays no sync; the three generation-side rows are sliced out after parsing."""

    def __init__(self, embed: torch.nn.Module, layers: list[torch.nn.Module], gen_budget: int, d_model: int, device):
        self.n = len(layers) + 1
        self.embed, self.layers = embed, layers
        self.buf: list[torch.Tensor | None] = [None] * self.n
        self.positions: torch.Tensor | None = None
        self.gen_buf = torch.empty(gen_budget, self.n, d_model, dtype=torch.float16, device=device)
        self.gen_step: int | None = None
        self.handles: list = []
        self.attach()

    def attach(self):
        if self.handles:
            return
        self.handles = [self.embed.register_forward_hook(self._hook(0))] + \
                       [layer.register_forward_hook(self._hook(i + 1)) for i, layer in enumerate(self.layers)]

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @property
    def attached(self) -> bool:
        return bool(self.handles)

    def _hook(self, i):
        def fn(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if out.dim() != 3:
                return
            if self.positions is not None and out.shape[1] > 1:                 # prefill
                self.buf[i] = out[0, self.positions, :].detach().to(torch.float16).cpu()
            elif self.gen_step is not None and out.shape[1] == 1:               # one greedy decode step
                self.gen_buf[self.gen_step, i].copy_(out[0, 0, :].detach())
        return fn

    def arm_prefill(self, positions: list[int]):
        self.buf = [None] * self.n
        self.positions = torch.tensor(positions)

    def take_prefill(self) -> torch.Tensor:
        self.positions = None
        assert all(b is not None for b in self.buf), "a layer did not fire during prefill"
        return torch.stack(self.buf, dim=1).contiguous()           # [n_positions, n_layers+1, d_model] fp16

    def take_gen(self, gen_indices: list[int | None]) -> torch.Tensor:
        rows = []
        for g in gen_indices:
            if g is None:
                rows.append(torch.full((self.n, self.gen_buf.shape[2]), float("nan"), dtype=torch.float16))
            else:
                rows.append(self.gen_buf[g].detach().cpu().clone())
        return torch.stack(rows, dim=0).contiguous()


def digit_ids(tok) -> dict[int, int]:
    out = {}
    for v in range(11):
        enc = tok.encode(str(v), add_special_tokens=False)
        assert len(enc) == 1, (v, enc)
        out[v] = enc[0]
    return out


def read_distribution(tok, logits: torch.Tensor, digits: dict[int, int]) -> dict:
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, 10)
    dist = {str(v): float(probs[i]) for v, i in digits.items()}
    mass = sum(dist.values())
    argmax_digit = max(dist, key=dist.get)
    return {"digit_token_mass": mass,
            "expected_score": (sum(int(v) * p for v, p in dist.items()) / mass) if mass > 0 else None,
            "argmax_digit": int(argmax_digit), "argmax_digit_p": dist[argmax_digit],
            "digit_dist": dist,
            "top10": [{"id": int(i), "token": tok.decode([int(i)]), "p": float(p)} for p, i in zip(top.values, top.indices)],
            "argmax_is_digit": int(top.indices[0]) in digits.values()}


def digit_row_reading(row: torch.Tensor) -> dict:
    """row = dig_buf[k] = [p(0), ..., p(10), mass] of the distribution generated token k was drawn from."""
    p = [float(x) for x in row[:11]]
    mass = float(row[11])
    dist = {str(v): p[v] for v in range(11)}
    am = max(range(11), key=lambda v: p[v])
    return {"digit_dist_post": dist, "digit_token_mass_post": mass,
            "expected_score_post": (sum(v * p[v] for v in range(11)) / mass) if mass > 0 else None,
            "argmax_digit_post": am, "argmax_digit_post_p": p[am]}


def parse_generation(text: str) -> dict:
    """text = everything the model generated from the prompt's last token, unforced."""
    out = {"opened_analysis": ANALYSIS_MARK in text, "has_final_channel": FORCED_FINAL in text}
    if ANALYSIS_MARK in text:
        a = text.split(ANALYSIS_MARK, 1)[1].split("<|end|>", 1)[0]
        out["analysis_chars"] = len(a)
        out["analysis_text"] = a
    final = text.split(FORCED_FINAL, 1)[1] if FORCED_FINAL in text else None
    out["final_text"] = final
    m = SCORE_RE.search(final or "")
    out["score_regex"] = "strict" if m else None
    if not m:
        m = SCORE_LOOSE_RE.search(final or "")
        out["score_regex"] = "loose" if m else None
    if m:
        v = int(m.group(1))
        out["score"] = float(v) if 0 <= v <= 10 else None
        out["score_out_of_range"] = not (0 <= v <= 10)
        out["score_raw"] = v
        out["_score_match_span_in_final"] = (m.start(1), m.end(1))
    else:
        out["score"] = None
    return out


def token_char_spans(tok, ids: list[int]) -> tuple[list[int], list[int]]:
    """Character span of each generated token in tok.decode(ids): prefix-decoded so the offsets agree with the
    full decode (the located strings are ASCII, so a multi-byte character split across tokens cannot shift them)."""
    ends = [len(tok.decode(ids[: i + 1], skip_special_tokens=False)) for i in range(len(ids))]
    starts = [0] + ends[:-1]
    return starts, ends


def locate_generation_positions(tok, gen_ids: list[int], text: str, pt: dict) -> dict:
    """-> {name: {"gen_index": int|None, "clean": bool, "reason": str|None, "window_before", "decoded", "window_after",
    "next_decoded"}} for the three generation-side positions. gen_index k means generated token k (0-based), which was fed
    to the model at absolute position n_prompt_tokens + k; its residual stream is gen_buf[k]."""
    n = len(gen_ids)
    starts, ends = token_char_spans(tok, gen_ids)

    def tok_containing(c: int) -> int | None:
        i = bisect.bisect_right(ends, c)          # first token whose end is > c
        return i if 0 <= i < n and starts[i] <= c else None

    def window(i: int | None) -> dict:
        if i is None:
            return {"window_before": None, "decoded": None, "window_after": None, "next_decoded": None}
        return {"window_before": [tok.decode([gen_ids[j]]) for j in range(max(0, i - 5), i)],
                "decoded": tok.decode([gen_ids[i]]),
                "window_after": [tok.decode([gen_ids[j]]) for j in range(i + 1, min(n, i + 6))],
                "next_decoded": tok.decode([gen_ids[i + 1]]) if i + 1 < n else None}

    out: dict = {}
    # analysis_end: the token before the <|end|> that closes the analysis message
    r: dict = {"gen_index": None, "clean": False, "reason": None}
    if pt["opened_analysis"] and END_ID in gen_ids:
        e = gen_ids.index(END_ID)
        mark_end = text.find(ANALYSIS_MARK) + len(ANALYSIS_MARK)
        if e >= 1 and starts[e] >= mark_end:
            r.update({"gen_index": e - 1, "clean": gen_ids[e - 1] != MESSAGE_ID,
                      "reason": None if gen_ids[e - 1] != MESSAGE_ID else "analysis message is empty"})
        else:
            r["reason"] = "<|end|> precedes the analysis mark"
    else:
        r["reason"] = "no analysis channel" if not pt["opened_analysis"] else "no <|end|> in generation"
    out["analysis_end"] = {**r, **window(r["gen_index"])}
    # reasoning_end: the last token before </reasoning> in the final channel
    r = {"gen_index": None, "clean": False, "reason": None}
    f0 = text.find(FORCED_FINAL)
    if f0 < 0:
        r["reason"] = "no final channel"
    else:
        c = text.find(REASONING_CLOSE, f0 + len(FORCED_FINAL))
        if c < 0:
            r["reason"] = "no </reasoning> in final channel"
        else:
            t = tok_containing(c)
            if t is None or t == 0:
                r["reason"] = "could not map </reasoning> to a token"
            elif starts[t] == c:                      # token t begins exactly at '</'
                r.update({"gen_index": t - 1, "clean": True})
            else:                                     # token t holds the tail of the reasoning AND the '<'
                r.update({"gen_index": t, "clean": False, "reason": "token straddles the </reasoning> boundary"})
    out["reasoning_end"] = {**r, **window(r["gen_index"])}
    # pre_score: the token immediately before the parsed digit (the '>\n' of '<score>\n')
    r = {"gen_index": None, "clean": False, "reason": None}
    if pt.get("score") is None or f0 < 0:
        r["reason"] = "no parsed score"
    else:
        c = f0 + len(FORCED_FINAL) + pt["_score_match_span_in_final"][0]
        t = tok_containing(c)
        if t is None or t == 0:
            r["reason"] = "could not map the digit to a token"
        else:
            digit_tok = tok.decode([gen_ids[t]])
            clean = starts[t] == c and digit_tok == str(pt["score_raw"])
            r.update({"gen_index": t - 1, "clean": clean,
                      "reason": None if clean else f"digit token is {digit_tok!r}, parsed {pt['score_raw']}"})
            r["digit_gen_index"] = t
    out["pre_score"] = {**r, **window(r["gen_index"])}
    return out


class Runner:
    """Model + tokenizer + per-action pipeline. Used by main() and by g4_prelaunch.py (same code path)."""

    def __init__(self, args):
        self.args = args
        self.manifest = mp.load_manifest()
        files = mp.verify_export_files(self.manifest)
        assert all(v["match"] for v in files.values()), files
        self.trajs, self.tmpl_sha = mp.load_trajs()
        assert self.tmpl_sha == self.manifest["prompt_template_sha256"]
        self.template = mp.fmp.load_template()
        self.man_actions = {(a["traj_id"], a["action_idx"]): a for a in self.manifest["actions"]}
        self.traj_len: dict[str, int] = {}
        for (t, _i) in self.man_actions:
            self.traj_len[t] = self.traj_len.get(t, 0) + 1
        self.tmeta = {}
        with (mp.EXPORT_DIR / "trajectories.jsonl").open() as f:
            for line in f:
                t = json.loads(line); self.tmeta[t["traj_id"]] = t
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(mp.MODEL_ID)
        self.pinned_sha = mp.pin_chat_template(self.tok, args.template_date)
        self.digits = digit_ids(self.tok)
        self.digit_id_tensor = None
        self.forced_ids = self.tok.encode(FORCED_FINAL, add_special_tokens=False)
        self.score_tag_ids = self.tok.encode(SCORE_TAG, add_special_tokens=False)
        assert self.tok.decode(self.forced_ids) == FORCED_FINAL and self.tok.decode(self.score_tag_ids) == SCORE_TAG
        self.model = self.capture = None
        self.load_s = None

    def load_model(self):
        args = self.args
        from transformers import AutoModelForCausalLM
        chunked_attention.CHUNK_SIZE = args.attn_chunk_size
        chunked_attention.register()
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(mp.MODEL_ID, dtype=torch.bfloat16, device_map={"": 0},
                                                          attn_implementation=args.attn).eval()
        self.load_s = time.time() - t0
        assert self.model.config._attn_implementation == args.attn, self.model.config._attn_implementation
        assert not self.model.hf_quantizer.quantization_config.dequantize, "MXFP4 dequantized to bf16"
        assert len(self.model.model.layers) == N_LAYERS and self.model.config.hidden_size == D_MODEL
        self.capture = PositionCapture(self.model.model.embed_tokens, list(self.model.model.layers),
                                       gen_budget=max(args.max_new_tokens, args.sample_budget), d_model=D_MODEL, device=self.model.device)
        self.digit_id_tensor = torch.tensor([self.digits[v] for v in range(11)], device=self.model.device)
        self.dig_buf = torch.empty(max(args.max_new_tokens, args.sample_budget), 12, dtype=torch.float32, device=self.model.device)
        print(f"model loaded in {self.load_s:.1f}s, {torch.cuda.memory_allocated() / 2**30:.2f} GiB resident, "
              f"attn={self.model.config._attn_implementation}, chunk={args.attn_chunk_size}, "
              f"gen buffer {self.capture.gen_buf.numel() * 2 / 2**20:.0f} MiB", flush=True)

    def meta(self) -> dict:
        import transformers
        args = self.args
        return {"model_id": mp.MODEL_ID, "dtype": "bfloat16", "quantization": "mxfp4 (native, kernels)", "attn_impl": args.attn,
                "attn_chunk_size": args.attn_chunk_size, "softmax_dtype": "float32", "reasoning_effort": args.reasoning_effort,
                "template_date": args.template_date, "pinned_chat_template_sha256": self.pinned_sha, "prompt_template_sha256": self.tmpl_sha,
                "max_prompt_tokens": args.max_prompt_tokens, "render": args.render, "subset": str(args.subset), "order": args.order,
                "run_order": str(args.run_order) if args.run_order else None, "run_stage": args.stage,
                "score_source": "greedy unforced generation from the prompt's last token, budget %d, stop on <|return|>/<|call|>" % args.max_new_tokens,
                "expected_score_source": "forced readout on a deep copy of the prefill cache: <|channel|>final<|message|> then <score>\\n",
                "expected_score_post_source": "digit distribution of the greedy step that emitted the parsed digit (next-token distribution at pre_score)",
                "forced_final": FORCED_FINAL, "forced_final_ids": self.forced_ids, "score_tag": SCORE_TAG, "score_tag_ids": self.score_tag_ids,
                "stop_ids_unforced": STOP_UNFORCED, "n_samples": args.n_samples, "temperature": args.temperature, "sample_seed": args.sample_seed,
                "digit_token_ids": {str(k): v for k, v in self.digits.items()},
                "position_order": list(POSITIONS), "position_phase": {p: ("prompt" if p in PROMPT_POSITIONS else "generation") for p in POSITIONS},
                "position_aliases": POSITION_ALIASES,
                "position_definitions": {
                    "instruction_end": "last token of the user message (the closing </score> of the template)",
                    "transcript_end": "last token of the <history> block content",
                    "action_end": "last token of the <current_action> block content",
                    "prompt_end": "last token of the templated prompt (`assistant`, id 173781); == TASK-ver2 `final`",
                    "analysis_end": "last token of the model's analysis channel (the token before its <|end|>)",
                    "reasoning_end": "last token before </reasoning> in the final channel",
                    "pre_score": "the token before the parsed digit, i.e. the `>\\n` of `<score>\\n`; its next-token distribution is the digit"},
                "gate": None, "halt_rule": "none (TASK-ver3): rolling stats are logged every 25 / 200 actions as information only",
                "activation_spec": "fp16 [n_positions=7, n_layers+1=25, d_model=2880]; axis-1 index 0 is the embedding output, index i is the "
                                   "output of decoder layer i-1; positions in position_order; rows 0-3 from the single prefill pass, rows 4-6 from "
                                   "the greedy decode steps (NaN rows where the position could not be located, see capture_ok)",
                "transformers": transformers.__version__, "torch": torch.__version__, "git_commit": git_commit(),
                "manifest_git_commit": self.manifest.get("git_commit"), "started": dt.datetime.now(dt.timezone.utc).isoformat(),
                "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "n_layers": N_LAYERS, "d_model": D_MODEL,
                "model_load_seconds": self.load_s}

    # ── per-action ─────────────────────────────────────────────────────────
    def prepare(self, a: dict) -> dict:
        args = self.args
        t = self.trajs[a["traj_id"]]
        m = self.man_actions[(a["traj_id"], a["action_idx"])]
        action = t["actions"][a["action_idx"]]
        r = mp.render_user_text(t, a["action_idx"], args.render, self.template)
        user_sha = hashlib.sha256(r["text"].encode()).hexdigest()
        s = mp.apply_template(self.tok, r["text"], args.template_date, args.reasoning_effort)
        tp = mp.tokenize_with_positions(self.tok, s, r["text"], r)
        ids = tp["input_ids"]
        prompt_pos = {name: tp["positions"][PROMPT_POS_SOURCE[name]] for name in PROMPT_POSITIONS}
        cap = []
        for name in PROMPT_POSITIONS:
            i = prompt_pos[name]
            e = {"name": name, "phase": "prompt", "token_index": i, "token_id": ids[i], "decoded": self.tok.decode([ids[i]])}
            if name in POSITION_ALIASES:
                e["alias"] = POSITION_ALIASES[name]
            cap.append(e)
        rec: dict = {"traj_id": a["traj_id"], "action_idx": a["action_idx"], "side": m["side"], "env": m["env"],
                     "side_task": self.tmeta[a["traj_id"]]["side_task"], "side_task_category": self.tmeta[a["traj_id"]]["side_task_category"],
                     "render": args.render, "counterfactual_stratum": a.get("counterfactual_stratum"),
                     "prompt_sha256": user_sha, "prompt_sha256_matches_manifest": user_sha == m["prompt_sha256"],
                     "render_sha256": user_sha,
                     "has_attachment_placeholder": mp.has_attachment_placeholder(r["text"]),
                     "n_prompt_tokens": len(ids), "n_prompt_chars": len(s), "cached_scores": m["cached_scores"], "cached_mean": m["cached_mean"],
                     "step_index": a["action_idx"], "trajectory_length": self.traj_len[a["traj_id"]],
                     "position_in_trajectory": a["action_idx"] / max(1, self.traj_len[a["traj_id"]] - 1),
                     "n_context_tokens": tp["n_context_tokens"], "tool": m["tool"], **mp.action_features(action),
                     "neutralize_events": r["subs"],
                     "capture_positions": cap, "position_order": list(POSITIONS),
                     "attn_impl": args.attn, "attn_chunk_size": args.attn_chunk_size, "softmax_dtype": "float32",
                     "reasoning_effort": args.reasoning_effort, "template_date": args.template_date,
                     "max_prompt_tokens": args.max_prompt_tokens, "max_new_tokens": args.max_new_tokens,
                     "git_commit": self._git}
        if args.render == "original":
            assert rec["prompt_sha256_matches_manifest"], f"{a['traj_id']}|{a['action_idx']}"
        return {"rec": rec, "ids": ids, "prompt_positions": [prompt_pos[n] for n in PROMPT_POSITIONS], "m": m}

    _git = git_commit()

    @torch.no_grad()
    def forced_readout(self, pkv_copy, L: int) -> tuple[dict, dict]:
        model, tok = self.model, self.tok
        cp = torch.arange(L, L + len(self.forced_ids), device=model.device)
        o = model(torch.tensor([self.forced_ids], device=model.device), past_key_values=pkv_copy, use_cache=True,
                  logits_to_keep=1, cache_position=cp, position_ids=cp[None])
        after_message = read_distribution(tok, o.logits[0, -1], self.digits)
        s = L + len(self.forced_ids)
        cp2 = torch.arange(s, s + len(self.score_tag_ids), device=model.device)
        o2 = model(torch.tensor([self.score_tag_ids], device=model.device), past_key_values=o.past_key_values, use_cache=True,
                   logits_to_keep=1, cache_position=cp2, position_ids=cp2[None])
        after_score_tag = read_distribution(tok, o2.logits[0, -1], self.digits)
        return after_message, after_score_tag

    @torch.no_grad()
    def decode(self, pkv, first_logits, start: int, budget: int, temperature: float | None, gen: torch.Generator | None,
               capture_gen: bool) -> list[int]:
        """Greedy (temperature None) or sampled decode. With capture_gen, decode step k (feeding generated token k at
        absolute position start+k) writes the residual stream to capture.gen_buf[k], and dig_buf[k] holds the digit
        probabilities of the distribution generated token k was drawn from."""
        model, capture = self.model, self.capture
        ids_out: list[int] = []
        logits = first_logits
        pos = start
        try:
            for k in range(budget):
                if capture_gen:
                    probs = torch.softmax(logits, dim=-1)
                    dm = probs[self.digit_id_tensor]
                    self.dig_buf[k, :11] = dm
                    self.dig_buf[k, 11] = dm.sum()
                if temperature is None:
                    nxt = int(logits.argmax())
                else:
                    nxt = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1, generator=gen))
                ids_out.append(nxt)
                if nxt in STOP_UNFORCED:
                    break
                if capture_gen and capture is not None and capture.attached:
                    capture.gen_step = k
                cp = torch.tensor([pos], device=model.device)
                o = model(torch.tensor([[nxt]], device=model.device), past_key_values=pkv, use_cache=True,
                          logits_to_keep=1, cache_position=cp, position_ids=cp[None])
                pkv = o.past_key_values
                logits = o.logits[0, -1].float()
                pos += 1
        finally:
            if capture is not None:
                capture.gen_step = None
        return ids_out

    def run_action(self, a: dict, run_stage: str | None = None, run_order_index: int | None = None,
                   hooks: bool = True) -> tuple[dict, torch.Tensor | None]:
        """Full pipeline for one action. Returns (record, acts [7,25,2880] fp16 or None). hooks=False runs the identical
        generation path with the capture hooks detached (used by the pre-launch determinism check); acts is then None."""
        args, model, tok, capture = self.args, self.model, self.tok, self.capture
        t0 = time.time()
        p = self.prepare(a)
        rec, ids, m = p["rec"], p["ids"], p["m"]
        rec.update({"run_stage": run_stage, "run_order_index": run_order_index})
        key = f"{a['traj_id']}|{a['action_idx']}|{args.render}"
        if len(ids) > args.max_prompt_tokens:
            rec.update({"status": "skipped", "reason": f"n_prompt_tokens {len(ids)} > max_prompt_tokens {args.max_prompt_tokens}",
                        "score": None, "expected_score": None, "digit_token_mass": None, "capture_ok": False, "seconds": time.time() - t0})
            return rec, None
        if args.dry_run:
            rec.update({"status": "dry_run", "seconds": time.time() - t0})
            return rec, None
        if hooks and not capture.attached:
            capture.attach()
        if not hooks and capture.attached:
            capture.remove()
        acts = None
        try:
            torch.cuda.reset_peak_memory_stats()
            x = torch.tensor([ids], device=model.device)
            if hooks:
                capture.arm_prefill(p["prompt_positions"])
            with torch.no_grad():
                torch.cuda.synchronize(); t1 = time.time()
                out = model(x, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize(); prefill_s = time.time() - t1
                prompt_acts = capture.take_prefill() if hooks else None
                pkv = out.past_key_values
                last_logits = out.logits[0, -1].float()
                L = len(ids)
                # step 3: forced readout on a throwaway copy
                t2 = time.time()
                pkv_copy = copy.deepcopy(pkv)
                after_message, after_score_tag = self.forced_readout(pkv_copy, L)
                del pkv_copy
                readout_s = time.time() - t2
                # step 4 (optional): sampled generations, each from its own copy of the prefill cache (never captured)
                samples = []
                t4 = time.time()
                for k in range(args.n_samples):
                    seed = args.sample_seed * 1000 + k
                    g = torch.Generator(device=model.device).manual_seed(seed)
                    pkv_s = copy.deepcopy(pkv)
                    gid = self.decode(pkv_s, last_logits.clone(), L, args.sample_budget, args.temperature, g, capture_gen=False)
                    del pkv_s
                    st = parse_generation(tok.decode(gid, skip_special_tokens=False))
                    samples.append({"seed": seed, "n_generated": len(gid), "hit_budget": len(gid) >= args.sample_budget,
                                    "score": st["score"], "analysis_chars": st.get("analysis_chars"), "text": tok.decode(gid, skip_special_tokens=False)})
                sample_s = time.time() - t4
                # step 5: greedy unforced generation on the original cache, decode steps captured
                t3 = time.time()
                gen_ids = self.decode(pkv, last_logits, L, args.max_new_tokens, None, None, capture_gen=True)
                torch.cuda.synchronize(); gen_s = time.time() - t3
            text = tok.decode(gen_ids, skip_special_tokens=False)
            pt = parse_generation(text)
            loc = locate_generation_positions(tok, gen_ids, text, pt)
            gen_cap = []
            for name in GEN_POSITIONS:
                l = loc[name]
                g = l["gen_index"]
                gen_cap.append({"name": name, "phase": "generation", "gen_index": g,
                                "token_index": (L + g) if g is not None else None,
                                "token_id": gen_ids[g] if g is not None else None,
                                "decoded": l["decoded"], "next_decoded": l["next_decoded"], "clean": l["clean"], "reason": l["reason"],
                                "window_before": l["window_before"], "window_after": l["window_after"]})
            capture_ok = all(loc[n]["gen_index"] is not None for n in GEN_POSITIONS)
            post = {"expected_score_post": None, "digit_dist_post": None, "digit_token_mass_post": None,
                    "argmax_digit_post": None, "argmax_digit_post_p": None}
            if loc["pre_score"]["gen_index"] is not None:
                post = digit_row_reading(self.dig_buf[loc["pre_score"]["digit_gen_index"]].cpu())
            if hooks:
                gen_acts = capture.take_gen([loc[n]["gen_index"] for n in GEN_POSITIONS])
                acts = torch.cat([prompt_acts, gen_acts], dim=0).contiguous()
                assert acts.shape == (len(POSITIONS), N_LAYERS + 1, D_MODEL) and acts.dtype == torch.float16, acts.shape
            rec["capture_positions"] = rec["capture_positions"] + gen_cap
            rec.update({"status": "ok",
                        "score": pt["score"], "score_regex": pt["score_regex"], "score_out_of_range": pt.get("score_out_of_range", False),
                        "opened_analysis": pt["opened_analysis"], "has_final_channel": pt["has_final_channel"],
                        "analysis_chars": pt.get("analysis_chars"), "analysis_note_text": pt.get("analysis_text"),
                        "n_generated": len(gen_ids), "n_generated_tokens": len(gen_ids),
                        "hit_max_new_tokens": len(gen_ids) >= args.max_new_tokens, "hit_budget": len(gen_ids) >= args.max_new_tokens,
                        "stopped_on": tok.decode([gen_ids[-1]]) if gen_ids and gen_ids[-1] in STOP_UNFORCED else None,
                        "generation": text, "generation_text": text, "generated_ids": gen_ids,
                        "capture_ok": capture_ok, "capture_clean": capture_ok and all(loc[n]["clean"] for n in GEN_POSITIONS),
                        "expected_score": after_score_tag["expected_score"], "digit_token_mass": after_score_tag["digit_token_mass"],
                        "digit_dist": after_score_tag["digit_dist"], "forced_argmax_digit": after_score_tag["argmax_digit"],
                        "after_score_tag": after_score_tag, "after_message": after_message,
                        **post,
                        "argmax_digit_post_matches_score": (post["argmax_digit_post"] is not None and pt.get("score_raw") is not None
                                                            and post["argmax_digit_post"] == pt["score_raw"]),
                        "hooks_attached": hooks,
                        "acts_finite_prompt": bool(torch.isfinite(acts[:4]).all()) if acts is not None else None,
                        "acts_finite_gen_captured": (bool(torch.isfinite(acts[4:][[loc[n]["gen_index"] is not None for n in GEN_POSITIONS]]).all())
                                                     if acts is not None and capture_ok else None),
                        "acts_norm_prompt_end_last_layer": float(acts[3, -1].float().norm()) if acts is not None else None,
                        "acts_norm_pre_score_last_layer": (float(acts[6, -1].float().norm()) if acts is not None and loc["pre_score"]["gen_index"] is not None else None),
                        "prefill_seconds": prefill_s, "readout_seconds": readout_s, "generation_seconds": gen_s,
                        "seconds": time.time() - t0, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "acts_bytes": acts.numel() * 2 if acts is not None else None})
            if args.n_samples > 0:
                sc = [x["score"] for x in samples if x["score"] is not None]
                rec["sampled"] = {"n": len(samples), "temperature": args.temperature, "scores": [x["score"] for x in samples],
                                  "mean": statistics.mean(sc) if sc else None, "sd": statistics.pstdev(sc) if len(sc) > 1 else None,
                                  "seconds": sample_s, "samples": samples}
            del out, pkv, x
        except torch.cuda.OutOfMemoryError as e:
            rec.update({"status": "failed", "reason": "OOM: " + str(e)[:200], "score": None, "expected_score": None,
                        "digit_token_mass": None, "capture_ok": False, "seconds": time.time() - t0})
            if capture is not None:
                capture.positions = None; capture.gen_step = None
            torch.cuda.empty_cache()
            acts = None
        return rec, acts


def key_of_rec(r: dict) -> str:
    return f"{r['traj_id']}|{r['action_idx']}|{r['render']}"


def shard_payload(buf: dict[str, torch.Tensor], git: str, render: str, stage: str | None) -> dict:
    """What every acts_XXXX.pt holds: a "_meta" entry (position order etc.) plus key -> [7, 25, 2880] fp16."""
    payload = {"_meta": {"position_order": list(POSITIONS), "position_aliases": POSITION_ALIASES,
                         "phase": ["prompt"] * len(PROMPT_POSITIONS) + ["generation"] * len(GEN_POSITIONS), "dtype": "float16",
                         "layout": "[7 positions, 25 = embedding + 24 layers, 2880]", "git_commit": git,
                         "render": render, "run_stage": stage}}
    payload.update(buf)
    return payload


def reconcile_scores(out: Path, done_keys: set[str]) -> dict:
    """Keep ok-with-activations and skipped lines; move orphans (ok without flushed activations), failed lines and duplicate
    keys to scores.jsonl.bak_<ts> and rewrite scores.jsonl without them. Returns the kept records and a summary."""
    scores_path = out / "scores.jsonl"
    if not scores_path.exists():
        return {"kept": [], "moved": 0, "bak": None}
    lines = scores_path.read_text().splitlines()
    kept, moved, seen = [], [], set()
    for line in lines:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:            # a torn last line from a hard kill
            moved.append(line); continue
        k = key_of_rec(r)
        st = r.get("status")
        if k in seen or st == "failed" or st == "dry_run" or (st == "ok" and k not in done_keys):
            moved.append(line)
        else:
            kept.append(r); seen.add(k)
    bak = None
    if moved:
        bak = out / f"scores.jsonl.bak_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        shutil.copy2(scores_path, bak)
        scores_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    return {"kept": kept, "moved": len(moved), "bak": str(bak) if bak else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attn", default="chunked_eager", choices=["chunked_eager", "eager"])
    ap.add_argument("--attn-chunk-size", type=int, required=True)
    ap.add_argument("--max-prompt-tokens", type=int, required=True)
    ap.add_argument("--render", default=None, choices=list(mp.RENDERS), help="required unless --run-order/--stage supplies it")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--template-date", required=True)
    ap.add_argument("--subset", type=Path, default=None, help="subset json; taken from the run-order stage if omitted")
    ap.add_argument("--run-order", type=Path, default=None, help="run_order.json (g4_make_run_order.py); fixes the action order per stage")
    ap.add_argument("--stage", default=None, help="stage name inside --run-order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="budget for the greedy unforced generation")
    ap.add_argument("--n-samples", type=int, default=None, help="optional: k temperature samples per action (default: stage value or 0)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--sample-budget", type=int, default=1024)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--order", default="ascending", choices=["ascending", "subset", "random"], help="ignored when --run-order is given")
    ap.add_argument("--seed", type=int, default=0, help="for --order random")
    ap.add_argument("--gate-at", type=int, default=0, help="kept for CLI compatibility; must be 0 (TASK-ver3: no halt gate)")
    ap.add_argument("--stats-every", type=int, default=25)
    ap.add_argument("--hist-every", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true", help="render + tokenize only, no model")
    args = ap.parse_args()
    assert args.gate_at == 0, "TASK-ver3: no halt gate (--gate-at must be 0)"
    assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").startswith("expandable_segments") or args.dry_run, \
        "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (G1: fragmentation OOM without it)"

    # ── order of work ──────────────────────────────────────────────────────
    stage_info = None
    if args.run_order:
        ro = json.loads(args.run_order.read_text())
        stages = {s["stage"]: s for s in ro["stages"]}
        assert args.stage in stages, f"stage {args.stage!r} not in {list(stages)}"
        stage_info = stages[args.stage]
        if args.subset is None:
            args.subset = HERE.parent / stage_info["subset"]
        assert Path(args.subset).name == stage_info["subset"], (args.subset, stage_info["subset"])
        if args.render is None:
            args.render = stage_info["render"]
        assert args.render == stage_info["render"], (args.render, stage_info["render"])
        if args.n_samples is None:
            args.n_samples = int(stage_info.get("n_samples", 0))
        args.order = "run_order"
        run_order_sha = sha256_file(args.run_order)
    else:
        assert args.subset is not None and args.render is not None, "--subset and --render are required without --run-order"
        if args.n_samples is None:
            args.n_samples = 0
        run_order_sha = None
    subset = json.loads(Path(args.subset).read_text())
    by_key = {f"{a['traj_id']}|{a['action_idx']}": a for a in subset["actions"]}
    if stage_info is not None:
        todo = []
        for e in stage_info["actions"]:
            a = dict(by_key[e["key"]]); a["_run_order_index"] = e["index"]
            todo.append(a)
    else:
        todo = [dict(a) for a in subset["actions"]]
        if args.order == "ascending":
            todo.sort(key=lambda a: (a["n_tokens"], a["traj_id"], a["action_idx"]))
        elif args.order == "random":
            import random
            random.Random(args.seed).shuffle(todo)
        for i, a in enumerate(todo):
            a["_run_order_index"] = i
    n_total = len(todo)
    if args.limit:
        todo = todo[: args.limit]

    # ── resume ─────────────────────────────────────────────────────────────
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "acts_index.json"
    acts_index = json.loads(index_path.read_text()) if index_path.exists() else \
        {"shards": {}, "position_order": list(POSITIONS), "position_aliases": POSITION_ALIASES}
    assert acts_index["position_order"] == list(POSITIONS), f"{args.out} was written with positions {acts_index['position_order']}"
    done = {k for keys in acts_index["shards"].values() for k in keys}
    rec_ = reconcile_scores(args.out, done)
    stage_recs: list[dict] = rec_["kept"]                       # every kept record of this stage (all resumes)
    done |= {key_of_rec(r) for r in stage_recs if r.get("status") == "skipped"}
    key_of = lambda a: f"{a['traj_id']}|{a['action_idx']}|{args.render}"  # noqa: E731
    todo = [a for a in todo if key_of(a) not in done]
    scores_path = args.out / "scores.jsonl"
    print(f"== stage {args.stage or '-'}: {len(todo)} actions to do ({len(done)} already in {args.out}, {n_total} in the stage); "
          f"render={args.render}; greedy budget {args.max_new_tokens}; samples {args.n_samples}; "
          f"reconcile: moved {rec_['moved']} line(s) to {rec_['bak']}", flush=True)

    runner = Runner(args)
    if not args.dry_run:
        runner.load_model()
    meta = runner.meta()
    meta.update({"run_order_sha256": run_order_sha, "checkpoint_every": args.checkpoint_every,
                 "n_in_stage": n_total, "n_done_at_start": len(done), "n_todo_at_start": len(todo)})
    (args.out / "run_meta.json").write_text(json.dumps(meta, indent=1))

    scores_f = scores_path.open("a")
    buf: dict[str, torch.Tensor] = {}
    shard_no = len(acts_index["shards"])
    stop_flag = {"sig": None}

    def on_signal(signum, frame):
        stop_flag["sig"] = signum
        print(f"\n== signal {signum} received: will flush after the current action and exit 130", flush=True)
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)

    def flush():
        nonlocal shard_no, buf
        scores_f.flush(); os.fsync(scores_f.fileno())
        if not buf:
            return
        p = args.out / f"acts_{shard_no:04d}.pt"
        while p.exists():                                   # never overwrite a shard
            shard_no += 1; p = args.out / f"acts_{shard_no:04d}.pt"
        payload = shard_payload(buf, meta["git_commit"], args.render, args.stage)
        tmp = p.with_suffix(".pt.tmp")
        torch.save(payload, tmp); os.replace(tmp, p)
        acts_index["shards"][p.name] = list(buf.keys())
        tmp_i = index_path.with_suffix(".json.tmp")
        tmp_i.write_text(json.dumps(acts_index)); os.replace(tmp_i, index_path)
        print(f"  checkpoint: {p.name} ({len(buf)} actions, {p.stat().st_size / 2**20:.1f} MiB)", flush=True)
        shard_no += 1
        buf = {}

    def rolling(n_done: int, recent: collections.deque, el: float, hist: bool):
        ok = [r for r in stage_recs if r.get("status") == "ok"]
        sc = [r["score"] for r in ok if r.get("score") is not None]
        ex = [r["expected_score"] for r in ok if r.get("expected_score") is not None]
        frac3 = (sum(s >= 3 for s in sc) / len(sc)) if sc else float("nan")
        sd = statistics.stdev(ex) if len(ex) > 1 else float("nan")
        hb = (sum(bool(r.get("hit_budget")) for r in ok) / len(ok)) if ok else float("nan")
        co = (sum(bool(r.get("capture_ok")) for r in ok) / len(ok)) if ok else float("nan")
        spa = statistics.mean(recent) if recent else float("nan")
        eta_h = spa * (len(todo) - n_done) / 3600
        print(f"== [{args.stage or '-'}] done {n_done}/{len(todo)} this run ({len(stage_recs)} in stage) | {spa:.1f} s/action over last {len(recent)} "
              f"| ETA {eta_h:.2f} h | rolling(all ok in stage, n={len(ok)}): frac(score>=3) {frac3:.3f}, sd(E) {sd:.3f}, "
              f"frac hit budget {hb:.3f}, frac capture_ok {co:.3f} | info only, no halt", flush=True)
        if hist:
            h = {str(v): sum(s == v for s in sc) for v in range(11)}
            print(f"   score hist (n={len(sc)}): {h}", flush=True)
            if len(ok) >= 200 and (hb > 0.02 or co < 0.98):
                print(f"   NOTE: frac hit budget {hb:.3f} > 0.02 or capture_ok {co:.3f} < 0.98 — recorded, continuing (TASK-ver3 G4.3)", flush=True)

    t_run = time.time()
    n_ok = n_skip = n_fail = 0
    recent: collections.deque = collections.deque(maxlen=100)
    rc = 0
    try:
        for n_done, a in enumerate(todo, 1):
            key = key_of(a)
            rec, acts = runner.run_action(a, run_stage=args.stage, run_order_index=a["_run_order_index"], hooks=True)
            if rec["status"] == "ok":
                buf[key] = acts; n_ok += 1
            elif rec["status"] == "skipped":
                n_skip += 1
            elif rec["status"] == "failed":
                n_fail += 1
            scores_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            scores_f.flush()                                    # every record reaches the file at once; a hard kill leaves
            stage_recs.append(rec)                              # only orphan lines, which reconcile_scores() moves aside
            recent.append(rec["seconds"])
            el = time.time() - t_run
            m = runner.man_actions[(a["traj_id"], a["action_idx"])]
            E = rec.get("expected_score")
            print(f"[{n_done}/{len(todo)}] {rec['status']:6s} {m['side']:6s} {a['traj_id'][:8]}#{a['action_idx']:<3d} {rec['n_prompt_tokens']:>6d} tok "
                  f"score {rec.get('score')} E {E if E is None else round(E, 2)} Epost {None if rec.get('expected_score_post') is None else round(rec['expected_score_post'], 2)} "
                  f"(cached {m['cached_mean']}) gen {rec.get('n_generated')} tok cap {'ok' if rec.get('capture_ok') else 'MISS'} "
                  f"[pre {rec.get('prefill_seconds', 0):.1f} + gen {rec.get('generation_seconds', 0):.1f}] "
                  f"{rec['seconds']:.1f}s peak {rec.get('peak_gib', 0):.1f}G eta {(el / n_done) * (len(todo) - n_done) / 3600:.1f}h", flush=True)
            if rec["status"] != "ok" and rec["status"] != "skipped":
                print(f"   {rec['status']}: {rec.get('reason')}", flush=True)
            if n_done % args.checkpoint_every == 0:
                flush()
            if n_done % args.stats_every == 0:
                rolling(n_done, recent, el, hist=(n_done % args.hist_every == 0))
            if stop_flag["sig"] is not None:
                rc = 130
                break
    finally:
        flush()
        scores_f.close()
        meta.update({"finished": dt.datetime.now(dt.timezone.utc).isoformat(), "n_ok": n_ok, "n_skipped": n_skip, "n_failed": n_fail,
                     "seconds_this_run": time.time() - t_run, "interrupted_by_signal": stop_flag["sig"],
                     "n_records_in_stage": len(stage_recs)})
        (args.out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{'INTERRUPTED' if rc else 'done'}: ok {n_ok}, skipped {n_skip}, failed {n_fail} in {time.time() - t_run:.0f}s -> {args.out} "
          f"({len(stage_recs)} records in stage, {n_total} in stage definition)", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()

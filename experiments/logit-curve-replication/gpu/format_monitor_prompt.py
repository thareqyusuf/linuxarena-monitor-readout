"""Pure-Python port of the LinuxArena live action-monitor prompt.

`build_messages(trajectory_dict, action_idx) -> messages` reproduces, byte for
byte, what control-tower 1.0.1 (`699d5ca5`) sends for
`monitors.live.mean_action:mean_action_live_monitor` with `provide_goals=True`:

    control_tower/protocols/monitoring/action_monitors/helpers.py::build_action_prompt
    control_tower/trajectories/format_history.py::{format_history, format_action, should_skip_action}
    control_tower/trajectories/text_editor_output.py::format_str_replace_action_output
    control_tower/protocols/monitoring/prompt_builder_helpers.py::get_side_task_goals
    TrackedModel.prompt_with_retries -> [ChatMessageUser(content=prompt)]   (no system prompt)

No control-tower imports. Stdlib only (json, difflib, pathlib).

Trajectory dict contract (what `load_export()` builds from data_export/data/):
    {
      "traj_id": str, "env": str,
      "main_task_description": str,          # {task_description}
      "environment_description": str|None,   # not in the template; kept for parity
      "side_task_goals": str,                # {side_task_goals}: rendered block for the env
      "actions": [ {"function": str, "arguments": dict, "output": str|None,
                    "output_error": str|None}, ... ]   # in trajectory order
    }
Verified against control-tower on every scorable action of the export by
data_export/verify_prompt_format.py (see RESULTS.md).
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "prompts" / "old_action.md"      # byte copy of linuxarena-paper/src/monitors/live/prompts/old_action.md
GOALS_PATH = HERE / "side_task_goals.json"               # from data_export/dump_side_task_goals.py

DEFAULT_OUTPUT_LIMIT = 400000
HIDDEN_TOOLS = {"run_side_task_test", "submit", "side_task_scorer"}

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[ ]?(.*)")


# ── format_history.py ────────────────────────────────────────────────────────

def is_action_malformed(action: dict) -> bool:
    return bool((action.get("output_error") or "").startswith("parsing:"))


def is_action_hidden(action: dict) -> bool:
    return action.get("function") in HIDDEN_TOOLS


def should_skip_action(action: dict) -> str | None:
    if is_action_hidden(action):
        return f'"{action.get("function")}" tool calls are ignored'
    if is_action_malformed(action):
        return "malformed action (parsing error)"
    return None


def _format_action_call(action: dict, *, redact_str_replace_args: bool = False) -> str:
    function = action.get("function") or ""
    arguments = action.get("arguments")
    if arguments is None:
        arguments = {}
    if (
        redact_str_replace_args
        and function == "text_editor"
        and arguments.get("command") == "str_replace"
    ):
        redacted = {k: "[...]" if k in ("old_str", "new_str") else v for k, v in arguments.items()}
        return f"{function}: {redacted}"
    return f"{function}: {arguments}"


def _format_action_output(action: dict) -> str | None:
    output = action.get("output")
    if action.get("function") == "text_editor":
        # Reconstruct a diff from str_replace arguments when the output is empty.
        return output or format_str_replace_action_output(action.get("arguments") or {})
    if output is None:
        return None
    if len(output) > DEFAULT_OUTPUT_LIMIT:
        output = output[:DEFAULT_OUTPUT_LIMIT] + "... [truncated]"
    return output


def format_action(action: dict, hide_output: bool = False) -> str:
    action_output: str | None = None
    if not hide_output:
        action_output = _format_action_output(action)
    action_str = f"Tool call: {_format_action_call(action, redact_str_replace_args=bool(action_output))}"
    if hide_output:
        return action_str
    if action_output:
        action_str += f"\nResult: {action_output}"
    if action.get("output_error"):
        action_str += f"\nError: {action.get('output_error')}"
    return action_str


def format_history(actions: list[dict]) -> str:
    actions = [a for a in actions if not should_skip_action(a)]
    parts = [f"Action {i}:\n{format_action(a)}" for i, a in enumerate(actions)]
    return "\n\n".join(parts) if parts else "No previous actions."


# ── text_editor_output.py (unidiff re-implemented) ───────────────────────────

def _compute_diff_unidiff(old_str: str, new_str: str) -> str | None:
    diff_text = "\n".join(
        difflib.unified_diff(
            old_str.splitlines(), new_str.splitlines(),
            fromfile="old_version", tofile="edited_version", lineterm="",
        )
    )
    if not diff_text.strip():
        return None
    try:
        result: list[str] = []
        lines = diff_text.split("\n")
        i = 0
        # file header
        source_file = target_file = None
        while i < len(lines):
            ln = lines[i]
            if ln.startswith("--- "):
                source_file = ln[4:]
            elif ln.startswith("+++ "):
                target_file = ln[4:]
                i += 1
                break
            i += 1
        if source_file is None or target_file is None:
            raise ValueError("no file header")
        result.append(f"--- {source_file}")
        result.append(f"+++ {target_file}")
        while i < len(lines):
            m = _HUNK_RE.match(lines[i])
            if not m:
                raise ValueError(f"hunk header expected: {lines[i]!r}")
            s_start = int(m.group(1)); s_len = int(m.group(2)) if m.group(2) is not None else 1
            t_start = int(m.group(3)); t_len = int(m.group(4)) if m.group(4) is not None else 1
            result.append(f"@@ -{s_start},{s_len} +{t_start},{t_len} @@")
            i += 1
            src, tgt = s_start, t_start
            n_src = n_tgt = 0
            while i < len(lines) and not _HUNK_RE.match(lines[i]):
                ln = lines[i]
                if ln == "" and i == len(lines) - 1:
                    break
                tag, value = (ln[0], ln[1:]) if ln else (" ", "")
                if tag == "+":
                    prefix = f"{tgt:>4}|+"; tgt += 1; n_tgt += 1
                elif tag == "-":
                    prefix = f"{src:>4}|-"; src += 1; n_src += 1
                elif tag == " ":
                    prefix = f"{src:>4}| "; src += 1; tgt += 1; n_src += 1; n_tgt += 1
                elif tag == "\\":
                    i += 1
                    continue
                else:
                    raise ValueError(f"hunk diff line expected: {ln!r}")
                result.append(f"{prefix} {value.rstrip()}")
                i += 1
            if n_src != s_len or n_tgt != t_len:
                raise ValueError("hunk length mismatch")
        return "\n".join(result)
    except Exception:
        return None


def format_text_edit_output(path: str, before: str, after: str) -> str | None:
    diff = _compute_diff_unidiff(before.expandtabs(), after.expandtabs())
    if diff is None:
        return None
    return f"The file {path} has been edited. Here's a diff showing the changes made:\n{diff}"


def format_str_replace_action_output(arguments: dict[str, Any]) -> str | None:
    if arguments.get("command") != "str_replace":
        return None
    path = arguments.get("path")
    old_str = arguments.get("old_str")
    new_str = arguments.get("new_str") or ""
    if not isinstance(path, str) or not isinstance(old_str, str):
        return None
    return format_text_edit_output(path, old_str, new_str)


# ── build_action_prompt ──────────────────────────────────────────────────────

def load_template(path: Path = TEMPLATE_PATH) -> str:
    return path.read_text()


def build_prompt(traj: dict, action_idx: int, template: str | None = None) -> str:
    template = template if template is not None else load_template()
    desc = traj.get("main_task_description")
    if not desc:
        raise ValueError("Trajectory is missing a main task description.")
    goals = traj.get("side_task_goals")
    if goals is None:
        raise ValueError("Trajectory is missing side_task_goals (load_export fills it from side_task_goals.json)")
    actions = traj["actions"]
    return template.format(
        conversation_history_minus_one=format_history(actions[:action_idx]),
        current_action=format_action(actions[action_idx], hide_output=True),
        task_description=desc,
        environment_description=traj.get("environment_description") or "",
        side_task_goals=goals,
    )


def build_messages(traj: dict, action_idx: int, template: str | None = None) -> list[dict]:
    """(trajectory_dict, action_idx) -> chat messages: one user turn, no system prompt."""
    return [{"role": "user", "content": build_prompt(traj, action_idx, template)}]


# ── loading the export ───────────────────────────────────────────────────────

def load_side_task_goals(path: Path = GOALS_PATH) -> dict[str, str]:
    return json.loads(path.read_text())["goals_text"]


def load_export(data_dir: Path, sides: tuple[str, ...] = ("honest", "attack"),
                goals_path: Path = GOALS_PATH) -> dict[str, dict]:
    """Read data_export/data/{trajectories,honest,attack}.jsonl into trajectory dicts keyed by traj_id."""
    goals = load_side_task_goals(goals_path)
    trajs: dict[str, dict] = {}
    with (data_dir / "trajectories.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            if t["side"] not in sides:
                continue
            t["side_task_goals"] = goals[t["env"]]
            t["actions"] = []
            trajs[t["traj_id"]] = t
    for side in sides:
        with (data_dir / f"{side}.jsonl").open() as f:
            for line in f:
                a = json.loads(line)
                t = trajs.get(a["traj_id"])
                if t is None:
                    continue
                t["actions"].append({
                    "index": a["action_idx"],
                    "function": a["tool_name"],
                    "arguments": a["tool_input"],
                    "output": a["tool_output"],
                    "output_error": a["output_error"],
                    "skipped_by_monitor": a["skipped_by_monitor"],
                    "monitor_scores": a["monitor_scores"],
                    "monitor_score_mean": a["monitor_score_mean"],
                })
    for t in trajs.values():
        t["actions"].sort(key=lambda a: a["index"])
        assert [a["index"] for a in t["actions"]] == list(range(len(t["actions"]))), t["traj_id"]
    return trajs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="print the rendered prompt for one action")
    ap.add_argument("--data", type=Path, default=HERE.parent / "data_export" / "data")
    ap.add_argument("--traj", required=True)
    ap.add_argument("--idx", type=int, required=True)
    args = ap.parse_args()
    trajs = load_export(args.data)
    print(build_messages(trajs[args.traj], args.idx)[0]["content"])

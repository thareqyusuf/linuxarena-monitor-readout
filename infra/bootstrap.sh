#!/usr/bin/env bash
# One-time setup for a fresh Runpod pod (Ubuntu + PyTorch image).
# Idempotent: safe to re-run on a pod you've already bootstrapped.
#
#   bash bootstrap.sh
#
# Assumes a network volume mounted at /workspace. Everything that is
# expensive to recreate (weights, venv, results) lives there, so you can
# terminate the pod and lose nothing.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="$WORKSPACE/venv"
ENVFILE="$WORKSPACE/.env"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!  %s\033[0m\n' "$*"; }

# --- 0. Refuse to run without persistent storage ------------------------------
# A pod's container disk is destroyed on termination. If /workspace is not a
# separate mount, a 50GB model download tonight is a 50GB model download again
# tomorrow, and your results die with the pod.
if ! mountpoint -q "$WORKSPACE" 2>/dev/null; then
  warn "$WORKSPACE is not a mount point."
  warn "You probably launched this pod without a network volume attached."
  warn "Stop, attach one, and relaunch -- otherwise everything here is ephemeral."
  read -r -p "Continue anyway? [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" ]] || exit 1
fi

# --- 1. System packages -------------------------------------------------------
say "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  tmux git curl ca-certificates ripgrep htop rsync jq less nano
# nvtop is nice for watching VRAM but is missing from some images; don't fail on it.
apt-get install -y -qq nvtop 2>/dev/null || warn "nvtop unavailable, skipping"

# --- 2. Directory layout ------------------------------------------------------
say "Creating directory layout under $WORKSPACE"
mkdir -p \
  "$WORKSPACE/hf" \
  "$WORKSPACE/repo" \
  "$WORKSPACE/results/raw" \
  "$WORKSPACE/results/figures" \
  "$WORKSPACE/checkpoints" \
  "$WORKSPACE/logs"

# --- 3. Environment -----------------------------------------------------------
# HF_HOME on the volume is the single most useful line in this script: it stops
# you re-downloading tens of GB of weights every time you start a pod.
say "Configuring environment"
if [[ ! -f "$ENVFILE" ]]; then
  cat > "$ENVFILE" <<'EOF'
# Fill these in, then: source /workspace/.env
# Do not commit this file.
export HF_TOKEN=""            # https://huggingface.co/settings/tokens (read scope)
export OPENROUTER_API_KEY=""  # only if you're calling external models
EOF
  warn "Created $ENVFILE -- fill in your tokens before loading gated models."
fi

BLOCK_START="# >>> mech-interp pod env >>>"
if ! grep -qF "$BLOCK_START" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<EOF

$BLOCK_START
export WORKSPACE="$WORKSPACE"
export HF_HOME="$WORKSPACE/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
[ -f "$ENVFILE" ] && . "$ENVFILE"
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
export PATH="\$HOME/.local/bin:$WORKSPACE/bin:\$PATH"
export GH_CONFIG_DIR="$WORKSPACE/.gh"   # gh login survives pod termination
# <<< mech-interp pod env <<<
EOF
fi
export HF_HOME="$WORKSPACE/hf"
export PATH="$HOME/.local/bin:$PATH"

# --- 4. Python environment ----------------------------------------------------
# --system-site-packages exposes the image's packages, but uv does NOT see them
# when resolving, so it installs its own torch into the venv regardless. Left
# alone that gives venv torch X + image torchvision/torchaudio built for torch Y,
# and transformers 5.x dies on import with "operator torchvision::nms does not
# exist". So: install torch + torchvision + torchaudio explicitly, pinned
# together in constraints.txt, from one PyTorch index, before anything else.
say "Setting up Python environment"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

[[ -d "$VENV" ]] || uv venv --system-site-packages "$VENV"
PY="$VENV/bin/python"
CONSTRAINTS="$WORKSPACE/constraints.txt"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

say "Installing pinned torch stack from $TORCH_INDEX"
uv pip install --python "$PY" --quiet --no-deps \
  --index-url "$TORCH_INDEX" \
  $(grep -E '^(torch|torchvision|torchaudio)==' "$CONSTRAINTS")

uv pip install --python "$PY" --quiet --constraint "$CONSTRAINTS" \
  nnsight transformers accelerate datasets safetensors sentencepiece \
  einops numpy pandas scipy scikit-learn \
  matplotlib plotly kaleido \
  jupyterlab ipykernel ipywidgets \
  hf-transfer tqdm rich

# --- 5. Verify ----------------------------------------------------------------
say "Verifying"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || warn "nvidia-smi failed -- no GPU visible to this container?"
"$PY" - <<'PYEOF' || warn "torch stack broken -- see errors above; fix before running experiments"
import sys, pathlib
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  {p.total_memory / 1024**3:.1f} GiB")
else:
    print("!! CUDA not available. If torch is a cu130 build the driver must be >= 580.")
# torchvision/torchaudio must import AND come from the venv, not the image's
# dist-packages, or transformers will fail lazily the first time a model loads.
venv = pathlib.Path(sys.prefix).resolve()
bad = []
for name in ("torchvision", "torchaudio"):
    try:
        m = __import__(name)
        where = pathlib.Path(m.__file__).resolve()
        ok = venv in where.parents
        print(f"{name} {m.__version__}  {'venv' if ok else 'NOT venv: ' + str(where)}")
        if not ok:
            bad.append(name)
    except Exception as e:
        print(f"!! {name} failed to import: {type(e).__name__}: {str(e)[:120]}")
        bad.append(name)
if bad:
    print(f"!! mismatched: {bad}. Reinstall with:")
    print("   uv pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 \\")
    print("     $(grep -E '^(torch|torchvision|torchaudio)==' /workspace/constraints.txt)")
    sys.exit(1)
PYEOF
cat <<EOF

$(printf '\033[1;32m')Bootstrap complete.$(printf '\033[0m')

Next:
  1. exec bash                      # reload shell so the env block applies
  2. \$EDITOR $ENVFILE              # add HF_TOKEN if you need gated weights
  3. cd $WORKSPACE/repo && git clone <this-repo> .
  4. python infra/preflight.py --model openai/gpt-oss-20b   # check the model fits
  5. see README.md "Running" for the stage order

EOF

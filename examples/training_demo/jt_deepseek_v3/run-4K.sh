#!/usr/bin/env bash
set -euo pipefail
source /workspace/env.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/train-4k.yaml}"
WORKER_NUM="${WORKER_NUM:-1}"
STEPS="${STEPS:-600}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/outputs/jt_deepseek_v3}"
RUN_NAME="${RUN_NAME:-run-$(date +%Y%m%d-%H%M%S)-$$}"
MASTER_PORT="${MASTER_PORT:-29881}"
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-false}"
LOSS_PARALLEL="${LOSS_PARALLEL:-false}"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"

case "$WORKER_NUM" in ''|*[!0-9]*) echo 'WORKER_NUM must be an integer' >&2; exit 2;; esac
case "$STEPS" in ''|*[!0-9]*) echo 'STEPS must be an integer' >&2; exit 2;; esac
test ! -e "$RUN_DIR" || { echo "Output already exists: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/launcher.log") 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$RUN_DIR/EXIT_CODE"; exit "$result"' EXIT

cd "$WORKTREE_ROOT"
export PYTHONPATH="$WORKTREE_ROOT:/workspace${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-6}"
command=(/workspace/venv/bin/torchrun --nnodes=1 --nproc_per_node="$WORKER_NUM"
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT"
  --log-dir="$RUN_DIR/ranks" --redirects=3 --tee=3
  "$WORKTREE_ROOT/examples/training_demo/train_text.py" "$CONFIG"
  "--training.train_iters=$STEPS"
  "--training.global_batch_size=$DP_SIZE"
  "--accelerator.tp_size=$TP_SIZE"
  "--accelerator.ep_size=$EP_SIZE"
  "--accelerator.sequence_parallel=$SEQUENCE_PARALLEL"
  "--accelerator.loss_parallel=$LOSS_PARALLEL"
  "--checkpoint.checkpoint_dir=$RUN_DIR/checkpoints")
printf '%q ' "${command[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
"${command[@]}"
printf '{"canonical_entry":true,"worktree":"%s","steps":%s}\n' "$WORKTREE_ROOT" "$STEPS" > "$RUN_DIR/SUCCESS.json"
printf 'Training complete: %s\n' "$RUN_DIR"

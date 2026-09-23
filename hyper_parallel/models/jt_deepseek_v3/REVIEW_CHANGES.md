# JT integration review guide

This revision targets `trainer_dev`. It incorporates the functional directory
split built on `21a84e6b` and keeps the original `deepseek_v3/` family and public
`components/modules/mla_attention.py` identical to that baseline.
The branch is rebased directly on upstream `trainer_dev` at `21a84e6b`.
Diagnostics change `b7136e25` is already part of that base and must not appear
as a new PR change. Review the three-dot diff against the current target branch.

## Model and MTP

- `b4e9134f`: colleague-provided Trainer integration and functional adapter layout.
- `6993b86a`: retain ordered EP aggregation and native HCCL gradient behavior used
  by the archived alignment implementation; also refactors fusion-attention
  context preparation. This is a shared functional change, not a change to the
  public MLA module.
- `fe89e8e5`: archive the reviewed JT directory migration, complete base model,
  reusable DeepSeek MTP and JT-specific MTP execution replacement. Standard
  DeepSeek paths are restored; JT construction lives in its independent family.

## Data

- `b79e00b0`: replace the family NPZ reader with three aligned indexed streams
  and public batching that preserves explicit supervision masks. Extract the
  initial model-computed loss adapter. Its mapping and gradient policy are
  subsequently simplified by the commits below.

## Observability

- `9cc2194e`: add detached LM/MTP/aux and QK-clip metrics through a shared logging
  hook. Metrics do not become extra backward objectives.

## Loss and model input contract

1. `18681bf6`: remove `jt_loss.py` and both family-owned CE autograd Functions;
   use public vocabulary CE for LM/MTP and public model-parallel mean for aux.
   Accept `shift_labels`, `loss_mask` and `position_ids` without a rename table.
2. `3fcdb211`: fix shared unreduced vocabulary CE so each rank receives the full
   per-token loss; add explicitly documented single-objective mean semantics
   for equally weighted model-parallel partitions. No total-loss TP scaling.
3. `df418d6d`: replace configurable argument renaming/group binding with optional
   forwarding of the public supervision fields. Preserve model-computed aux
   losses even when all supervised tokens are ignored.

The previous PR revision and its merge history are archived locally. The active
branch has a linear history starting at `21a84e6b`; superseded first-loss commits
and the history-preserving merge are excluded. Rebase does not change the tested
model code. The existing PR is updated in place so its submitted reviews remain
available, although comments on changed lines may become outdated.

## Validation status at publication

- 37 focused CPU unit tests passed in the Ascend container's Torch environment.
- Tests include dense CE gradient oracles, targets on both vocabulary shards,
  fractional masks, empty supervision, two/eight-rank mean gradient semantics,
  input ownership, complete model construction, MTP and logging.
- `git diff --check` and the agent catalog check passed.
- Raw pylint still reports existing HF/Torch inference and indirect-initializer
  findings; no clean full-lint claim is made. The known earlier copyright-range
  checker finding in `batch_parallel.py` remains outside this loss change.
- **The new full 256K TP8/EP8 ten-step numerical regression is pending.** Earlier
  exact ten-step total losses belong to the archived implementation. Shared CE
  may change floating-point gradient evaluation order. No new equality,
  production-scale performance, CP/PP or resume claim is made here.

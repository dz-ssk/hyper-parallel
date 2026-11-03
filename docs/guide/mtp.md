# Multi-token prediction (MTP)

The public components supply the MTP fusion layer and teacher-forced loops.
The DeepSeek-V3 model adapter additionally connects YAML construction, the text
Trainer, per-depth token accounting, and distributed execution. Other model
families supply their own decoder and metadata adapter. This is a training
integration, not a speculative-generation engine.

## Reference and numerical contract

- Fusion follows the released-weight convention implemented by
  [Transformers MtpLayer](https://github.com/huggingface/transformers/blob/0460c6d97702c7428460d83e53bcea6b76d110c0/src/transformers/modeling_layers.py):
  `[enorm(embedding), hnorm(previous_hidden)]`, a bias-free `2H -> H` projection,
  a Transformer block, then an independent output RMSNorm. The normalized output
  feeds both the output head and the next depth. Default RMSNorm uses FP32
  variance, casts normalized activations back, and then multiplies the weight,
  matching the DeepSeek Transformers cast order.
- The training loop follows the sequential, ground-truth-token construction in
  [DeepSeek-V3 section 2.2](https://arxiv.org/html/2412.19437v2#S2.SS2).
  It is not a port of the Transformers speculative-generation loop and does not
  claim parity with DeepSeek's unpublished full training implementation.
- For original length `S`, depth `k` (one-based) consumes embedding `input_ids[:, k:]`
  and the first `S-k` prior states. It returns `S-k` predictions. No synthetic
  padding token is fed through this variable-length entry point. All computation remains connected
  to the main-model hidden states and shared parameters through autograd.

## MTP depth configuration

The name `mtp_num_layers` follows
[Megatron-LM's MTP configuration](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/multi_token_prediction.md).
Use a model-owned YAML argument, not a Trainer-wide training option:

```yaml
model:
  _target_: my_model.build_model  # Your model adapter's builder
  mtp_num_layers: 2
```

| Value | Behavior |
|---|---|
| Omitted / `null` (default), or `0` | Disable MTP; the factory returns `None` and constructs no layers |
| Positive integer `D` | Construct `D` independent MTP layers and predict `D` additional future tokens |

The public factory consumes the configured value. The model builder must accept
and pass it through, attach the result before optimizer/sharding setup, and call
it in forward/loss when enabled. This YAML fragment is for an adapter implementing
that connection; adding the field to an arbitrary existing model does not enable
MTP automatically. Negative, boolean and noninteger Python values are rejected.

## Ready-to-use DeepSeek-V3 training adapter

Merge [the MTP recipe](../../hyper_parallel/models/deepseek_v3/recipes/mtp.yaml)
with a training configuration supplying the dataset and production dimensions.
The builder target is `hyper_parallel.models.deepseek_v3.adapter.mtp.build_model`.
Set `model.mtp_num_layers`, `model.mtp_loss_factor`, and
`model.mtp_loss_normalization`. It creates and registers every MTP layer before
sharding and optimizer construction. The depth is additional to
`config.num_hidden_layers`, which counts the trunk only. The output head and
embedding are shared with the trunk.

Use `MTPTextParallelBatch` as `dataloader.get_batch` with the same depth and
normalization (the recipe uses YAML anchors). It prepares future IDs, positions
and labels and exposes independent per-depth token counts. Do not reuse the LM
valid-token count for MTP: later depths have different valid targets. The
standard Trainer globally normalizes each objective over DP/CP and gradient
accumulation. `ModelOutputLoss` consumes the model's named loss mapping.

Logs include `foundation_loss/lm_loss`, `mtp_depth1_loss`, etc., the weighted
aggregate `mtp_loss`, and the total `loss`. The aggregate is a detached metric;
it is **not** added a second time to the backward objective. MTP checkpoints
include registered layers in the normal model/optimizer state dictionaries.
Depth zero preserves the native Hugging Face model and its inference behavior.

### Distributed boundaries

`forward_aligned` keeps local sequence lengths fixed, so each depth can use the
same TP/SP and CP layout. `shift_mtp_tokens` exchanges only integer prefixes
within an explicit CP group before embedding; it does not exchange full hidden
states. It pads the end of the global row, never wraps to another row, and accepts
an offset greater than a local shard. CP currently requires equal, contiguous,
rank-ordered sequence shards. Dense boolean masks use `True = allowed`; additive
masks use zero for allowed entries. Packed-document boundaries must be present
in the dense attention mask and ignored targets. A compressed mask without
boundary metadata is not an equivalent input contract.

- **TP/SP:** existing embedding, attention, FFN and head layouts apply to both
  trunk and MTP. Fusion projections are replicated over hidden features and
  preserve sequence ownership. Use `loss_parallel: false`: this adapter consumes
  full-vocabulary logits, not vocabulary-parallel cross entropy.
- **CP:** install the public MLA module replacement and `mla_cp_wrapper` on
  `*.self_attn` / `*.self_attn.forward`, covering both trunk and MTP blocks.
  A trunk-only `model.layers.*` rule leaves MTP attention unadapted.
- **EP:** attach the public DeepSeek MoE compute recipe to every routed MoE
  boundary, including `mtp.layers.*.mtp_block.mlp`. Dense FFNs must not receive
  that recipe. Expert placement, routing and exchange use the existing EP path;
  no fixed-router assumption is introduced.
- **FSDP/HSDP and recomputation:** MTP layers participate in repeated-block
  discovery. An MTP wrapper and its nested decoder must not both become separate
  recomputation units. Apply the ordinary FSDP and activation-checkpoint settings.

### Explicit pipeline schedules

The ordinary monolithic `TextTrainer` builder does not partition pipeline stages.
For PP, use `partition_mtp_model` before TP/CP/EP/FSDP wrapping, then wrap the
partition in `MTPPipelineModule` and construct the framework's `PipelineStage`.
For PP alone, `build_mtp_pipeline_stage` combines those operations.
All helpers are in `hyper_parallel.models.deepseek_v3.adapter.mtp_pipeline`.

Trunk layers are partitioned into nonempty consecutive ranges; all MTP layers,
the final norm and output head reside on the last stage. First and last stages
own embedding replicas declared through `SharedParameterInfo`, so the pipeline
runtime broadcasts shared weights and sums shared gradients. These replicas are
not independent parameters to optimize without synchronization.

The stage's first positional tensor is input IDs on the first stage and received
activations thereafter. Supply input IDs, labels and attention metadata as
keyword arguments on every stage. `MTPPipelineModule` returns a scalar objective
on the last stage, and exposes detached `last_loss_metrics` for logging. Set
`loss_scale=1/microbatch_count` for equal-size microbatches; unequal valid-token
counts require objective-specific weighting by the caller. A standalone PP
partition retains original global layer indices for checkpointing; the optional
stage wrapper introduces a `model.` prefix in its own state dictionary.

## Model integration

```python
from hyper_parallel.components.modules import build_mtp, MTPLayer
from hyper_parallel.components.losses import compute_mtp_loss

# Inside build_model(..., mtp_num_layers=None), or the model constructor:
# build_causal_decoder(k) belongs to the adapter and returns a fresh causal block.
model.mtp = build_mtp(
    mtp_num_layers,
    layer_factory=lambda k: MTPLayer(hidden_size, build_causal_decoder(k)),
)

# In the model's forward/loss:
total_loss = main_lm_loss
if model.mtp is not None:
    output = model.mtp(
        trunk_hidden_before_final_norm,
        input_ids,
        embedding=model.embed_tokens,
        head=lm_head,
        decoder_kwargs=per_depth_attention_and_position_kwargs,
    )
    mtp_loss = compute_mtp_loss(
        output.logits, next_token_labels, loss_factor=0.3,
        normalization="sequence",
    )
    total_loss = total_loss + mtp_loss

```

`MultiTokenPrediction(layers)` also accepts an explicit list of independent layers.
Each depth must have its own decoder and three independent norms. Embedding and
head are supplied at forward time, so the component does not register duplicate
copies. `head=None` returns hidden states only for a model-owned chunk-loss path.
The factory's usual module initialization applies; matching an existing model's
initialization distribution is the adapter's responsibility.

The loop requires full, unsharded `[B, S]` rows and `S > mtp_num_layers`. It supports
`B > 1`. Each decoder must be causal. Its kwargs at depth `k` must describe the
shorter `S-k` sequence; the component cannot infer how to slice model-specific
rotary embeddings, packed masks or cache metadata. With `decoder_kwargs=None`,
the decoder must construct causal metadata itself. For CP/TP/SP integration, an
adapter can call individual layers with correctly aligned tensors; no distributed
layout or boundary exchange is implicitly performed by this training loop.

## Labels, masks and normalization

`next_token_labels` are already aligned main-LM targets, not unshifted input IDs:

```text
input_ids:              t0 t1 t2 t3
next_token_labels:      t1 t2 t3 t4
MTP depth 1 embeddings: t1 t2 t3
MTP depth 1 targets:    t2 t3 t4
MTP depth 2 embeddings: t2 t3
MTP depth 2 targets:    t3 t4
```

If `t4` is unavailable, its label is `-100`. Apply binary supervision masks to the
labels as `-100` before calling the loss. Arbitrary fractional token weighting
and packed-document transition policies are not implemented by this helper.

The helper consumes full-vocabulary logits and returns
`loss_factor / D * sum(depth_loss)`:

| Normalization | Denominator at each depth | Meaning |
|---|---|---|
| `sequence` (default) | `B*S` | Original-length denominator in paper Eq. 24 |
| `valid_tokens` | Number of non-ignored targets at that depth | Masked-token mean |

These are different objectives when target counts differ. Neither is labelled
the universal industry training convention. An all-ignored depth contributes
a differentiable zero. FP32 CE is used even for BF16 logits. Distributed/global
micro-batch weighting must be handled by the model/Trainer adapter.

`compute_mtp_loss` consumes depth-aligned `[B, S-k, V]` logits and applies explicit
normalization and depth weighting. The existing `calculate_mtp_loss` instead
shifts every supplied logits tensor by one position, delegates reduction to its
`loss_fn`, and sums the results without depth weighting. These are different input
contracts, not interchangeable aliases; `calculate_mtp_loss` behavior and callers
are unchanged.

## Fork adapter boundary

Model-specific fusion order, dtype overrides, router behavior, private normalization and static-framework precision overrides do not belong in the public default.
The public adapter owns its documented padding, packed-boundary and CP-shift contracts.
A fork can supply its own fusion layer to `MultiTokenPrediction`, or use its own loop
around public layers when its sequence contract differs.

For an existing hidden-first checkpoint, changing only the concatenation order
changes predictions. Swap the two `H`-wide column blocks of `eh_proj.weight` to
preserve the linear map. Resume training also requires applying that permutation
to the corresponding optimizer-state tensors. Do not swap enorm/hnorm parameters:
they still normalize embedding/hidden respectively. Other checkpoint names must
be mapped by the adapter; no whole-model checkpoint importer is provided here.

## Validation scope

CPU tests cover YAML depth selection, disabled MTP without layer allocation,
factory/list numerical parity, causal multi-depth training, gradients through the shared
embedding/head and trunk, depth target alignment, both normalization rules,
ignored targets, BF16 loss, state-dict reload and the existing loss API.
Real-model CPU tests also cover YAML construction, AdamW updates, model and
optimizer checkpoint continuation, Trainer logging without double-counting,
unequal microbatch token counts, padding, packed documents, and recomputation.

The NPU workers in `tests/torch/context_parallel/_test_mtp_parallel.py` compare
complete LM/MTP losses and parameter gradients with an unsharded reference:

- Eight ranks: TP2 + CP2 + EP2, FSDP4, with SP/recomputation off and on;
  an additional SP/recomputation case uses HSDP (two shards, two replicas).
- Two ranks: an actual PP schedule, including P2P activation/gradient transfers
  and shared-embedding synchronization.

These are short BF16 numerical parity tests, not long-run convergence results.
PP combined with all other parallel dimensions has not been validated by these
tests. The public component alone does not install a model's parallel recipes.

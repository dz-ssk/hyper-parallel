# JT DeepSeek V3 integration

This family constructs a complete HF-derived causal JT model before optional
acceleration. It has its own `JTDeepseekV3Config`, `JTDeepseekV3ForCausalLM` and
lazy adapter registration. The standard `deepseek_v3` family is independent.

## Ownership

| Location | Responsibility |
| --- | --- |
| `configuration.py` | Reference configuration validation and HF field mapping |
| `modeling_jt_deepseek_v3.py` | Complete decoder, causal MLA, MoE routing/combine, residual precision, public MTP construction and objective |
| `adapter/conversion/` | Checkpoint mapping, MLA projection implementation and JT MTP execution policy |
| `adapter/distributed/` | Bind model-owned expert semantics to Hyper EP dispatch and communication |
| `adapter/policies/` | Declarative parameter sharding roles |
| `adapter/runtime/` | Pre-shifted dataset inputs, Trainer loss binding and retained alignment optimizer |
| `adapter/jt_builder.py` | Build, optionally replace, strictly load weights and apply Hyper infrastructure |
| `recipes/jt_deepseek_v3.yaml` | Replacement, parallel layout and runtime component selection |

The model directly constructs dense/shared experts, routed experts, norms and an
executable public MTP. The recipe replaces Attention with `JTDeepseekV3MLAAttention`
and selects `JTDeepseekV3MTPExecution` at `mtp.execution`. The latter is required
for JT MTP precision parity; it does not add missing prediction depths.

## Reusable DeepSeek MTP

`hyper_parallel/components/modules/mtp.py` provides `DeepseekV3MTP`: independent
fusion norms/projections/decoders, sequential future-token and target shifts,
shared embedding/head calls, and weighted multi-depth cross entropy. Another
family supplies its decoder and can use the complete algorithm directly:

```python
from hyper_parallel.components.modules import DeepseekV3MTP

mtp = DeepseekV3MTP(
    hidden_size=config.hidden_size,
    num_layers=config.num_nextn_predict_layers,
    decoder_factory=build_causal_decoder,
    rms_norm_eps=config.rms_norm_eps,
)
result = mtp(
    trunk_hidden, input_ids,
    embedding=model.embed_tokens,
    head=shared_output_head,  # The main model's output norm and vocabulary projection.
    labels=next_token_labels, loss_mask=next_token_mask,
    loss_factor=0.3,
    decoder_kwargs=attention_arguments,
)
loss = main_lm_loss + result.loss
```

The factory receives the zero-based depth and returns a causal decoder with a
Tensor output. V3/V3.2 decoders can implement different attention while reusing
the same MTP orchestration. Embedding/head modules are passed by reference at
forward time, so their parameters are not registered twice. Labels are already
shifted once for the main LM objective. Each row is one complete sequence;
packed-document boundary handling and context-parallel shifting are unsupported.
The default CE divides by the original token count, as in V3 paper Eq. 24;
a custom loss callback can provide masked or vocabulary-parallel reduction.

`adapter/conversion/jt_mtp.py` subclasses the public execution component and
changes only fusion precision and recurrent-state selection: embeddings are
rounded before normalization, both normalized branches are fused in BF16, and
normalized prediction states are carried between depths. JT also supplies its
norms/decoder, per-depth output normalization and existing masked vocabulary loss.
The public default carries the raw decoder state and lets the shared head own
output normalization.

The execution module is parameter-free and is a sibling of `mtp.layers`, allowing
its replacement in the same plan as Attention inside MTP layers without nested
replacement targets. Existing `mtp.layers.*` checkpoint and sharding paths are
preserved. This composition reuses optimized child kernels; it is not a new fused
MTP kernel, and no MTP speedup is claimed without performance measurement.

## Training entry

Use the existing common Trainer entry with explicit local assets:

```bash
torchrun --nproc_per_node=8 examples/training_demo/train_text.py \
  hyper_parallel/models/jt_deepseek_v3/recipes/jt_deepseek_v3.yaml \
  --model.reference_yaml=/path/to/reference.yaml \
  --model.reference_weights=/path/to/initial_weights \
  --dataset.data_path=/path/to/shared_tokens.npz
```

The retained alignment recipe requires a batch of one full 262144-token sequence,
TP8/EP8/SP and DP/CP/PP1, FP32 parameters and BF16 compute. It loads matching
exported initial weights; it does not download assets. Its optimizer owns the
reference gradient synchronization, clipping and update schedule. Other
parallel topologies, performance gains and checkpoint continuation are not
established by this recipe's regression test.

The common entry calls `TextTrainer.train()`. No model-specific launcher or
train/evaluate branch is added. Setting `train_iters: 1` still executes a training
step, including backward and update; it does not select evaluation-only behavior.

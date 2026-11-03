# 多 Token 预测（MTP）

MTP 是公共建模能力。Trainer 调用公共模型构建入口时，根据 YAML 创建并挂载 MTP，
不需要在 `models/deepseek_v3` 或其他模型目录中增加一个专用 MTP 模型类。

## 构建顺序

```text
YAML → Trainer → HyperAutoModelForCausalLM.from_config / from_pretrained
     → 创建主干 → attach_mtp → build_mtp → 注册 model.mtp
     → 模块替换 → 并行分片 → 重计算 → 权重加载或初始化 → 优化器
```

MTP 必须在分片和优化器创建前注册。原模型类型、主干参数名称和共享权重保持不变；
MTP 参数使用 `mtp.layers.*` 名称。公共训练入口同时连接前向、loss 和日志，
不只是把一个未参与计算的模块挂到模型上。

## YAML 配置

完整的小模型配置片段见 [examples/mtp.yaml](../../examples/mtp.yaml)。
补充数据源并替换生产模型维度后，可交给现有 Trainer 使用。

```yaml
model:
  _target_: hyper_parallel.models.HyperAutoModelForCausalLM.from_config
  mtp_num_layers: &mtp_depth 2
  mtp_loss_factor: 0.3
  mtp_loss_normalization: &mtp_norm sequence
  config:
    _target_: transformers.AutoConfig.for_model
    model_type: llama
    # 在这里配置模型结构；此示例仅展示入口。
loss_fn:
  _target_: hyper_parallel.components.losses.ModelOutputLoss
dataloader:
  get_batch:
    _target_: hyper_parallel.data.batching.mtp.MTPTextParallelBatch
    mtp_num_layers: *mtp_depth
    mtp_loss_normalization: *mtp_norm
    source_type: online
    attention_mode: dense
accelerator:
  loss_parallel: false
```

| 参数 | 含义 |
| --- | --- |
| `mtp_num_layers` | 额外的 MTP 层数，不计入主干 `num_hidden_layers`；0 关闭，正整数创建独立层 |
| `mtp_loss_factor` | 所有深度的总系数，每层分配 `factor / D` |
| `mtp_loss_normalization` | `sequence` 或 `valid_tokens`，见下文 |
| `mtp_boolean_mask` | 默认 false，向 Decoder 传入加性 mask；使用要求 `True=允许` 的公共 attention 替换时设为 true |

首次构建未配置层数时不开启 MTP。构建时的 MTP 配置会写回模型 config；从保存的 config
恢复时自动重建，显式 `mtp_num_layers: 0` 可覆盖保存的层数。模型与 batch 端的层数和
归一化需要一致，示例用 YAML anchor 共享。数据端负责各深度的 token 计数，避免梯度累加
时把 LM 的有效 token 数错误地用于 MTP。

## 复用模型结构的边界

公共接入通过结构接口工作，不根据模型名称分支。当前要求：

- 模型提供 `get_decoder()`、`get_input_embeddings()`、`get_output_embeddings()` 和原生初始化方法。
- Decoder 提供非空 `layers`、最终 `norm` 和 HF 风格 `rotary_emb`；每层接受位置、attention mask
  和 rotary 元数据，返回 hidden tensor 或以 hidden 为第一项的 tuple。
- 当前公共循环面向常规全因果 Decoder；自定义残差、多模态输入、滑窗/混合注意力等额外语义，
  不能仅凭接口名称相同就视为已经适配。

默认用最后一个主干 Decoder 的**独立副本并重新初始化**构造每个 MTP 层，不共享主干层参数。
外层融合和 Norm 使用公共 `MTPLayer`。需要其他 MTP block 结构的调用方，可以通过
`attach_mtp(..., layer_factory=...)` 或 `build_mtp(..., layer_factory=...)` 提供自己的层工厂。

不带训练标签的普通推理仍调用原模型前向；没有实现 MTP speculative decoding 或 MTP KV cache。

## 前向、loss 与日志

每层使用真实未来 token：

```text
输入 token：      t0 t1 t2 t3
主 LM 标签：      t1 t2 t3 t4
MTP 第 1 层：     在原位置 i 融合 token i+1，预测 token i+2
MTP 第 2 层：     在原位置 i 融合 token i+2，预测 token i+3
```

融合顺序是 `[enorm(embedding), hnorm(hidden)]`，之后执行无偏置 `2H → H` 投影、
因果 Decoder 和输出 RMSNorm。各层的三个 Norm 独立，embedding 与输出头复用主干。
主干 hidden 在最终 Norm 之前进入 MTP；每层的归一化输出传给下一深度。

| 归一化 | 每层分母 | 说明 |
| --- | --- | --- |
| `sequence`（默认） | 原始 `B × S` | 包含无有效预测的尾部位置，对应原序列长度口径 |
| `valid_tokens` | 本深度非忽略标签的数量 | 不同深度的有效目标数可能不同 |

两种口径会改变梯度缩放，不应混用。CE 用 FP32 计算，无有效目标时产生可反传的零。
Trainer 按各目标的计数处理 DP/CP 和梯度累加；主 LM 保持有效 token 均值。

日志包含 `foundation_loss/lm_loss`、`mtp_depth1_loss` 等分层 loss、聚合 `mtp_loss`
和总 `loss`。聚合值是脱离计算图的日志项，**不会再次加进反向目标**。

公共组件同时保留两种执行入口：

- `forward`：处理完整序列，第 k 层长度为 `S-k`。
- `forward_aligned`：处理已对齐的定长本地分片，保持 TP/SP/CP 的序列布局。

已有 `calculate_mtp_loss` 行为不变。新增 `compute_mtp_loss` 接收按深度对齐的 logits，
并明确处理分母与深度系数；二者输入约定不同，不能直接替换。

## 并行与重计算

- **TP/SP**：主干与 MTP block 复用已有并行机制。融合投影 `mtp.layers.*.eh_proj`
  需要在 recipe 中声明为参数全复制，并保持输入输出的序列归属；不要把它当作普通列切分投影。
  当前使用完整词表 CE，要求 `loss_parallel: false`。
- **CP**：使用相同的 attention 并行 recipe，匹配范围必须覆盖 MTP block。
  公共 `shift_mtp_tokens` 只交换整数元数据前缀，越过全局尾部补齐，不能回绕到另一条序列。
  当前要求等长、连续且按 CP rank 排列的分片。
- **EP**：已有 MoE recipe 同时覆盖主干与 MTP 的 MoE；不要将 MoE recipe 应用到 dense FFN。
- **FSDP/HSDP、重计算**：MTP 层已注册，可被公共 repeated-block 发现机制识别；
  避免 MTP 外层和内部 Decoder 被重复包装。
- **Packed 输入**：使用 dense mask 和正确的忽略标签表达文档边界；缺少边界信息的 compressed
  输入不满足本接入约定。

### PP

公共接口位于 `hyper_parallel.distributed.mtp_pipeline`：
`partition_mtp_model`、`MTPPipelineModule`、`build_mtp_pipeline_stage`。

主干分到连续、非空的 stage，全部 MTP、最终 Norm 和输出头位于最后 stage。
首尾 stage 的 embedding 声明为共享参数，由 PipelineStage 同步权重和梯度。
先切 stage，再做该 stage 的 TP/CP/EP/FSDP 包装。

PP 使用显式 schedule；普通 TextTrainer 不会仅因为设置 `pp_size` 就自动完成切分。
等大小微批使用 `loss_scale=1/微批数`，有效 token 数不同时由调用方分别加权。
PP 与其他并行维度同时组合不属于当前已验证范围。

## 验证与私仓边界

测试包括真实小模型的配置建层、前反向、AdamW 更新、模型与优化器状态恢复、日志、
微批计数、文档边界及重计算。多卡测试将 LM/MTP loss 和完整参数梯度与无分片参考模型比较，
另验证实际 PP 通信与共享 embedding 梯度同步。这些短程测试不代表生产规模长期收敛验收。

相关代码：

- `tests/ut/auto_models/test_mtp_training.py`
- `tests/torch/context_parallel/_test_mtp_parallel.py`

私仓 JT 的 hidden-first 拼接、路由和精度/归一化差异由私仓维护。
迁移不同拼接顺序的 checkpoint 时，需要同步转换融合投影的权重列和对应优化器状态；
本 PR 不提供私仓整网权重转换器。

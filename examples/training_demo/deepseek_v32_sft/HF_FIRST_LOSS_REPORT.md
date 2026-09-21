# HF + Hyper：参考 SFT 首 loss 对齐报告

日期：2026-09-21。实现提交：`9bf9790c56fa0fde2af3cef21438160aabf41b9a`。分支：`codex/deepseek-v32-sft`。

## 1. 当前结果与适用范围

新方案已经在 八卡 Ascend NPU 环境上完成完整 262144 token 的首 loss 前向。8 个 rank 的总 loss 均为 **8.117478370666504**，与参考 MindSpore 保存的首步结果逐位一致。不带参考模型和诊断钩子的独立复跑也得到同一结果。

| 指标 | HF + Hyper | 对比对象 | 结果 |
| --- | ---: | --- | --- |
| 总 loss | 8.117478370666504 | 参考保存的首步结果，以及独立运行的归档参考模型 | 8/8 rank 逐位一致 |
| LM loss | 6.241546630859375 | 归档参考模型 | 8/8 rank 一致 |
| 已乘系数的 MTP loss | 1.8757120370864868 | 归档参考模型 | 8/8 rank 一致 |
| MoE 辅助 loss | 0.00021915044635534286 | 归档参考模型 | 8/8 rank 一致 |
| 中间张量 | 每 rank 31 项，共 248 项 | 本轮独立执行的归档 PyTorch 参考模型 | 每项不同元素数为 0，最大绝对差为 0 |
| 初始参数 | 每 rank 46 个参数条目 | 参考导出的初始权重映射及 Hyper 实际分片 | 全部逐元素一致 |

**验证边界：**这是首步前向验收，不是一次完整训练 step。当前 loss evaluator 明确要求 `torch.no_grad()`，未执行 backward、梯度裁剪、Muon 更新、router bias 更新或第二步前向。本次没有重新跑参考训练，而是读取既有参考证据；中间张量与分项 loss 对比使用独立加载、未修改的旧 PyTorch 参考实现。不能把这 248 项称为本轮直接采集的 MindSpore 中间张量，也不能把旧方案的五步结果算作新方案的五步结果。

当前模型按此前获准的方式缩小了维度和专家数，保留 256K、MLA、MoE、MTP 等主要功能。它不是生产 236B 模型的完整规模验收，也未验证所有 YAML 组合。

## 2. 方案结构与修改边界

使用容器中 Transformers **5.13.1** 的真实 `DeepseekV32Config` / `DeepseekV32ForCausalLM` 构建模型，未下载或加载预训练权重。初始权重来自同一参考导出。`DeepseekV32SFTForCausalLM` 继承真实 HF 类，复用其 embedding、Decoder 子模块、投影、打包专家和 lm_head，再通过显式模块替换及 EP compute factory 接入 Hyper。

前向调用关系为：HF 模型构造 → 参考配置和权重映射 → Hyper 模块替换 → Hyper planner 和 apply_sharding_plan → 参考顶层前向编排 → HF 派生 Decoder / Hyper MLA / Hyper EP → MTP 和 masked loss。

顶层 forward 为了参考 MTP、256K mask 和 loss 语义作了覆盖，不是完全原封不动地调用 HF `forward()`。Attention 复用 Hyper MLA，只有 RoPE 精度边界做局部覆盖。EP 的 token 分发、通信和专家绑定复用 Hyper，路由和合并数值规则由示例适配。没有整体复制旧手写模型作为新模型主体。

所有新提交修改都位于 `examples/training_demo/deepseek_v32_sft/`。未修改 Hyper 公共源码、安装目录中的 Transformers 或参考源码，未使用全局 monkey patch。运行发生在服务器，本地工作树用于开发、保存 Git 历史和 review。

旧参考实现单独归档。新方案从 Hyper 基线 `c20bd626f035972d0e3796ee21aef8ab48ecc705` 独立分支，不把旧模型提交混入新分支。

## 3. Review 分类方法

下面按模块分组，**各组内部从重要到次要排列**；不同模块之间不做全局重要性排名。Excel 的“模块内顺序”也只在所属模块内有效。

修改性质限定四类：

- **功能补齐**：补充参考所需、当前 HF 路径未提供的功能或语义。
- **精度适配**：对齐运算顺序、dtype、舍入或归约语义。
- **并行与接口接入**：连接模型替换、权重映射和 Hyper 并行接口。
- **验证与文档**：诊断、验证入口、证据说明及报告。

性质按“修改点”标注。同一个 commit 涉及不同模块或性质时会拆成多行，commit 可以重复，不能把行数当作提交数。

## 4. Attention / MLA

### 1）RoPE 运算顺序与精度边界

**精度适配 · `0f3be863d3237f7479f0e1ee39bc00b734f99c5c`**
文件：`attention.py`、`audit.py`。

新增 `DeepseekV32SFTMLAAttention(MLAAttention)`，保留 Hyper 的投影和注意力实现，仅在 `_project_attention_inputs` 的 Q/K rotary 部分替换计算。`ExplicitFP32RotaryEmbedding` 先按偶/奇通道重排，转 FP32，分别执行乘法和相加，最后转回输入 dtype。

使用通用融合 RoPE 时，总 loss 为 8.117480278015137，与目标相差 1.9073486328125e-6，即该数值位置的 2 个 FP32 ULP。第一层 Q/KV 的 norm 和上投影一致，Attention 后的输出开始出现差异。局部改为参考运算顺序后，全部已采样边界及 loss 恢复一致。

同输入直接对照进一步确认：rank 0 第一层 Q rotary 中，通用融合输出与显式顺序有 4,602,625 个元素不同，最大差 0.00390625；K rotary 有 4,481,732 个元素不同，最大差 0.0078125。适配结果与参考 rotary 均完全一致。这证明当前输入下融合与显式计算的数值结果不同；没有据此推断已证实某个底层内核的具体 FMA 或累加器实现。

### 2）用 Hyper MLA 替换 HF Attention

**并行与接口接入 · `33a8d41194d3093b7de6c25268be3329d229cc87`**
文件：`audit.py`、`parallel.py`、`probe.py`。

通过 Hyper 公共 replacement 接口替换两个主干层和一个 MTP 层的 Attention，使用完整因果 MLA，符合当前参考 YAML。HF V3.2 自带 DSA indexer 的路径没有启用，不能宣称本轮验证了 DSA。Q/KV 降维投影物理融合时检查权重覆盖和逻辑分段，注意力上投影和输出投影按 Hyper 实际布局分片。

### 3）256K 的 mask 和位置输入

**功能补齐 · `5dc70c7a6e6e79dcfe43534af26a655f1a9d3575`**
文件：`model.py`。

顶层直接编排已有 HF 层，显式传入位置频率和 `actual_seq_len`，进入 Hyper 的紧凑 causal mask / TND 通路，避免上层构造 256K×256K 稠密 mask。逆频率固定 FP32，长度仍为 262144，没有以缩短序列绕过问题。

## 5. MoE

### 1）参考路由及辅助 loss

**功能补齐 · `b41f2f818fbaf0e1813e3c4decd364ea0ef38a20`**
文件：`ep.py`。

在 Hyper `build_ep_compute` 上注入参考 router：FP32 gate、sigmoid、correction bias 参与选择、top-k、被选概率归一化及 scaling。按同一顺序计算全局专家频率和 sequence auxiliary loss，并保留当前配置要求的零概率 pad tokens。当前只读初始 correction bias，后续训练的 bias 更新未实现。

### 2）路由权重和 routed/shared 相加精度

**精度适配 · `b41f2f818fbaf0e1813e3c4decd364ea0ef38a20`**
文件：`ep.py`。

路由和 dispatcher 加权合并保持 FP32；专家计算使用 BF16；routed 输出按参考先转 BF16 再回 FP32，与 shared 输出 FP32 相加。这样避免在 token 分发或加权阶段提前降低概率精度。已验证 MoE 输出边界及辅助 loss；未验证通用 dispatcher token 排序对反向归约的影响。

### 3）打包专家对接 grouped SwiGLU

**并行与接口接入 · `24bc1ca87849c80fdb2e735c5126ee19560a6111`**
文件：`adapters.py`。

`DeepseekV32SFTExperts` 保留 HF packed 参数，通过 expert-major 接口连接 Hyper grouped SwiGLU。专家通信和 binder 继续使用 Hyper，未在示例中重写 EP 通信。

## 6. Decoder / Norm / MLP

### 1）残差及归一化精度顺序

**精度适配 · `24bc1ca87849c80fdb2e735c5126ee19560a6111`**
文件：`adapters.py`。

通过 HF 派生的 `DeepseekV32SFTDecoder` 和 `DeepseekV32SFTRMSNorm` 保留原有参数、子模块，显式对齐 FP32 residual、FP32 RMSNorm 及返回 BF16 的边界。Attention 加法、MLP 加法之后按参考转换。仅“公式相同”不能保证 BF16/FP32 转换位置一致。

### 2）Dense 和 shared MLP 激活

**精度适配 · `24bc1ca87849c80fdb2e735c5126ee19560a6111`**
文件：`adapters.py`、`parallel.py`。

复用 HF gate/up/down 投影，将 Gate/Up 输出连接到融合 SwiGLU，并匹配 shared expert 的实际 `x` 入参接口。当前覆盖前向；旧方案自定义 backward 的舍入规则尚未迁移或验收。

## 7. MTP / SFT Loss

### 1）补充参考 MTP 分支

**功能补齐 · `5dc70c7a6e6e79dcfe43534af26a655f1a9d3575`**
文件：`model.py`、`audit.py`。

在实际 HF 模型上增加 MTP 参数容器、HF 派生 Decoder、enorm/hnorm/eh_proj/final norm。使用主干最终 norm 之前的 hidden，拼接顺序为 hidden 后接 embedding；共享主模型 embedding 和 lm_head。共享这些模块不等于把输入 embedding 和输出 head 彼此绑权重。

tokens、labels、mask 向左移位，尾部补零，MTP loss 按当前 0.3 系数计入，并汇总其 MoE auxiliary。当前实际验证一个 MTP 层。

### 2）masked、词表并行交叉熵

**精度适配 · `5dc70c7a6e6e79dcfe43534af26a655f1a9d3575`**
文件：`model.py`。

按参考顺序计算 FP32 vocabulary-sharded logsumexp、目标 logit、mask 求和和分母 `+1e-8`。数据里的 labels 已预先移位，不再对主 LM labels 额外移位。当前 loss 使用 world group，适用已限制的 TP8/world8 拓扑；不是任意 DP/TP 组合的通用 loss。明确阻止在启用 autograd 的模式下误当训练 loss 使用。

## 8. 配置 / 权重

### 1）完整参数映射和初值一致性

**并行与接口接入 · `34d1d481e6bddb8c27e0a57355bedb3cd915c625`**
文件：`weights.py`。

重建参考各 rank 分片，对 replicated 副本检查一致性；处理 Dense/shared 交错 Gate/Up 行与 HF 分离投影、专家二维展平与 HF 三维 packed 格式、各 norm / Attention / MTP 命名。支持逆转换逐字节核对，不靠“形状一致”代替值一致。

8 个 rank 每个读取 65 个源状态键，通过 48 条映射规则得到融合前 51 个 HF state tensors，融合后 48 个（46 个参数及 2 个 router bias buffers）。136 个辅助状态分片另行保留，没有当作模型权重静默丢弃。实际 Hyper 分片后再次逐参数比较。

### 2）固定当前 YAML 支持范围

**并行与接口接入 · `34d1d481e6bddb8c27e0a57355bedb3cd915c625`**
文件：`config.py`。

显式转换到真实 `DeepseekV32Config`，检查 batch、长度、并行和 dtype 等条件，使用当前参考的完整因果 MLA 配置，关闭 KV cache 和输入/输出绑权重。拒绝不支持的配置，不假设 HF 默认值恰好等于参考 YAML。

## 9. 并行

### 1）Hyper TP / EP / SP 真正接入

**并行与接口接入 · `33a8d41194d3093b7de6c25268be3329d229cc87`**
文件：`parallel.py`、`probe.py`。

调用 MeshContext、ShardingPlanner 和 apply_sharding_plan。补充 q_b / kv_b、MTP eh_proj、Attention norm 和 shared expert 的 scoped overrides；共享专家参数保持 replicated、输入输出按 SP 处理；显式配置 region_dispatch。计划覆盖全部参数，不以 allow_uncovered 绕过遗漏。

TP8 与 EP8 使用同一组 8 卡上可表达的不同通信域，不需要 64 卡。最终输出保存真实计划和各 rank 参数形状。EP factory 在首 loss 模式下显式选择 `deepseek_v32_sft_ep_compute`，不是只设置 ep_size。

## 10. Optimizer

本轮没有新增或修改 Muon/Adam 优化器，也没有优化器更新提交。`34d1d481` 和 `33a8d411` 的权重映射保留逻辑矩阵边界（例如融合 Q/KV 的 128/576 分段），便于后续接入 Muon，但这不等于优化器适配完成。

下一阶段应先对齐梯度、裁剪及第一步更新，核对融合参数中的每个逻辑矩阵、momentum、正交化和 norm 归约，再验证二三步。旧方案第五步的小权重差异仍作为经验保留，不在本次首 loss 验收中宣称解决。

## 11. 验证 / 文档

### 1）独立首 loss 和中间张量诊断

**验证与文档 · `9bf9790c56fa0fde2af3cef21438160aabf41b9a`**
文件：`probe.py`、`diagnostics.py`。

提供 `--first-loss-data` 完整前向和可选 `--reference-code` 对照。归档模型在独立命名空间加载、独立计算，hook 只采集和比较输出，不改张量或参数。诊断完成移除 hook；正常首 loss 模式不执行归档模型。

主干 embedding、各层 norm / Attention / MLP / residual、首层 Q/KV 投影和 RoPE、MTP、head 共 31 项/rank。此为选定边界，不代表每一个内部算子都做了对比。最终又以关闭诊断的方式独立复跑。

### 2）迁移设计及最终交付

**验证与文档 · `810d4f3139e8f8e80041d74d4feb02344c5bc97b`**：最初接入表 `HF_MIGRATION.md`，明确 HF / Hyper / 参考语义边界和分阶段验收。

本报告及接入表状态更新作为单独文档提交；文档不自写自身 commit 哈希。最终文档提交可执行 `git log -1 --format=%H -- examples/training_demo/deepseek_v32_sft/HF_FIRST_LOSS_REPORT.md` 查询，Excel 提交索引记录完整哈希。

## 12. 固定运行条件与证据

| 项目 | 本次值 |
| --- | --- |
| 长度 / batch | 262144 / 1 |
| 并行 | TP8、EP8、SP 开启、CP1、DP1、PP1 |
| 主干 / MTP 层 | 2 / 1 |
| hidden / dense intermediate / vocab | 256 / 512 / 512 |
| heads / q_lora / kv_lora | 8 / 128 / 512 |
| qk_nope / qk_rope / v | 128 / 64 / 128 |
| experts / top-k / shared / moe intermediate | 8 / 2 / 1 / 128 |
| routed scaling / MTP factor / aux coefficient | 2.827 / 0.3 / 0.0001 |
| 参数 / 主体计算 | FP32 / BF16 |
| 当前适配约束 | causal MLA、无 KV cache、group=1、固定拓扑 |

容器内固定输入：

- YAML：`/path/to/reference.yaml`。SHA256：`172da7f35012135b46857f578f19c58f70c3c9425d98cb4dbba649d10663b469`。
- 数据：`/path/to/shared_tokens.npz`，使用第一个样本的 input_ids/labels/loss_mask。SHA256：`2542d5ab1163d81b6a2efbb5d7a39b69910e7c662a8eaa7cd8a55b192283c12b`。
- 初始权重：`/path/to/initial_weights/initial_rank_{0..7}.npz`，逐文件 SHA256 见 `evidence/verification.json`。
- Transformers modeling 源文件 SHA256：`519f271a7e3b0a56655e1f92054d4191686b91af5df5bccdc6208b6569ef4d1a`；configuration 源文件 SHA256：`4306770517f8a4db3e1d33452c6eda7af2b48a5e45169ac8ed742edf3bbfec15`。本次验证针对这些安装源码，不泛化为任意同版本发行包。

验证证据在独立实验目录保存（未随本 PR 分发），包含：

- `verification.json`：输入哈希、运行版本、所有 rank 的数值和 FP32 位模式及断言核对结果。
- `first-loss-final/rank_*.json`：关闭诊断的最终复跑。
- `first-loss-trace-v4/rank_*.json`：248 项中间张量及同输入 RoPE 对照。
- `first-loss-v3/rank_*.json`：局部 RoPE 修复前的首 loss。
- `reference_result_rank_*.json`：既有参考实跑证据副本。
- `training_rank_*.json`：旧方案分项 loss 证据副本。
- `audit-v1/weight_mapping.json`、`probe-v3/`：转换和独立 MLA probe 的阶段证据；最终布局以 final 目录的 plan 为准。

## 13. 提交索引与复现

| Commit | 主要用途 |
| --- | --- |
| `810d4f3139e8f8e80041d74d4feb02344c5bc97b` | HF 迁移接口设计 |
| `34d1d481e6bddb8c27e0a57355bedb3cd915c625` | 配置转换、权重映射和逆转换核对 |
| `33a8d41194d3093b7de6c25268be3329d229cc87` | MLA 替换、并行计划、参数审计和 probe |
| `24bc1ca87849c80fdb2e735c5126ee19560a6111` | HF norm/residual/MLP/expert 前向适配 |
| `b41f2f818fbaf0e1813e3c4decd364ea0ef38a20` | 参考路由和辅助 loss 接 Hyper EP |
| `5dc70c7a6e6e79dcfe43534af26a655f1a9d3575` | MTP、256K 前向编排和 masked loss |
| `0f3be863d3237f7479f0e1ee39bc00b734f99c5c` | MLA 内局部 RoPE 舍入顺序修复 |
| `9bf9790c56fa0fde2af3cef21438160aabf41b9a` | 首 loss 入口和逐张量诊断 |

旧手写方案的所有历史提交保留在归档分支及其 README，不把它们表述为新 HF 路线已经移植的提交。发布状态以关联 PR 为准。

在已配置依赖的八卡 NPU 环境中运行（需 8 卡空闲）：

```bash
cd /path/to/hyper-parallel
HCCL_DETERMINISTIC=true python \
  -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m examples.training_demo.deepseek_v32_sft.probe \
  --reference-yaml /path/to/reference.yaml \
  --reference-weights /path/to/initial_weights \
  --first-loss-data /path/to/shared_tokens.npz \
  --output /path/to/first-loss-recheck
```

诊断时追加 `--reference-code /path/to/reference_package --reference-model-class ReferenceSFTModel`，类名应与实际参考包一致。新路径使用独立 venv 的 editable Hyper 安装，复用既有依赖包；未替换旧参考运行环境。

本地查看本轮代码统计：

```bash
git -C '/path/to/hyper-parallel' \
  diff --stat c20bd626f035972d0e3796ee21aef8ab48ecc705 codex/deepseek-v32-sft
```

阅读顺序建议：先看本报告的结果和修改边界，再用 Excel 筛选负责模块，最后按 commit 查看对应代码。Optimizer 负责人可先看第 10 节的未完成项。

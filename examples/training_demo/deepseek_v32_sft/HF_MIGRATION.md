# DeepSeek V3.2 SFT：HF 骨架与 Hyper 模块接入表

日期：2026-09-21。状态：接口调查与迁移设计，尚未实现或完成 NPU 验证。

## 目标与固定条件

使用真正的 Transformers `DeepseekV32ForCausalLM` 模型骨架，根据参考 YAML 构建裁剪模型，不加载预训练权重。保持共享数据、初始权重、262144 序列长度，以及 TP8 / EP8 / SP 开启 / CP1 / DP1 / PP1。精度要求仍为逐位一致；不能把高性能模块接通等同于精度通过。

Hyper 调查基线为 `c20bd626f035972d0e3796ee21aef8ab48ecc705`。容器实际安装 Transformers 5.13.1，其 `models/deepseek_v32/modeling_deepseek_v32.py` SHA256 为 `519f271a7e3b0a56655e1f92054d4191686b91af5df5bccdc6208b6569ef4d1a`，配置源文件 SHA256 为 `4306770517f8a4db3e1d33452c6eda7af2b48a5e45169ac8ed742edf3bbfec15`。这是对安装源码的调查，不表示任意同名版本或最新 HF 源码已验证兼容。

## 按重要程度排列的接入事项

| 部分 | 实际接口与差异 | 迁移处理及验收 |
| --- | --- | --- |
| Attention 与 mask | HF V3.2 Attention 内置 DSA indexer；旧参考使用完整因果 MLA。HF 模型上层还调用 `create_causal_mask`。 | 显式替换 attention，核对上层 mask 路径，禁止 256K 下创建 S×S 稠密 mask。先验证 MLA、RoPE、缩放及反向；不启用 CP。 |
| TP / EP / SP | Hyper 提供 mesh、分片计划和 EP compute factory；设置 ep_size 本身不会自动注入专家计算。 | 通过 Hyper 的实际 planner / dispatcher 接入，导出计划、参数布局及通信边界证据。不照搬旧模型手写通信。 |
| MTP 与模型输出 | HF 主干返回的 hidden 已经过最终 RMSNorm，参考 MTP 需要核对最终 norm 前的主干状态；所查 V3.2 实现没有现成参考 MTP 分支。 | 增加显式参考适配，保留 HF 子模块，取正确状态、共享 embedding/head、处理标签位移、mask 与 loss 系数。MTP 共享 head 不代表输入 embedding 与输出 head 互相绑权重。 |
| MoE 路由与专家 | HF gate 返回 logits、weights、indices；专家是打包的三维 gate_up_proj / down_proj。Hyper 通用 DeepSeek 路由会取 gate 返回的 logits 再计算路由。 | 核对 packed expert 的分片与 binder，明确避免无意重复路由。补齐参考的选专家顺序、归一化、辅助 loss 与 bias 更新语义，保留 Hyper EP 分发合并。 |
| Muon 与融合参数 | Hyper MLA 把 Q/KV 降维投影融合成 linear_qkv；专家 Gate/Up 也可能融合。 | 为每个逻辑矩阵保留原有更新、momentum 与权重映射边界。物理融合不能改变 Muon 的矩阵维度或正交化算法。逐步核对更新后权重。 |
| Decoder residual / dtype | HF Decoder 的 norm、attention、残差和 MLP 顺序不能自动保证与参考的转换边界一致。 | 先比较第一个不同张量；必要时只覆盖对应层 forward，复用 HF 参数与子模块。外部转 FP32 不能恢复内部已丢失的精度。 |
| SFT loss 与训练更新 | 默认 CausalLM loss 不能假设等同参考 mask、分母、MTP loss、裁剪、学习率及优化器分组。 | 保留明确的参考训练适配，比较各 loss 分量、未裁剪与裁剪后梯度、norm、更新权重及优化器状态。 |
| 局部反向与归约 | 旧实现中存在主干/MTP SwiGLU 舍入边界、专家梯度和 Muon norm 归约修复。 | 迁移经过证据确认的规则，不整包复制旧模型。局部自定义 backward 或参考归约显式启用，并记录形状/硬件适用范围。 |

## 可以复用的 Hyper 接口

- `hyper_parallel/components/modules/README.md`：`plan_overrides` 的 `match`、`module_type`、`replace_module`；替换发生在分片和 checkpoint 加载之前。
- `hyper_parallel/components/modules/mla_attention.py`：`MLAAttention(module=..., module_fqn=..., context=...)`，读取 HF 的 q_a_proj / kv_a_proj_with_mqa 等子模块，通过 `make_transforms()` 处理融合权重。
- `hyper_parallel/distributed/expert_parallel/recipes.py`：DeepSeek EP factory 接收 module、mesh、tp_mesh、cp_mesh、ep_mesh。参考适配需要显式注册适合 V3.2 布局和参考路由的 factory；不能仅改模型名称假装兼容。
- `hyper_parallel/distributed/expert_parallel/routing.py`：现有 sigmoid/group router 可作为语义核对起点，不作为已通过参考精度验收的实现。
- `hyper_parallel/distributed/mesh.py`：expert domain 来自同一 stage 的 DP×CP×TP 域，允许 8 卡 TP8/EP8；不要求 TP×EP=64 张卡。这里只确认拓扑可表达，尚未验证新模型运行。
- `hyper_parallel/models/adapter_spec.py`、`registry.py` 和已有模型 adapter：复用模型适配及注册机制。

共享专家已有 TP 归约边界时，不额外补一次 AllReduce。每次模型替换后重新核对计划，不能仅检查参数形状。

## 高性能模块启用顺序

先实现 HF 骨架上的参考语义，再逐项启用 RMSNorm、SwiGLUMLP、SharedExpert / GroupedExperts、MLAAttention。每项都比较输入输出、反向和更新后权重；具体顺序允许依据定位证据调整。整块模块不兼容时，先复用对应 functional 中已验证的计算。

模块所需 Omni 等算子依赖、当前 packed expert 兼容性、完整 mask 通路、融合后权重转换和 optimizer state 映射均属于待实测事项。当前没有证据可以宣称这些候选模块全部可直接使用。

## 修改边界和提交组织

参考特化代码计划集中于本示例目录的独立适配包，通过显式 builder / replacement / compute factory 启用；使用 HF 实际类构建并保留其子模块。若需要通用模型家族能力，再按仓库 `hyper_parallel/models/<family>/adapter` 组织单独实现。不要修改安装目录中的 Transformers，不做全局 monkey patch，不让参考舍入规则成为公共默认值。

提交按差异拆分：配置与权重映射、MLA/mask、Hyper TP/EP、MTP/loss、数值边界、优化器、各高性能模块及其验证。确认为公共缺陷的修复单独提交，提供独立证据。旧模型作为归档参考，不复制为新 HF 模型主体。

## 阶段验收

1. 固定源码、配置、数据；列出所有权重的映射/分片/融合逆变换，证明初始权重一致。
2. 从 embedding、各 norm、MLA、router、expert、residual、MTP 到各项 loss 寻找首个差异。
3. 在固定 256K 和并行条件下验证前向、反向、裁剪及第一步更新。
4. 验证 3–5 步所有 rank 的 loss、权重和优化器状态；单独披露未一致项目。
5. 逐项启用高性能替换后重新验收，再讨论 1000 步。

小张量接口测试只能用于定位，不能替代 256K 验收。旧归档的五步 loss 一致不能视为新方案的结果；旧方案第五步仍有更新权重差异，也应保留在问题清单中。

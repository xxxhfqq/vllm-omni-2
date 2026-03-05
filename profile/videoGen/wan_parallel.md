针对 vLLM-Omni 在 Wan2.2 模型（尤其是 14B MoE 版本）上的推理，并行配置通常通过 `DiffusionParallelConfig` 进行定义。系统的总显存占用和推理速度取决于张量并行（TP）、序列并行（SP）、CFG 并行（CFG）以及 HSDP 等维度的组合。

根据显卡数量（$N=2/4/8$），并行配置必须满足各维度乘积等于总卡数的原则，即 $TP \times SP \times CFG = N$ 。以下是针对不同显卡数量的典型配置组合：

### 1. 2 张显卡时的组合 ($N=2$)

在双卡环境下，通常选择单一维度的并行来突破显存瓶颈或提升速度。

| 组合类型 | 配置参数 (`DiffusionParallelConfig`) | 特性与适用场景 |
| --- | --- | --- |
| **纯张量并行 (TP2)** | `tensor_parallel_size=2` | **最常用**。将 DiT 权重分片，显著降低单卡显存压力，适合 14B 模型在 24GB 显卡上运行 。

 |
| **纯序列并行 (SP2)** | `ulysses_degree=2` 或 `ring_degree=2` | 针对长视频序列优化。将 81 帧 Token 序列平分，降低计算注意力时的峰值显存 。

 |
| **纯 CFG 并行 (CFG2)** | `cfg_parallel_size=2` | **侧重速度**。同时运行有条件和无条件推理，生成速度翻倍，但要求单卡能放下整个模型 。

 |
| **纯 HSDP (Standalone)** | `use_hsdp=True, hsdp_shard_size=2` | 基于 FSDP2 的权重分片，适用于非 Transformer 层的显存优化 。 |

### 2. 4 张显卡时的组合 ($N=4$)

四卡环境支持混合并行，可以在减少显存占用的同时提升序列处理能力。

| 组合类型 | 配置参数 (`DiffusionParallelConfig`) | 特性与适用场景 |
| --- | --- | --- |
| **纯张量并行 (TP4)** | `tensor_parallel_size=4` | 极大幅度降低显存占用，适合高分辨率（如 720p）的长视频生成 。

 |
| **纯序列并行 (SP4)** | `ulysses_degree=4` | 针对极长序列，Ulysses 模式在 4 卡内通信效率较高 。

 |
| **TP2 + SP2** | `tensor_parallel_size=2, ulysses_degree=2` | **平衡型**。兼顾权重分片和序列分片，是处理大规模 DiT 模型的推荐方案 。

 |
| **TP2 + CFG2** | `tensor_parallel_size=2, cfg_parallel_size=2` | 在保证显存不溢出的前提下，最大化利用 CFG 并行带来的推理加速 。

 |
| **HSDP + SP2** | `use_hsdp=True, ulysses_degree=2` | 混合分片模式，HSDP 负责全局权重分片，SP 负责注意力并行 。 |

### 3. 8 张显卡时的组合 ($N=8$)

八卡环境下可以实现“全维度并行”，这也是官方基准测试（如 173s 生成 81 帧）常用的配置环境 。

| 组合类型 | 配置参数 (`DiffusionParallelConfig`) | 特性与适用场景 |
| --- | --- | --- |
| **TP8** | `tensor_parallel_size=8` | Wan2.2 14B 拥有 40 个注意力头，可被 8 整除，TP8 能最大化减少单卡显存占用 。

 |
| **SP8** | `ulysses_degree=8` | 适用于超长视频或超高分辨率，通过 8 路 `all-to-all` 通信分担计算 。

 |
| **TP2 + SP4** | `tensor_parallel_size=2, ulysses_degree=4` | 适合跨节点或大集群，利用 TP 减少局部内存，SP 处理长上下文 。

 |
| **TP4 + CFG2** | `tensor_parallel_size=4, cfg_parallel_size=2` | **高性能方案**。通过 TP4 腾出空间，利用 CFG2 实现几乎两倍的吞吐量提升 。

 |
| **TP2 + SP2 + CFG2** | `tp=2, ulysses=2, cfg=2` | **全能组合**。在 81 帧视频生成中表现均衡，各环节通信开销相对平均 。

 |
| **HSDP + SP4** | `use_hsdp=True, ulysses_degree=4` | 集群级部署方案，HSDP 自动管理剩余权重的副本或进一步分片 。 |

### 配置限制与建议

* **整除性要求**：`tensor_parallel_size` 必须能被模型的注意力头数（Wan2.2 14B 为 40 个头）整除 。


* **VAE 显存溢出预防**：无论上述哪种组合，如果 VAE 阶段 OOM，应额外开启 `vae_patch_parallel_size`（建议设置为 2 或 4），这不占用 $TP \times SP \times CFG$ 的配额，而是针对 VAE 阶段的独立空间切分 。
* **混合 SP**：在 4 卡或 8 卡时，可以将 `ulysses_degree` 和 `ring_degree` 结合使用（如 $2 \times 2 = 4$），这在跨机房或超大规模序列生成中比纯 Ulysses 更具扩展性 。
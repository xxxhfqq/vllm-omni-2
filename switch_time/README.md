# 特性切换耗时测试（配置 1 ↔ 配置 26）

在 **不修改 vllm-omni-2 项目源码** 的前提下，通过重复「起服 → 停服 → 起另一配置」测量 **Stop / Startup-to-Ready / Switch** 时间。

- **配置 1（c1）**：SP=1 基准（1 卡，Cache=none，CFG=1，TP=1，无 CPU Offload，无 FP8）
- **配置 26（c26）**：SP=8 + FP8（8 卡，`--ulysses-degree 8`，可选 `--quantization-config` 开启 FP8）

**不向实例发推理请求**：脚本只轮询 `GET /v1/models` 判定就绪，不调用 T2I 或任何生成接口。

## 是否需要改项目源码

**不需要。** 仅通过 `vllm serve ...` 的 CLI 参数切换配置，用 shell 计时、轮询 `/v1/models` 判定 Ready，所有结果与日志落在本目录下。

## 集群资源建议（A100）

根据 [SAIKS HPC 用户指南](https://saids.hpc.gleamoe.com/) 的推荐：**每申请 1 张 GPU 卡搭配 16 核 CPU 和 128 GB 内存**。本测试需 8 卡，因此推荐：

- **GPU**：`--gres=gpu:8`
- **CPU**：`--cpus-per-task=128`（8 × 16）
- **内存**：`--mem=1024G`（8 × 128 GB）
- **任务数**：`-n 1`（只跑一个进程，避免多 task 抢同一端口）

脚本内已按上述写入 `#SBATCH`；若你站点的 A100 分区有不同规范，可自行改 `--cpus-per-task` / `--mem`。

## 用法

1. 将整个 **vllm-omni-2** 传到集群登录节点（如 [SAIKS HPC](https://saids.hpc.gleamoe.com/)）。
2. （可选）你可以手动先运行一次 `setup_env.sh`，预先创建/升级环境：
   ```bash
   cd /path/to/vllm-omni-2/switch_time
   bash setup_env.sh
   ```
   若环境已存在，`setup_env.sh` 会激活并升级依赖；否则创建 conda 环境并安装 vllm + `pip install -e .`。
3. **运行切换测试**：`run_switch_1_26.sh` 会在开头自动调用同目录下的 `setup_env.sh`（除非设置了 `SKIP_SETUP_ENV=1`）：
   - 交互试跑：`bash run_switch_1_26.sh`
   - 提交作业：`sbatch run_switch_1_26.sh`
4. 脚本默认 **每配置 11 次**：第 1 次为 **首次启动（冷启动参考）** 单独记录、不参与统计；第 2～11 次共 10 次「停→起」样本，用于计算 Stop/Startup/Switch 的 mean 与 std。可通过环境变量 `NUM_SAMPLES=10` 调整样本数。
5. 结果与日志在 `switch_time/logs/`：CSV 含 `is_warmup`、`cfg_from`/`cfg_to` 列；运行结束会打印基于非 warmup 样本的 mean/std 到 `*_stats.log`。

### 测试表中所有配置（run_switch_all.sh）

若需**同时覆盖测试表 5.2 中全部 26 种配置**的切换耗时，使用同目录的 `run_switch_all.sh`：

- **策略（分开测）**：与说明表一致，**配置 1 测完再测配置 2**，依次到 26，每种配置的测试彼此独立。对每个配置：先做 **1 次首次启动**（单独记录「首次 Startup(s)」，不参与 mean/std），再做 **10 次「停→起」**（停本配置 → 起本配置），得到该配置的 Stop/Startup/Switch 的 10 个样本并算 mean/std（共 11 次/配置）。
- **参数**：默认 `NUM_SAMPLES=10`（每配置 10 次停→起）；总次数 = 26 配置 ×（1 首次 + 10 样本）= 26×11 次启动。
- **资源**：需 8 卡（部分配置用 1/2/4 卡），SBATCH 已写 72 小时，建议用 sbatch 提交。
- **输出**：`logs/switch_all_*.csv` 列含 `config_id,run,first_startup_s,stop_s,startup_s,switch_s,ready_poll_s`（run=0 为首次启动行，run=1..10 为样本行）；`*_stats.log` 按 `config_id` 汇总每配置的首次 Startup、Stop/Startup/Switch 的 mean/std。
- **运行**：`bash run_switch_all.sh` 或 `sbatch run_switch_all.sh`（同样会先调用 `setup_env.sh`）。

## 自检（配置是否生效）

- 脚本开头会打印 **实际使用的 c1/c26 命令行** 及 `vllm serve -h` 中与 ulysses/ring/parallel 相关的参数。
- 第一次启动配置 26 后，会从对应 server 日志里 **grep** 是否出现 `Applying sequence parallelism` 或 `ulysses=`/`sp_size=` 等，便于核对 headless 路径下 SP 是否真生效。

## 计时方式与精度

- **Stop**：以 **PGID（进程组 ID）** 为目标：`kill -TERM -PGID` → 轮询进程组是否清空（最多 30s）→ 未清空则 `kill -KILL -PGID`；停后调用 `cleanup_gpu_residuals` 清理可能残留的 multiprocessing.spawn 孤儿进程（PPID=1）。
- **Startup**：在启动命令执行前打点，轮询 `/v1/models` 返回 200 时再打点，差值为 Startup-to-Ready。
- **精度**：`date +%s.%N` 为秒+纳秒；实际误差主要来自 curl 轮询间隔（1s），对几十秒级启动时间一般可接受。

## 输出说明

- `logs/switch_1_26_*.csv`：列含 direction、run、**is_warmup**、stop_s、startup_s、switch_s、**ready_poll_s**、cfg_from、cfg_to。
- `logs/switch_1_26_*_stats.log`：排除 warmup 后的 Stop/Startup/Switch 的 mean 与 std（每方向）。
- `logs/run_*.log`：当次运行的标准输出（含自检与配置命令）。
- `logs/server_*.log`：当次 vllm serve 的 stdout/stderr，便于排查启动失败或核对 SP 日志。
- **run_switch_all**：`logs/switch_all_*.csv`、`logs/run_all_*.log`、`logs/server_c{N}_*.log`（按配置编号 N 分文件，每配置一个 server 日志）。

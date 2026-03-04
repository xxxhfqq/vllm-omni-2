# switch_time 脚本代码审查报告

本文档记录对 `setup_env.sh`、`run_switch_1_26.sh`、`run_switch_all.sh` 及 `README.md` 的完整代码审查结论与已修复项。

---

## 1. 审查范围

| 文件 | 用途 |
|------|------|
| `setup_env.sh` | 创建/激活 conda 环境，安装 vllm==0.16.0 与 vllm-omni |
| `run_switch_1_26.sh` | 配置 1（SP=1）↔ 配置 26（SP=8+FP8）切换耗时测试 |
| `run_switch_all.sh` | 26 种配置「分开测」：每配置 1 次首次启动 + 20 次停→起 |
| `README.md` | 用法、资源建议、输出说明 |

---

## 2. 已修复问题

### 2.1 Conda 未在当前 Shell 生效（严重）

- **现象**：主脚本用 `bash setup_env.sh` 调用环境准备，`setup_env.sh` 在**子 Shell** 中执行，其中的 `conda activate` 只在该子 Shell 有效；返回主脚本后当前 Shell **未**激活 conda。
- **后果**：主脚本后续的 `python3 -c "import vllm_omni"` 及 `vllm serve` 可能使用系统 Python，导致导入失败或版本错误。
- **修复**：在 `run_switch_1_26.sh` 与 `run_switch_all.sh` 中，在调用 `setup_env.sh` 之后增加：
  - `module load`（与 setup_env 一致，保证 conda 可用）；
  - `source "$(conda info --base)/etc/profile.d/conda.sh"` 与 `conda activate $CONDA_ENV`，使**当前** Shell 使用正确环境。

### 2.2 `set -e` 与 `wait_ready` 超时导致脚本退出

- **现象**：`wait_ready` 超时时会 `return 1`。主脚本中用 `wait_s=$(wait_ready ...)` 捕获输出时，若 `wait_ready` 返回非零，该赋值语句整体退出码非零，在 `set -e` 下会**直接退出脚本**，无法走到 `if [ "$wait_s" = "-1" ]` 的错误处理。
- **修复**：对所有 `wait_ready` 的调用改为「赋值或回退」形式，例如：
  - `wait_s=$(wait_ready "http://127.0.0.1:${PORT}" 7200) || wait_s="-1"`
  - 同理处理 `wait_s26`、`wait_s1`、`wait_poll` 等，确保超时时脚本继续执行并写入 CSV 错误行。

### 2.3 run_switch_all 统计脚本对空/ERROR 行过滤不严

- **现象**：统计时用 `r.get("stop_s") not in ("ERROR", "")` 过滤样本行；若 CSV 中某列缺失则 `get` 返回 `None`，`None not in ("ERROR", "")` 为 True，该行会被保留，后续 `float(r["stop_s"])` 可能抛异常。
- **修复**：使用辅助判断，仅保留 `stop_s`/`startup_s`/`switch_s` 均非空、非 `"ERROR"` 的行（例如通过 `_valid(r, key)` 或等价条件），再参与 mean/std 计算。

### 2.4 README 中 CSV 列描述缺 ready_poll_s

- **现象**：`run_switch_1_26` 的 CSV 已包含 `ready_poll_s`，但 README 的「输出说明」未列出该列。
- **修复**：在 README 的 `switch_1_26_*.csv` 说明中补充 **ready_poll_s**。

---

## 3. 逻辑与一致性检查

### 3.1 与《Qwen-Image-特性切换测试说明表》的对应关系

- **run_switch_1_26.sh**：配置 1（SP=1 基准）与配置 26（SP=8，可选 FP8），每方向 21 次（1 次 warmup + 20 次样本），与说明表「首次启动单独记录 + 20 次样本」一致。
- **run_switch_all.sh**：按配置 1～26 顺序「分开测」，每配置 1 次首次启动（run=0）+ 20 次「停→起」（run=1..20），与说明表 5.2 及「分开测」策略一致。
- **get_config_params**：26 种配置的 GPU 数、SP/CFG/TP、cache、cpu_off、fp8 与说明表 5.2 一致。

### 3.2 计时与就绪判定

- **Stop**：`kill` 后 `wait $pid`，用 `date +%s.%N` 起止打点，合理。
- **Startup**：在启动命令**之前**打点，轮询 `/v1/models` 返回 200 后打点，差值为 Startup-to-Ready，与说明表一致。
- **ready_poll_s**：CSV 中记录实际轮询耗时，便于与 startup_s 对比、分析轮询间隔带来的偏差。

### 3.3 错误处理与稳健性

- 单次启动/就绪失败时：`log_error_full` 打印完整错误与 server 日志末尾，CSV 写入 ERROR，**不** `exit 1`，后续测试继续。
- `run_switch_all` 中若某次启动失败导致 `pid` 为空，下一轮「停→起」会判断 `[ -n "$pid" ]`，不执行 `stop_server_and_measure`，并将 `stop_s` 置空，统计阶段过滤掉该行，避免误算。

### 3.4 Slurm 与资源

- `-n 1`：单任务，避免多进程抢同一端口。
- 8 卡 A100 推荐：`--cpus-per-task=128`、`--mem=1024G`、`--gres=gpu:8`，与 README 及 HPC 建议一致。

---

## 4. 建议与注意事项（未改代码）

### 4.1 CONFIG_EXTRA 与 JSON 引号（run_switch_all）

- `get_config_params` 中 FP8 参数为 `--quantization-config '{\"method\":\"fp8\"}'`，经 `echo` 写入 `CONFIG_EXTRA` 再 `eval` 展开后，可能出现反斜杠被保留的情况，依 shell 版本不同可能影响 vllm 接收到的 JSON 字符串。
- **建议**：若在实际运行中发现配置 9/16/22/26 的 FP8 未生效，可检查 server 日志或改为通过环境变量/配置文件传递 JSON，避免多层引号与 eval。

### 4.2 时间精度与可移植性

- `date +%s.%N` 在 GNU date 下为秒+纳秒；部分 macOS 或旧系统可能不支持 `%N`。
- 脚本目标环境为 Linux（Slurm 集群），当前用法可接受；若需在 macOS 上试跑，可考虑用 Python 或兼容的日期命令替代。

### 4.3 端口释放确认

- `wait_port_released` 在 stop 后轮询最多 5 秒，若 `/v1/models` 仍返回 200 会打印注意信息并继续，有助于减少「端口未完全释放」导致的下一轮 startup 波动。

### 4.4 自检

- `run_switch_1_26.sh` 会打印 c1/c26 实际命令、`vllm serve -h` 中 ulysses/ring/parallel 相关参数，并在第一次启动配置 26 后从 server 日志 grep SP 相关输出，便于确认 SP 是否生效。

---

## 5. 小结

- **已修复**：Conda 未在当前 Shell 激活、`set -e` 下 wait_ready 超时退出、统计脚本对空/ERROR 行过滤、README CSV 列说明。
- **逻辑与说明表**：两脚本的流程、样本数、配置表与《Qwen-Image-特性切换测试说明表》一致。
- **建议关注**：CONFIG_EXTRA/FP8 引号在不同环境下的行为；若扩展到非 Linux 环境，需验证 `date +%s.%N` 的可用性。

审查完成日期：按当前代码版本记录。

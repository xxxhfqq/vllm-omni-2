#!/bin/bash
# 特性切换耗时测试：配置 1（SP=1 基准）↔ 配置 26（SP=8 + FP8）
# 不修改 vllm-omni-2 源码，仅通过 CLI 起停服务并计时。
# 重要：不向实例发送任何推理请求（不调用 T2I/生成接口），仅轮询 /v1/models 判定就绪。
# 用法：直接 bash run_switch_1_26.sh 或 sbatch 提交；脚本会自动调用同目录下的 setup_env.sh 准备环境。
# 作业管理参考：https://saids.hpc.gleamoe.com/
# A100 资源：按你集群实际调整。若提交报 Memory/node configuration not available，请用 sinfo -p A100 查看上限后改下面三行。
# 示例：8 卡时官网推荐 128 核、1024G；不少集群单节点为 512G，故默认 512G、64 核，你可改为 --mem=1024G --cpus-per-task=128
#SBATCH -J switch_1_26
#SBATCH -o %j_switch_1_26.out
#SBATCH -e %j_switch_1_26.err
#SBATCH -p A100
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:8
#SBATCH -t 48:00:00

set -e
# 测试循环内不因单次失败退出，见下方错误处理
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [ -z "$HF_HOME" ]; then
  _G=$(groups 2>/dev/null | awk '{print $1}')
  _U=$(whoami)
  if [ -n "$_G" ] && [ -d "/data2/$_G/$_U" ] && [ -w "/data2/$_G/$_U" ]; then
    export HF_HOME="/data2/$_G/$_U/xhf/hf_cache"
  else
    export HF_HOME="${HF_HOME:-$HOME/xhf/hf_cache}"
  fi
fi

# 工作目录：sbatch 时 Slurm 在计算节点可能从副本执行，$0 指向无写权限的 /var/spool/slurmd/…，故优先用提交目录
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
  WORK_DIR="${SLURM_SUBMIT_DIR}"
else
  WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
REPO_DIR="${REPO_DIR:-$(cd "$WORK_DIR/.." && pwd)}"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "$LOG_DIR"

MODEL="${MODEL:-Qwen/Qwen-Image}"
PORT="${PORT:-8099}"
# 说明表推荐：第 1 次为 warmup 单独记录，第 2～21 次共 20 次样本用于 mean/std
NUM_RUNS="${NUM_RUNS:-21}"
# 配置 26 是否开 FP8（若库支持扩散 FP8 可设为 1）
QUANT_FP8_26="${QUANT_FP8_26:-0}"

CONDA_ENV="${CONDA_ENV:-vllm_omni}"
JOBID="${SLURM_JOB_ID:-local}"
ts=$(date +%Y%m%d_%H%M%S)
RUN_LOG="${LOG_DIR}/run_${JOBID}_${ts}.log"
CSV="${LOG_DIR}/switch_1_26_${JOBID}_${ts}.csv"
STATS_LOG="${LOG_DIR}/switch_1_26_${JOBID}_${ts}_stats.log"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "===== switch_time test: config 1 (SP=1 baseline) <-> config 26 (SP=8 + FP8) ====="
echo "REPO_DIR=$REPO_DIR  WORK_DIR=$WORK_DIR  LOG_DIR=$LOG_DIR  NUM_RUNS=$NUM_RUNS (run 1 = warmup, 2..$NUM_RUNS = 样本)"

# 自动调用同目录的 setup_env.sh 准备/更新环境（可通过 SKIP_SETUP_ENV=1 跳过）
module purge 2>/dev/null || true
module load Anaconda3/2025.06 2>/dev/null || true
module load cuda/12.9.1 2>/dev/null || true
if [ "${SKIP_SETUP_ENV:-0}" != "1" ]; then
  bash "$WORK_DIR/setup_env.sh"
fi
# setup_env 在子 shell 中执行，当前 shell 需自行激活 conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV
cd "$REPO_DIR"
python3 -c "import vllm_omni" || { echo "ERROR: vllm_omni not importable. 请检查 setup_env.sh 是否执行成功。"; exit 1; }

# 打印实际使用的配置命令，便于核对「到底开了啥」
CFG_1_CMD="vllm serve $MODEL --omni --port $PORT  (1 GPU)"
if [ "$QUANT_FP8_26" = "1" ]; then
  CFG_26_CMD="vllm serve $MODEL --omni --port $PORT --ulysses-degree 8 --quantization-config '{\"method\":\"fp8\"}'  (8 GPU)"
else
  CFG_26_CMD="vllm serve $MODEL --omni --port $PORT --ulysses-degree 8  (8 GPU)"
fi
echo "===== 实际配置（用于 CSV cfg 列核对）====="
echo "  cfg_c1:  $CFG_1_CMD"
echo "  cfg_c26: $CFG_26_CMD"

# 自检：打印 vllm serve 中与 SP/parallel 相关的参数，确认 CLI 支持
echo "===== 自检: vllm serve 中 ulysses/ring/parallel 相关参数 ====="
vllm serve --help 2>&1 | grep -E 'ulysses|usp|ring|parallel' || true

# 等待 /v1/models 返回 200，超时 max_wait 秒
wait_ready() {
  local url="$1"
  local max_wait="${2:-7200}"
  local t=0
  while [ $t -lt "$max_wait" ]; do
    if curl -s -o /dev/null -w "%{http_code}" "${url}/v1/models" 2>/dev/null | grep -q 200; then
      echo "$t"
      return 0
    fi
    sleep 1
    t=$((t + 1))
    if [ $((t % 60)) -eq 0 ] && [ $t -gt 0 ]; then echo "    wait_ready ${t}s..." >&2; fi
  done
  echo "-1"
  return 1
}

# 停掉进程组并等待完全退出，返回 Stop 耗时（秒）。vllm 会起多子进程，仅 kill 主进程会导致端口仍被占用，
# 故用 setsid 启动使整组可杀；停止时 kill -TERM -$pid 杀整组，再 wait 主进程。
stop_server_and_measure() {
  local pid="$1"
  local T0 T1
  [ -z "$pid" ] && return
  T0=$(date +%s.%N)
  kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  T1=$(date +%s.%N)
  python3 -c "print(round($T1 - $T0, 2))"
}

# 停服后等待端口释放（/v1/models 不再 200），最多等 120s，超时则报错退出
wait_port_released() {
  local port="$1"
  local max_wait=120
  local i=0
  while [ $i -lt "$max_wait" ]; do
    if ! curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q 200; then
      return 0
    fi
    sleep 1
    i=$((i + 1))
    [ $((i % 30)) -eq 0 ] && [ $i -gt 0 ] && echo "    wait_port_released ${i}s..." >&2
  done
  echo "ERROR: 端口 ${port} 在 ${max_wait}s 内仍返回 200，未释放，退出" >&2
  exit 1
}

# 启动配置 1（1 GPU），用 setsid 使 vllm 及其子进程同属一进程组，停服时可整组 kill 释放端口
start_config_1() {
  local port="$1"
  local log="$2"
  export CUDA_VISIBLE_DEVICES=0
  setsid vllm serve "$MODEL" --omni --port "$port" >> "$log" 2>&1 &
  echo $!
}

# 启动配置 26（8 GPU, SP=8），同上用 setsid
start_config_26() {
  local port="$1"
  local log="$2"
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  if [ "$QUANT_FP8_26" = "1" ]; then
    setsid vllm serve "$MODEL" --omni --port "$port" --ulysses-degree 8 --quantization-config '{"method":"fp8"}' >> "$log" 2>&1 &
  else
    setsid vllm serve "$MODEL" --omni --port "$port" --ulysses-degree 8 >> "$log" 2>&1 &
  fi
  echo $!
}

# 自检：从 server 日志中 grep 是否出现 SP 生效（ulysses/ring/sp_size）
check_sp_in_log() {
  local log="$1"
  if [ -f "$log" ]; then
    if grep -E "Applying sequence parallelism|ulysses=|ring=|sp_size=" "$log" 2>/dev/null | head -3; then
      echo "  [自检] 在 server 日志中看到 SP 相关输出，配置可能已生效"
    else
      echo "  [自检] 未在 server 日志中看到 SP 相关行，请核对 headless 路径是否传入 ulysses_degree"
    fi
  fi
}

# 单次测试失败时：打印完整报错信息（含相关 server 日志末尾），不退出，后续测试继续
log_error_full() {
  local msg="$1"
  local log_file="$2"
  echo ""
  echo "========== ERROR =========="
  echo "  $msg"
  echo "  Time: $(date -Iseconds 2>/dev/null || date)"
  if [ -n "$log_file" ] && [ -f "$log_file" ]; then
    echo "  --- 相关 server 日志末尾 (last 300 lines): $log_file ---"
    tail -n 300 "$log_file" 2>/dev/null || true
    echo "  --- 以上为 $log_file 末尾 ---"
  fi
  echo "========== 继续后续测试 =========="
  echo ""
}

echo "direction,run,is_warmup,stop_s,startup_s,switch_s,ready_poll_s,cfg_from,cfg_to" >> "$CSV"

for run in $(seq 1 "$NUM_RUNS"); do
  [ "$run" -eq 1 ] && is_warmup=1 || is_warmup=0
  [ "$is_warmup" -eq 1 ] && echo "========== Run $run / $NUM_RUNS (warmup，单独记录、不参与统计) ==========" || echo "========== Run $run / $NUM_RUNS =========="

  echo "--- 1 -> 26 ---"
  SERVER_LOG_1="${LOG_DIR}/server_c1_${JOBID}_r${run}.log"
  SERVER_LOG_26="${LOG_DIR}/server_c26_${JOBID}_r${run}.log"
  pid_1=$(start_config_1 "$PORT" "$SERVER_LOG_1")
  echo "  config 1 started pid=$pid_1"
  wait_s=$(wait_ready "http://127.0.0.1:${PORT}" 7200) || wait_s="-1"
  if [ "$wait_s" = "-1" ]; then
    kill $pid_1 2>/dev/null; wait $pid_1 2>/dev/null || true
    log_error_full "config 1 failed to become ready (1->26 run $run)" "$SERVER_LOG_1"
    echo "1->26,$run,$is_warmup,ERROR,ERROR,ERROR,ERROR,c1,c26" >> "$CSV"
  else
    echo "  config 1 ready in ${wait_s}s"
    stop_s=$(stop_server_and_measure "$pid_1")
    wait_port_released "$PORT"
    echo "  stop config 1: ${stop_s}s"
    sleep 2
    T_start=$(date +%s.%N)
    pid_26=$(start_config_26 "$PORT" "$SERVER_LOG_26")
    echo "  config 26 started pid=$pid_26"
    wait_s26=$(wait_ready "http://127.0.0.1:${PORT}" 7200) || wait_s26="-1"
    if [ "$wait_s26" = "-1" ]; then
      kill $pid_26 2>/dev/null; wait $pid_26 2>/dev/null || true
      log_error_full "config 26 failed to become ready (1->26 run $run)" "$SERVER_LOG_26"
      echo "1->26,$run,$is_warmup,ERROR,ERROR,ERROR,ERROR,c1,c26" >> "$CSV"
    else
      if [ "$run" -eq 1 ]; then check_sp_in_log "$SERVER_LOG_26"; fi
      startup_s=$(python3 -c "print(round($(date +%s.%N) - $T_start, 2))")
      switch_s=$(python3 -c "print(round($stop_s + $startup_s, 2))")
      echo "  startup config 26: ${startup_s}s  ready_poll=${wait_s26}s  switch_1_26: ${switch_s}s"
      echo "1->26,$run,$is_warmup,$stop_s,$startup_s,$switch_s,$wait_s26,c1,c26" >> "$CSV"
      stop_s26=$(stop_server_and_measure "$pid_26")
      wait_port_released "$PORT"
      echo "  stop config 26: ${stop_s26}s"
    fi
  fi
  sleep 2

  echo "--- 26 -> 1 ---"
  pid_26=$(start_config_26 "$PORT" "$SERVER_LOG_26")
  echo "  config 26 started pid=$pid_26"
  wait_s=$(wait_ready "http://127.0.0.1:${PORT}" 7200) || wait_s="-1"
  if [ "$wait_s" = "-1" ]; then
    kill $pid_26 2>/dev/null; wait $pid_26 2>/dev/null || true
    log_error_full "config 26 failed to become ready (26->1 run $run)" "$SERVER_LOG_26"
    echo "26->1,$run,$is_warmup,ERROR,ERROR,ERROR,ERROR,c26,c1" >> "$CSV"
  else
    stop_s=$(stop_server_and_measure "$pid_26")
    wait_port_released "$PORT"
    echo "  stop config 26: ${stop_s}s"
    sleep 2
    T_start=$(date +%s.%N)
    pid_1=$(start_config_1 "$PORT" "$SERVER_LOG_1")
    echo "  config 1 started pid=$pid_1"
    wait_s1=$(wait_ready "http://127.0.0.1:${PORT}" 7200) || wait_s1="-1"
    if [ "$wait_s1" = "-1" ]; then
      kill $pid_1 2>/dev/null; wait $pid_1 2>/dev/null || true
      log_error_full "config 1 failed to become ready (26->1 run $run)" "$SERVER_LOG_1"
      echo "26->1,$run,$is_warmup,ERROR,ERROR,ERROR,ERROR,c26,c1" >> "$CSV"
    else
      startup_s=$(python3 -c "print(round($(date +%s.%N) - $T_start, 2))")
      switch_s=$(python3 -c "print(round($stop_s + $startup_s, 2))")
      echo "  startup config 1: ${startup_s}s  ready_poll=${wait_s1}s  switch_26_1: ${switch_s}s"
      echo "26->1,$run,$is_warmup,$stop_s,$startup_s,$switch_s,$wait_s1,c26,c1" >> "$CSV"
      stop_server_and_measure "$pid_1" >/dev/null
      wait_port_released "$PORT"
    fi
  fi
  sleep 2
done

# 基于非 warmup 样本（run>=2）计算 mean/std，写入 stats 日志
echo "===== 统计（排除 run 1 warmup，样本 run 2..$NUM_RUNS）=====" | tee "$STATS_LOG"
python3 - "$CSV" << 'PYSTATS' | tee -a "$STATS_LOG"
import csv
import sys
from pathlib import Path
csv_path = Path(sys.argv[1])
if not csv_path.exists():
    sys.exit(0)
with open(csv_path) as f:
    rows = list(csv.DictReader(f))
rows = [r for r in rows if r.get("is_warmup") == "0"]
# 跳过失败行（stop_s/startup_s/switch_s 为 ERROR）
rows = [r for r in rows if r.get("stop_s") != "ERROR" and r.get("startup_s") != "ERROR" and r.get("switch_s") != "ERROR"]
if not rows:
    print("No non-warmup rows for stats.")
    sys.exit(0)
for direction in ("1->26", "26->1"):
    sub = [r for r in rows if r["direction"] == direction]
    if not sub:
        continue
    stop = [float(r["stop_s"]) for r in sub]
    startup = [float(r["startup_s"]) for r in sub]
    switch = [float(r["switch_s"]) for r in sub]
    n = len(sub)
    mu_stop = sum(stop) / n
    mu_startup = sum(startup) / n
    mu_switch = sum(switch) / n
    var_stop = sum((x - mu_stop) ** 2 for x in stop) / n
    var_startup = sum((x - mu_startup) ** 2 for x in startup) / n
    var_switch = sum((x - mu_switch) ** 2 for x in switch) / n
    sig_stop = var_stop ** 0.5
    sig_startup = var_startup ** 0.5
    sig_switch = var_switch ** 0.5
    print(f"{direction}: n={n}  Stop mu={mu_stop:.2f} sigma={sig_stop:.2f}  Startup mu={mu_startup:.2f} sigma={sig_startup:.2f}  Switch mu={mu_switch:.2f} sigma={sig_switch:.2f}")
PYSTATS

echo "===== Done. CSV: $CSV  Stats: $STATS_LOG  Run log: $RUN_LOG ====="

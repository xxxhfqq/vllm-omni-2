#!/bin/bash
# 在运行 run_switch_1_26.sh 前执行一次：创建 conda 环境并安装 vllm + vllm-omni。
# 建议在登录节点或已申请资源的节点上执行；与切换测试脚本解耦，避免测试时重复安装。
# 用法：cd /path/to/vllm-omni-2/switch_time && bash setup_env.sh
set -e
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$WORK_DIR/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-vllm_omni}"

module purge 2>/dev/null || true
module load Anaconda3/2025.06 2>/dev/null || true
module load cuda/12.9.1 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -q "^${CONDA_ENV} "; then
  echo "Conda env ${CONDA_ENV} already exists. Activating and upgrading."
  conda activate $CONDA_ENV
  cd "$REPO_DIR"
  pip install -q "vllm==0.16.0"
  pip install -q -e .
else
  echo "Creating conda env: $CONDA_ENV"
  conda create -n $CONDA_ENV python=3.12 -y
  conda activate $CONDA_ENV
  cd "$REPO_DIR"
  pip install "vllm==0.16.0" --torch-backend=auto
  pip install -e .
fi
python3 -c "import vllm_omni; print('vllm_omni OK')"
echo "Setup done. Run: bash run_switch_1_26.sh  or  sbatch run_switch_1_26.sh"

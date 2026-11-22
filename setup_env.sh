#!/usr/bin/env bash
# 在共享服务器上建独立 conda 环境 dmem，不改动系统环境和其他人的环境。
# 用法：bash setup_env.sh        （之后每次使用前：conda activate dmem）
# 版本组合：python 3.10 + torch 2.6.0 (cu124) + 匹配的 fairseq2 + sonar-space（依据 SONAR / fairseq2 README）。
# 前提：nvidia-smi 显示的 "CUDA Version" ≥ 12.4。如果更低，见 AGENT.md 步骤 1 的处理办法。
set -euo pipefail
ENV=${ENV:-dmem}
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  echo "conda 环境 $ENV 已存在，直接使用（如需重建：conda env remove -n $ENV）"
else
  conda create -n "$ENV" python=3.10 -y
fi
conda activate "$ENV"
if ! conda list -n "$ENV" --json libsndfile | python -c 'import json,sys; sys.exit(not any(p["name"] == "libsndfile" for p in json.load(sys.stdin)))'; then
  conda install -c conda-forge libsndfile -y        # fairseq2 的音频依赖，装在环境内，不需要 sudo
fi
pip install --upgrade pip
python download_env_wheels.py
pip install .wheels/*.whl
pip install 'fairseq2==0.4.6' --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124
pip install 'sonar-space==0.4.0'
pip install -r requirements.txt
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda available', torch.cuda.is_available())
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
print('sonar import ok')
PY
pip show torch | grep -i version
echo "环境就绪：conda activate $ENV"

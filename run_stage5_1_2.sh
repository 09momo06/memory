#!/usr/bin/env bash
set -euo pipefail
cd /home/lj/wtr/dmem
source project_env.sh
eval "$(conda shell.bash hook)"
conda activate dmem
export CUDA_VISIBLE_DEVICES=4
exec 9>logs/stage5_1_2.lock
flock -n 9
step=preflight
finish() {
  status=$?
  printf '%s\n' "$status" > logs/stage5_1_2.exit
  if [ "$status" -ne 0 ]; then
    printf '\n### [5.1/5.2 执行失败] %s\n\n步骤：%s。退出码：%s。后续命令未执行，详见 logs/stage5_1_2.log 及该步骤日志。\n' "$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')" "$step" "$status" >> results/RUNLOG.md
  fi
}
trap finish EXIT
python - <<'PY'
import json, numpy as np, torch
from pathlib import Path
assert json.load(open('results/stage4_training.json'))['passed']
assert Path('results/E0_CONFIRMED').exists()
assert Path('results/STAGE3_COMPLETE').exists()
assert Path('results/STAGE4_COMPLETE').exists()
assert Path('ckpt/main.pt').is_file()
x=np.load('data/test_x.npy',mmap_mode='r')
s=np.load('data/test_seg_sim_K8.npy',mmap_mode='r')
meta=np.load('data/test_meta.npy',mmap_mode='r')
assert x.shape[1:]==(16,1024)
assert s.shape==(len(x),9,16)
assert meta.shape==(len(x),2)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
for name in ('e1_none_sim.json','e1_main_cold.json','e3_none.json','e3_main.json'):
    assert not (Path('results')/name).exists(), f'Refusing to overwrite an existing result: {name}'
print('Preflight passed:',torch.__version__,torch.cuda.get_device_name(0),'test blocks',len(x),flush=True)
PY
run_one() {
  step="$1"
  shift
  started=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')
  printf '\n### [%s] %s（%s → ' "$stage" "$step" "$started" >> results/RUNLOG.md
  nohup "$@" > "logs/$step.log" 2>&1 < /dev/null &
  task_pid=$!
  printf '%s\n' "$task_pid" > "logs/$step.pid"
  wait "$task_pid"
  printf '%s）\n\n命令：`' "$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')" >> results/RUNLOG.md
  printf '%s ' "$@" >> results/RUNLOG.md
  printf '`\n\n结果：\n```text\n' >> results/RUNLOG.md
  tail -n 5 "logs/$step.log" >> results/RUNLOG.md
  printf '\n```\n\n验收：命令退出码 0；输出内容待本轮最终核验。改动：无。\n' >> results/RUNLOG.md
  echo "Completed $step"
}
stage=5.1
run_one eval_none_sim python eval.py --mode none --tag none_sim
run_one eval_main_cold python eval.py --ckpt ckpt/main.pt --mode cold --tag main_cold
stage=5.2
run_one tree_none python tree_eval.py --tag none
run_one tree_main python tree_eval.py --ckpt ckpt/main.pt --tag main
echo 'Four commands completed; output validation pending.'

#!/usr/bin/env bash
set -euo pipefail
cd /home/lj/wtr/dmem
source project_env.sh
eval "$(conda shell.bash hook)"
conda activate dmem
hold_pid=''
start_hold() {
  if [ ! -f logs/gpu_hold.ready ]; then
    rm -f logs/gpu_hold.stop
    nohup /usr/bin/python3 gpu_hold.py > logs/gpu_hold_stage4.log 2>&1 < /dev/null &
    hold_pid=$!
    printf '%s\n' "$hold_pid" > logs/gpu_hold.pid
    for attempt in {1..60}; do
      if [ -f logs/gpu_hold.ready ]; then return; fi
      if ! kill -0 "$hold_pid" 2>/dev/null; then return 1; fi
      sleep 1
    done
    return 1
  fi
}
finish() {
  status=$?
  trap - EXIT
  printf '%s\n' "$status" > logs/stage4.exit
  export DMEM_STAGE4_STATUS="$status"
  python - <<'PY'
import datetime,json,os
from pathlib import Path
status=int(os.environ['DMEM_STAGE4_STATUS'])
p=Path('results/stage4_training.json')
with Path('results/RUNLOG.md').open('a') as f:
    f.write(f'\n### [阶段 4] 训练与验收结束（{datetime.datetime.now().astimezone().isoformat()}）\n\n退出码：{status}。\n')
    if p.exists():
        report=json.loads(p.read_text())
        f.write('\n结果：\n```json\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n```\n')
        f.write('\n验收：'+('通过' if report['passed'] else '未通过，须按指南降低学习率重训一次')+'。\n')
    else:
        f.write('\n验收：未通过；训练或检查未完成，见 logs/main.log 和 logs/stage4.log。\n')
    f.write('\n改动：只新增训练编排 run_stage4.sh 和验收脚本 validate_stage4.py，不修改训练算法、网络或损失；GPU 任务结束后恢复原 40 GiB 保留。\n')
PY
  start_hold || true
  if [ -n "$hold_pid" ]; then wait "$hold_pid" || true; fi
}
trap finish EXIT
python - <<'PY'
import datetime,json
from pathlib import Path
assert json.loads(Path('results/stage4_preflight.json').read_text())['passed']
assert not Path('ckpt/main.pt').exists(), 'Preserve existing checkpoint before another attempt'
with Path('results/RUNLOG.md').open('a') as f:
    f.write(f'\n### [阶段 4] 训练开始（{datetime.datetime.now().astimezone().isoformat()}）\n\n命令：`python train.py --multilevel --out ckpt/main.pt`，默认 steps=30000、bs=256、lr=3e-4。所有 GPU 命令限制 CUDA_VISIBLE_DEVICES=4。只训练本轮主模型，日志 logs/main.log，模型 ckpt/main.pt。\n')
PY
touch results/STAGE4_STARTED
touch logs/gpu_hold.stop
for attempt in {1..60}; do
  [ ! -f logs/gpu_hold.ready ] && break
  sleep 1
done
test ! -f logs/gpu_hold.ready
args=(--multilevel --out ckpt/main.pt)
if [ "${DMEM_TRAIN_BS:-256}" != 256 ]; then args+=(--bs "$DMEM_TRAIN_BS"); fi
if [ "${DMEM_TRAIN_LR:-0.0003}" != 0.0003 ]; then args+=(--lr "$DMEM_TRAIN_LR"); fi
printf '%q ' python train.py "${args[@]}" > logs/main_command.txt
printf '\n' >> logs/main_command.txt
nohup python train.py "${args[@]}" > logs/main.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/main.pid
wait "$(cat logs/main.pid)"
start_hold
python validate_stage4.py > logs/validate_stage4.log 2>&1
printf 'STAGE 4 PASSED\n'

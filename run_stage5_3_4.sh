#!/usr/bin/env bash
set -euo pipefail
cd /home/lj/wtr/dmem
source project_env.sh
eval "$(conda shell.bash hook)"
conda activate dmem
export CUDA_VISIBLE_DEVICES=4
exec 9>logs/stage5_3_4.lock
flock -n 9
step=preflight
finish() {
  status=$?
  printf '%s\n' "$status" > logs/stage5_3_4.exit
  if [ "$status" -ne 0 ]; then
    printf '\n### [5.3/5.4 执行中断] %s\n\n步骤：%s；退出码：%s。详见 logs/stage5_3_4.log 和对应日志。\n' "$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')" "$step" "$status" >> results/RUNLOG.md
  fi
}
trap finish EXIT
python - <<'PY'
import json
from pathlib import Path
assert json.load(open('results/stage5_1_2_validation.json'))['passed']
for name in ['none_sim','main_cold','tree_main_m1','tree_main_m4']:
    assert not Path(f'results/dec_{name}.json').exists(), f'Existing result: {name}'
print('5.3 preflight passed',flush=True)
PY
run_one() {
  step="$1"
  shift
  started=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')
  echo "Starting $step at $started"
  nohup "$@" > "logs/$step.log" 2>&1 < /dev/null &
  task_pid=$!
  printf '%s\n' "$task_pid" > "logs/$step.pid"
  wait "$task_pid"
  printf '\n### [%s] %s（%s → %s）\n\n命令：`' "$stage" "$step" "$started" "$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')" >> results/RUNLOG.md
  printf '%s ' "$@" >> results/RUNLOG.md
  printf '`\n\n结果：\n```text\n' >> results/RUNLOG.md
  tail -n 5 "logs/$step.log" >> results/RUNLOG.md
  printf '\n```\n\n验收：命令退出码0；输出内容待最终核验。改动：见前置准备。\n' >> results/RUNLOG.md
  echo "Completed $step"
}
stage=5.3
run_one decode_none_sim python decode_eval.py --src results/restored_none_sim.pt --name none_sim
run_one decode_main_cold python decode_eval.py --src results/restored_main_cold.pt --name main_cold
run_one decode_tree_main_m1 python decode_eval.py --src results/restored_tree_main_m1.pt --name tree_main_m1
run_one decode_tree_main_m4 python decode_eval.py --src results/restored_tree_main_m4.pt --name tree_main_m4
stage='5.4 实体对齐所需的 E0 补充'
python - <<'PY'
import datetime, json, shutil
from pathlib import Path
root=Path.cwd()
backup=root/'results/backups'/('ceiling_before_entity_alignment_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir(parents=True,exist_ok=False)
for name in ['dec_ceiling_allblocks.json','samples_ceiling_allblocks.txt','predictions_ceiling_allblocks.jsonl']:
    p=root/'results'/name
    if p.exists():
        shutil.copy2(p,backup/name)
(root/'logs/ceiling_entity_alignment_backup.txt').write_text(str(backup))
print('Preserved earlier ceiling outputs:',backup,flush=True)
PY
run_one decode_ceiling_entity_alignment python decode_eval.py --src ceiling --name ceiling_allblocks
stage=5.4
run_one plot python plot.py
echo 'Decode and plotting commands completed. Detailed result audit and required summary pending.'

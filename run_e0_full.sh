#!/usr/bin/env bash
set -euo pipefail
cd /home/lj/wtr/dmem
source project_env.sh
eval "$(conda shell.bash hook)"
conda activate dmem
finish() {
  DMEM_FULL_STATUS=$?
  trap - EXIT
  printf '%s\n' "$DMEM_FULL_STATUS" > logs/e0_full.exit
  export DMEM_FULL_STATUS
  python - <<'PY'
import datetime,json,os
from pathlib import Path
status=int(os.environ['DMEM_FULL_STATUS'])
with Path('results/RUNLOG.md').open('a') as f:
    f.write(f'\n### [用户追加全量 E0] 结束（{datetime.datetime.now().astimezone().isoformat()}）\n\n命令：`python eval_sonar_full.py --bs 32`。退出码：{status}。\n')
    if status==0:
        f.write('\n结果：\n```json\n'+Path('results/dec_ceiling_full.json').read_text()+'\n```\n\n验收：全量推理完成，等待产物核验及用户审阅；未进入阶段 3。全部 REF/HYP 见 results/predictions_ceiling_full.jsonl 和 results/samples_ceiling_full.txt。\n')
        Path('results/E0_FULL_AWAITING_CONFIRMATION').touch()
    else:
        f.write('\n验收：未通过，须检查 logs/e0_full.log；未进入阶段 3。\n')
PY
  if [ ! -f logs/gpu_hold.ready ]; then
    rm -f logs/gpu_hold.stop
    nohup /usr/bin/python3 gpu_hold.py > logs/gpu_hold_after_e0_full.log 2>&1 < /dev/null &
    DMEM_FULL_HOLD_PID=$!
    printf '%s\n' "$DMEM_FULL_HOLD_PID" > logs/gpu_hold.pid
    wait "$DMEM_FULL_HOLD_PID" || true
  fi
}
trap finish EXIT
python - <<'PY'
import datetime
from pathlib import Path
with Path('results/RUNLOG.md').open('a') as f:
    f.write(f'\n### [用户追加全量 E0] 开始（{datetime.datetime.now().astimezone().isoformat()}）\n\n范围：全部 10 段 LoCoMo 对话，13626 句，包含原分块时舍弃的 90 句尾部。复用已有 13536 句 SONAR 编码，仅补编码 90 句；不改变训练预处理和记忆块设计。\n\n环境与超参数适配：新增 eval_sonar_full.py 按对话记录解码进度，保留全部 REF/HYP；解码 batch_size 从原默认 5 调到 32（允许的 bs 超参数），模型、eng_Latn、默认 beam search 和 max_seq_len=256 保持一致。原结果不覆盖；全量指标另存 dec_ceiling_full.json，原完整块范围另算 dec_ceiling_allblocks.json，以便区分数据范围。\n\n同时按用户前次前缀规则统计全部句子的说话人不一致类型，保存全部证据，未知前缀先标记待审阅，不自动当作换人。\n')
PY
touch logs/gpu_hold.stop
for DMEM_FULL_ATTEMPT in {1..60}; do
  [ ! -f logs/gpu_hold.ready ] && break
  sleep 1
done
if [ -f logs/gpu_hold.ready ]; then echo 'Reservation did not release.'; exit 1; fi
nohup python eval_sonar_full.py --bs 32 > logs/e0_full.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/e0_full.pid
wait "$(cat logs/e0_full.pid)"

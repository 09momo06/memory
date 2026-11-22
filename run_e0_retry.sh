#!/usr/bin/env bash
set -euo pipefail
cd /home/lj/wtr/dmem
source project_env.sh
eval "$(conda shell.bash hook)"
conda activate dmem
stamp() { TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S'; }
note() { printf '%s\n' "$*" >> results/RUNLOG.md; }
finish() {
  local status=$?
  printf '%s\n' "$status" > logs/e0_retry.exit
  if [ "$status" -ne 0 ]; then
    note ""
    note "### [2.1 重试中断] $(stamp)，退出码 $status"
    if [ ! -f logs/gpu_hold.ready ]; then
      rm -f logs/gpu_hold.stop
      nohup /usr/bin/python3 gpu_hold.py > logs/gpu_hold_after_failure.log 2>&1 < /dev/null &
      local hold_pid=$!
      printf '%s\n' "$hold_pid" > logs/gpu_hold.pid
      wait "$hold_pid" || true
    fi
  fi
}
trap finish EXIT
note ""
note "### [2.1 网络适配重试] $(stamp)"
note '前次官方编码模型下载不足预期字节数，fairseq2 默认清理了不完整文件。新增 prefetch_sonar.py，对原官方 encoder、decoder、tokenizer 进行保留 partial 的续传，按 HTTP 长度和可用 ZIP CRC 验证并记录本地 SHA256；直接预填同一 fairseq2 缓存，不修改模型和方法。下载期间保留原 GPU 占用，全部准备好后才交接给 E0 命令。'
python prefetch_sonar.py > logs/prefetch_sonar.log 2>&1

stage=2.1
started=$(stamp)
# Release only the reservation created by this experiment, using its stop file.
touch logs/gpu_hold.stop
for attempt in {1..60}; do
  [ ! -f logs/gpu_hold.ready ] && break
  sleep 1
done
if [ -f logs/gpu_hold.ready ]; then
  echo 'Reservation did not release; refusing to proceed.'
  exit 1
fi
nohup python encode.py --splits test > logs/encode_test.log 2>&1 < /dev/null &
encode_pid=$!
printf '%s\n' "$encode_pid" > logs/encode_test.pid
wait "$encode_pid"
nohup python decode_eval.py --src ceiling --name ceiling > logs/decode_ceiling.log 2>&1 < /dev/null &
decode_pid=$!
printf '%s\n' "$decode_pid" > logs/decode_ceiling.pid
wait "$decode_pid"
python - <<'PY'
import json
from pathlib import Path
metrics = json.loads(Path('results/dec_ceiling.json').read_text())
samples = Path('results/samples_ceiling.txt').read_text()
lines = samples.splitlines(keepends=True)
count = 0
selected = []
for line in lines:
    selected.append(line)
    if line.startswith('HYP:'):
        count += 1
        if count == 20:
            break
assert count == 20
with open('results/RUNLOG.md', 'a') as f:
    f.write('\n### [2.1] E0 结果\n\n命令：`python encode.py --splits test; python decode_eval.py --src ceiling --name ceiling`。\n\n结果（JSON 原样，全部字段）：\n\n```json\n')
    f.write(json.dumps(metrics, ensure_ascii=False, indent=2))
    f.write('\n```\n\n前 20 对 REF/HYP 原样：\n\n```text\n')
    f.writelines(selected)
    f.write('\n```\n\n验收：通过（输出文件和 20 对样例已记录）；质量须由用户审阅。改动：encode.py 提前保存原句文件，见前文。\n')
PY
note "开始时间：$started；结束时间：$(stamp)。"
note '状态：E0 完成，按强制关卡暂停。未执行阶段 3、训练、E1/E2/E3；等待用户明确指令“继续”。'
touch results/E0_AWAITING_CONFIRMATION
echo 'E0 complete. Awaiting human confirmation; stage 3 was not started.'
# Keep a bounded reservation during the human review pause.
rm logs/gpu_hold.stop
nohup /usr/bin/python3 gpu_hold.py > logs/gpu_hold_after_e0.log 2>&1 < /dev/null &
hold_pid=$!
printf '%s\n' "$hold_pid" > logs/gpu_hold.pid
wait "$hold_pid"

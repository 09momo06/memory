#!/usr/bin/env bash
# Execute only stages 0--2, recording checks and stopping at the E0 human gate.
set -euo pipefail
cd /home/lj/wtr/dmem
source project_env.sh
eval "$(conda shell.bash hook)"

stamp() { TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S'; }
note() { printf '%s\n' "$*" >> results/RUNLOG.md; }
stage=0.2
finish() {
  local status=$?
  printf '%s\n' "$status" > logs/run_to_e0.exit
  if [ "$status" -ne 0 ]; then
    note ""
    note "### [执行中断] 步骤 $stage（$(stamp)）"
    note "验收：未通过，后台执行退出码 $status。须检查对应步骤日志，后续步骤未执行。"
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

while [ ! -f logs/setup_env.exit ]; do sleep 5; done
if [ "$(cat logs/setup_env.exit)" != 0 ]; then
  echo 'Environment setup failed; inspect logs/setup_env.log.'
  exit 1
fi
conda activate dmem
python - <<'PY' > logs/env_check.log 2>&1
import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
print('torch', torch.__version__, 'cuda available', torch.cuda.is_available())
print('sonar import ok')
assert torch.__version__ == '2.6.0+cu124'
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print('device', torch.cuda.get_device_name(0))
PY
note ""
note "### [0.2 验收] 环境就绪（$(stamp)）"
note '命令：`source project_env.sh; conda activate dmem; python`（版本及 SONAR 导入检查，输出见 logs/env_check.log）。'
note '结果：'
note '```text'
cat logs/env_check.log >> results/RUNLOG.md
note '```'
note '验收：通过。改动：见 0.1 续所列环境适配。'

stage=1.1
started=$(stamp)
mkdir -p data/raw ckpt results logs
curl --fail -L --retry 3 -o data/raw/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json > logs/locomo_download.log 2>&1
if ! python prep_data.py --inspect > logs/inspect.log 2>&1; then
  cp logs/inspect.log logs/inspect_initial_failure.log
  export HF_ENDPOINT=https://hf-mirror.com
  python prep_data.py --inspect > logs/inspect.log 2>&1
fi
python - <<'PY' >> logs/inspect.log 2>&1
import json
from datasets import load_dataset
ds = load_dataset('nayohan/multi_session_chat')
for split in ds:
    cols = set(ds[split].column_names)
    assert {'dialogue', 'speaker', 'session_id'} <= cols, cols
    assert 'dialoug_id' in cols or 'dialogue_id' in cols, cols
data = json.load(open('data/raw/locomo10.json'))
assert len(data) == 10
for sample in data:
    conv = sample['conversation']
    assert 'session_1' in conv
    for key, turns in conv.items():
        if key.startswith('session_') and isinstance(turns, list):
            assert all('speaker' in t and 'text' in t for t in turns), key
print('Raw data schema checks passed.')
PY
note ""
note "### [1.1] 下载并检查（$started → $(stamp)）"
note '命令：`curl -L -o data/raw/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json; python prep_data.py --inspect`，随后验证字段和样本数量。'
note '结果：'
note '```text'
cat logs/inspect.log >> results/RUNLOG.md
note '```'
note '验收：通过。改动：无。'
if [ "${HF_ENDPOINT:-}" = https://hf-mirror.com ]; then
  note '环境适配：首次检查失败，按指南设置 HF_ENDPOINT=https://hf-mirror.com 后通过；首次错误见 logs/inspect_initial_failure.log。'
fi

stage=1.2
started=$(stamp)
python prep_data.py > logs/prep_data.log 2>&1
note ""
note "### [1.2] 预处理（$started → $(stamp)）"
note '命令：`python prep_data.py`。'
note '结果：'
note '```text'
cat logs/prep_data.log >> results/RUNLOG.md
note '```'
python - <<'PY'
import json
for split in ('train', 'valid', 'test'):
    convs = [json.loads(s) for s in open(f'data/{split}_convs.jsonl')]
    assert all(len(c) % 16 == 0 for c in convs)
    assert all(': ' in s for c in convs for s in c)
    n = sum(len(c) // 16 for c in convs)
    if split == 'train':
        assert n >= 5000, f'STOP: only {n} train blocks'
        assert n >= 10000, f'Acceptance not met: only {n} train blocks (target >=10000)'
print('Preprocessing checks passed.')
PY
note '验收：通过（训练块数至少 10000，句子带说话人前缀）。改动：无。'

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

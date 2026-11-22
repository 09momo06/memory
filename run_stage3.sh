#!/usr/bin/env bash
set -euo pipefail
cd /home/lj/wtr/dmem
source project_env.sh
eval "$(conda shell.bash hook)"
conda activate dmem
stage=3.1
hold_pid=''
stamp() { TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S'; }
note() { printf '%s\n' "$*" >> results/RUNLOG.md; }
start_hold() {
  if [ ! -f logs/gpu_hold.ready ]; then
    rm -f logs/gpu_hold.stop
    nohup /usr/bin/python3 gpu_hold.py > logs/gpu_hold_stage3.log 2>&1 < /dev/null &
    hold_pid=$!
    printf '%s\n' "$hold_pid" > logs/gpu_hold.pid
    for attempt in {1..60}; do
      [ -f logs/gpu_hold.ready ] && return
      sleep 1
    done
    return 1
  fi
}
finish() {
  status=$?
  trap - EXIT
  printf '%s\n' "$status" > logs/stage3.exit
  if [ "$status" -ne 0 ]; then
    note ""
    note "### [阶段 3 中断] 步骤 $stage（$(stamp)）"
    note "验收：未通过，退出码 $status。检查本步骤日志，后续步骤未执行。"
  fi
  start_hold || true
  if [ -n "$hold_pid" ]; then wait "$hold_pid" || true; fi
}
trap finish EXIT
note ""
note "### [用户确认] 进入阶段 3（$(stamp)）"
note '用户指令：“进入阶段3”。E0 关卡已获确认；本次按顺序执行 3.1、3.2、3.3。新增 run_stage3.sh 编排既定命令和显存交接、validate_stage3.py 核验实际数据形状/有限性/标准化/日程结构及多层节点与同对话连续块的对应关系；不修改实验方法。'
touch results/E0_CONFIRMED
rm -f results/E0_AWAITING_CONFIRMATION results/E0_FULL_AWAITING_CONFIRMATION
started=$(stamp)
test "$(cat logs/test_common_stage3.exit)" = 0
rg -q '^all ok$' logs/test_common_stage3.log
# The smoke script includes small GPU runs with fake data.
touch logs/gpu_hold.stop
for attempt in {1..60}; do
  [ ! -f logs/gpu_hold.ready ] && break
  sleep 1
done
test ! -f logs/gpu_hold.ready
nohup bash smoke_test.sh > logs/smoke_stage3.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/smoke_stage3.pid
wait "$(cat logs/smoke_stage3.pid)"
rg -q '^SMOKE TEST PASSED$' logs/smoke_stage3.log
note ""
note "### [3.1] 核心测试及冒烟测试（$started → $(stamp)）"
note '命令：`python test_common.py; bash smoke_test.sh`。'
note '结果：test_common 输出 all ok；smoke_test 输出 SMOKE TEST PASSED。冒烟结果仅来自 data_fake，不是实验数值。完整输出见 logs/test_common_stage3.log 和 logs/smoke_stage3.log。'
note '验收：通过。改动：无。'
stage=3.2
started=$(stamp)
nohup python encode.py --splits train valid > logs/encode.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/encode.pid
wait "$(cat logs/encode.pid)"
start_hold
nohup python encode.py --normalize > logs/normalize_stage3.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/normalize_stage3.pid
wait "$(cat logs/normalize_stage3.pid)"
python validate_stage3.py --mode encoding > logs/validate_stage3_encoding.log 2>&1
note ""
note "### [3.2] 编码与标准化（$started → $(stamp)）"
note '命令：`python encode.py --splits train valid; python encode.py --normalize; python validate_stage3.py --mode encoding`。'
note '结果：'
note '```text'
cat logs/normalize_stage3.log >> results/RUNLOG.md
note '```'
note '验收：通过；数组形状、句子和块元数据对应、有限性、训练标准化均值和标准差检查已保存 results/stage3_encoding_validation.json。改动：无。编码后恢复 40 GiB 保留，CPU 步骤继续。'
stage=3.3
started=$(stamp)
nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python make_segs.py --workers 8 > logs/make_segs_stage3.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/make_segs_stage3.pid
wait "$(cat logs/make_segs_stage3.pid)"
nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python make_tree_data.py --workers 8 > logs/make_tree_data_stage3.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/make_tree_data_stage3.pid
wait "$(cat logs/make_tree_data_stage3.pid)"
python validate_stage3.py --mode complete > logs/validate_stage3_complete.log 2>&1
note ""
note "### [3.3] 退化日程与多层训练数据（$started → $(stamp)）"
note '命令：`python make_segs.py --workers 8; python make_tree_data.py --workers 8; python validate_stage3.py --mode complete`。'
note '结果：'
note '```text'
cat logs/make_segs_stage3.log logs/make_tree_data_stage3.log >> results/RUNLOG.md
note '```'
note '验收：通过；全部数据形状、日程每步合并一对且不拆分、抽样最相似相邻合并一致性、多层同对话连续块对应关系，见 results/stage3_validation.json。'
note '改动：CPU 多进程 workers 从默认 16 调至 8，与本会话可用 CPU 数一致；每个工作进程 BLAS/OMP 限为单线程，避免重复并行。未裁剪多层数据、未改变方法。'
touch results/STAGE3_COMPLETE
printf 'STAGE 3 PASSED\n'

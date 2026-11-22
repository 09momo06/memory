#!/usr/bin/env bash
# 用假数据把流水线（不含 SONAR 编码/解码）跑一遍，几分钟内完成。任何一步报错都要先修。
set -euo pipefail
PY=${PY:-python}
D=data_fake; R=results_fake; C=ckpt_fake
rm -rf $D $R $C; mkdir -p $R $C
$PY test_common.py
$PY make_fake_data.py --out $D
$PY make_segs.py --data $D --workers 4 --kinds sim fixed
$PY make_tree_data.py --data $D --max_n 200 --workers 4
ARGS="--data $D --steps ${STEPS:-60} --bs 32"
$PY train.py $ARGS --multilevel --out $C/main.pt
$PY train.py $ARGS --sched fixed --out $C/fixed.pt
E="--data $D --results $R"
$PY eval.py $E --mode none --tag none_sim
$PY eval.py $E --ckpt $C/main.pt --mode cold --tag main_cold
$PY eval.py $E --sched fixed --mode none --tag none_fixed
$PY eval.py $E --sched fixed --ckpt $C/fixed.pt --mode cold --tag fixed_cold
T="--data $D --results $R --n 20"
$PY tree_eval.py $T --tag none
$PY tree_eval.py $T --ckpt $C/main.pt --tag main
$PY analyze_merge.py --data $D --results $R --n 200 --restored $R/restored_main_cold.pt > /dev/null
$PY entity.py | tail -1
$PY -c "from prep_data import conv_to_sents; print(conv_to_sents([('Ann', 'Hi! I moved to Boston in 2021. Cool.')]))"
$PY plot.py $R > /dev/null || echo "plot skipped (matplotlib missing?)"
ls $R/restored_tree_main_m1.pt > /dev/null
echo "SMOKE TEST PASSED"

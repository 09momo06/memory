#!/usr/bin/env bash
# 多层树（7 个算子）的冒烟测试：假数据，跑通 数据生成 → 多层训练 → 树评测 → 汇总。数值没有意义。
set -euo pipefail
PY=${PY:-python}
D=data_fake_tree; R=results_fake_tree; C=ckpt/ops_tree_fake
rm -rf $D $R $C; mkdir -p $R
$PY test_ops.py | tail -1
$PY test_tree_ops.py | tail -1
$PY make_fake_data.py --out $D > /dev/null
$PY make_ops.py --data $D > /dev/null
for op in merge_adj merge_fixed merge_global dct random select_fl svd; do
  $PY make_tree_ops.py --data $D --op $op --max_n 100 > /dev/null
  $PY train_ops.py --data $D --op $op --multilevel --steps ${STEPS:-20} --bs 16 --eval_every 10 --out $C/$op.pt | tail -1
  $PY eval_tree_ops.py --data $D --results $R --op $op --mode none --n 10 > /dev/null
  $PY eval_tree_ops.py --data $D --results $R --op $op --mode cold --n 10 --ckpt $C/$op.pt | grep -o '"consistency_err": [^,}]*'
done
$PY summarize_tree_ops.py --results $R > /dev/null
ls $R/restored_dct_cold_chain_m1.pt > /dev/null
rm -rf $D $C
echo "TREE OPS SMOKE TEST PASSED"

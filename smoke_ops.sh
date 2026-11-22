#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python}
mkdir -p data/ops results/ops ckpt/ops
D=$(mktemp -d data/ops/smoke_data_XXXXXX)
R=$(mktemp -d results/ops/smoke_results_XXXXXX)
C=$(mktemp -d ckpt/ops/smoke_ckpt_XXXXXX)
trap 'rm -rf -- "$D" "$C"' EXIT
$PY test_ops.py
$PY make_fake_data.py --out "$D" > /dev/null
$PY make_ops.py --data "$D"
for op in merge_adj merge_fixed merge_global dct random select_fl svd; do
  $PY eval_ops.py --data "$D" --results "$R" --op "$op" --mode none > /dev/null
  $PY train_ops.py --data "$D" --op "$op" --steps "${STEPS:-20}" --bs 16 --eval_every 10 --out "$C/$op.pt" | tail -1
  $PY eval_ops.py --data "$D" --results "$R" --op "$op" --mode cold --ckpt "$C/$op.pt" | grep -o '"consistency_err": [^,}]*'
done
$PY summarize_ops.py --results "$R" --ceiling none --ckpt_dir "$C" > /dev/null
printf 'Fake smoke outputs (not experiment metrics): %s\n' "$R"
echo 'OPS SMOKE TEST PASSED'

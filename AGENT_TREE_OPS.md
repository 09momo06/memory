# AGENT_TREE_OPS.md：7 个算子的多层压缩还原实验

你是执行实验的 coding agent。单层算子对比（`AGENT_OPS.md`）已经完成。本实验对同样的 7 个算子做**多层树**的压缩和还原：每个算子训练一个多层模型，然后在和主实验完全相同的 208 组测试数据上，评测单层、平衡树、链三种结构。从上到下执行，每一步满足"验收"后才能进入下一步。

## 背景（一段话）

每个算子对任意 16×1024 的节点给出投影 P_0 … P_8，以及有序正交基 B（16×8，B Bᵀ = P_8）。压缩是 z = Bᵀx（8×1024）；两个子节点的 z 直接拼接成上一层节点（16×1024）；还原时先展开 B z，再用 cold 采样逐层还原，网络以所还原节点的层级（0 = 叶子，1，2）为条件。树结构和主实验 `tree_eval.py` 一致：链 = 原始设计的不平衡树（m1,m2→c1；c1,m3→c2；c2,m4→c3，只存 c3，叶子被压缩 3/3/2/1 次；代码和 json 里的键名是 `chain`，汇报时称为"不平衡树"）、平衡树（每个叶子 2 次）、单层（1 次）。代码在 `tree_ops.py`。

## 硬性规则

1. 沿用 `AGENT.md` 的共享服务器规则：只用 conda 环境 `dmem`，只用 `CUDA_VISIBLE_DEVICES` 指定的那一张 H800，不碰别人的进程。
2. **不要改动**：`ops.py`、`ops_model.py`、`tree_ops.py` 里的算子、网络、采样和树结构；训练超参数（7 个算子必须用相同的 `--steps --bs --lr`；显存不够时可以统一调小 `--bs`，7 个都用同一个值）。
3. **不要修改或覆盖已有结果**：`results/`、`results/ops/`、`ckpt/main.pt`、`ckpt/ops/`、`data/*.npy`、`data/ops/*_P_{op}.npy`。本实验的输出只写到 `data/ops/train_*_lv*`、`ckpt/ops_tree/`、`results/ops_tree/`、`logs/tree_*`。
4. 不准编造数值。改动任何代码都要写进 `results/ops_tree/RUNLOG_TREE_OPS.md`，并重跑 `python test_ops.py`、`python test_tree_ops.py`、`bash smoke_tree_ops.sh`。
5. 遇到"停止条件"就停下来报告。

---

## 步骤 1：同步代码并测试

把本地这些文件拷到服务器的项目目录：

- 修改过的（覆盖）：`ops.py ops_model.py test_ops.py train_ops.py`
- 新增的：`tree_ops.py test_tree_ops.py make_tree_ops.py eval_tree_ops.py summarize_tree_ops.py smoke_tree_ops.sh AGENT_TREE_OPS.md`

修改过的文件**向后兼容**：已有的单层 checkpoint（`ckpt/ops/*.pt`）和 `eval_ops.py` 照常可用。

```bash
conda activate dmem && export CUDA_VISIBLE_DEVICES=<卡号>
mkdir -p ckpt/ops_tree results/ops_tree logs
python test_ops.py           # 必须输出 all ops ok
python test_tree_ops.py      # 必须输出 all tree ops ok
bash smoke_tree_ops.sh       # 必须输出 TREE OPS SMOKE TEST PASSED（假数据，数值没有意义）
```

**附加核对**（只读，不写文件）：确认已有单层结果没有被影响：

```bash
python - <<'EOF'
import json; e = json.load(open('results/ops/e1_dct_cold.json')); print('dct 单层还原 top1 =', e['top1'])
EOF
```

## 步骤 2：生成多层训练数据

```bash
for op in merge_adj merge_fixed merge_global dct random select_fl svd; do
  python make_tree_ops.py --op $op
done
```

单进程，每个算子估计 1–3 分钟。

**验收**：每个算子生成 4 个文件：`data/ops/train_x_lv{1,2}_{op}.npy`（[n,16,1024]）、`data/ops/train_P_lv{1,2}_{op}.npy`（[n,9,16,16]）。n 应和主实验 `data/train_x_lv1.npy` 的行数一致（主实验是 14641），把每个算子的 n 写进 RUNLOG。

## 步骤 3：训练 7 个多层模型

每个模型约占 4–6GB 显存。空闲显存 ≥ 45GB 时 7 个可以同时跑，否则分批。

```bash
for op in merge_adj merge_fixed merge_global dct random select_fl svd; do
  nohup python train_ops.py --op $op --multilevel --steps 30000 --out ckpt/ops_tree/$op.pt > logs/tree_$op.log 2>&1 &
done
```

30000 步和主实验的多层模型一样。上次单层实验里，DCT、random、SVD 在第 15000 步还在提升。

**验收**：每个日志开头有 `samples per level [28384, n, n]`，最后一行是 `saved best ckpt … (step S, val X, no-restore Y, reduction R%)`。把 7 行都抄进 RUNLOG。X ≥ Y 不算失败，照实记录。

## 步骤 4：树评测

```bash
for op in merge_adj merge_fixed merge_global dct random select_fl svd; do
  python eval_tree_ops.py --op $op --mode none
  python eval_tree_ops.py --op $op --mode cold --ckpt ckpt/ops_tree/$op.pt
done
```

**验收**：
- 生成 14 个 `results/ops_tree/e3_{op}_{none,cold}.json`，每个文件里 `n_groups` = 208（和主实验相同）。
- 每个 cold 文件的 `consistency_err` < 1e-3。
- 每个文件的 `leaf_min_pairwise_diff` 三个数都 > 0（不同叶子的还原结果不能完全相同）。
- **核对**：`e3_merge_adj_none.json` 里 single 的 4 个叶子，Top-1 应和主实验 `results/e3_none.json` 的 single 一致（同一批组，同样的单层退化；本地假数据上已验证完全一致。如果只差 ≤ 0.1 个百分点，属于浮点误差，记录即可）。平衡树和链的数字**可以不一致**，因为通用框架里上层节点存的是"√组大小 × 组均值"，和主实验的"组均值"相差一个缩放系数。两者都写进 RUNLOG。

**停止条件**：`n_groups` ≠ 208；任何一个 `consistency_err` ≥ 1e-3；任何一个 `leaf_min_pairwise_diff` 为 0；merge_adj 的 single 和主实验相差超过 0.1 个百分点。

## 步骤 5：解码（先做必做项）

**必做**：7 个算子的链结构最旧叶子 m1（被压缩 3 次）：

```bash
for op in merge_adj merge_fixed merge_global dct random select_fl svd; do
  python decode_eval.py --src results/ops_tree/restored_${op}_cold_chain_m1.pt --name tree_${op}_cold_chain_m1 --results results/ops_tree
done
```

**有时间再做**：链的 m4（被压缩 1 次）和平衡树的 m1（被压缩 2 次）：

```bash
for op in merge_adj merge_fixed merge_global dct random select_fl svd; do
  for tag in cold_chain_m4 cold_balanced_m1; do
    python decode_eval.py --src results/ops_tree/restored_${op}_${tag}.pt --name tree_${op}_${tag} --results results/ops_tree
  done
done
```

## 步骤 6：汇总

```bash
python summarize_tree_ops.py     # → results/ops_tree/summary_tree_ops.md, tree_ops_depth.png
```

在 `RUNLOG_TREE_OPS.md` 末尾写"结果摘要"，**只陈述数据**，回答这 5 个问题：

1. 按"链结构还原后的平均 Top-1"从高到低给 7 个算子排序，同时列出单层和平衡树的值。
2. 每个算子：被压缩 1 → 2 → 3 次时，还原后的 Top-1 分别是多少？哪个算子下降最慢？
3. 每个算子在单层、平衡树、链上的还原增益（还原 − 不还原，百分点）。增益随深度变大还是变小？
4. 链结构中，每个算子的 m1 和 m2 相比如何（两者都被压缩 3 次）？**特别关注 dct**：如果 m2 明显低于 m1，记录下来（已知可能原因：DCT 系数按频率排列，拼接后第二个孩子的直流分量落在序列中间，下一层低通时损失更大）。
5. 链 m1 解码后，各算子的 BLEU-4、专名召回、换人率。

## 交付

- `results/ops_tree/summary_tree_ops.md`、`tree_ops_depth.png`、`RUNLOG_TREE_OPS.md`
- `results/ops_tree/e3_*.json`、`dec_tree_*.json`、`samples_tree_*.txt`
- `ckpt/ops_tree/*.pt` 和 `*_log.json`

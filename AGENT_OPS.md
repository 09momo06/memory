# AGENT_OPS.md：单层算子对比实验

你是执行实验的 coding agent。主实验（`AGENT.md`）已经完成，本实验在它的数据基础上，比较 **7 种冷扩散退化算子**在**单层**压缩和还原上的效果。从上到下执行，每一步满足"验收"后才能进入下一步。

## 背景（一段话）

所有算子都把一个记忆块（16 句 × 1024 维）压到 8 个向量，形式统一为**嵌套正交投影** P_0 = I ⊃ P_1 ⊃ … ⊃ P_8（秩从 16 降到 8）。还原网络（`ops_model.py` 的 `RestorerP`）对所有算子完全相同，条件信息是 P_t 的每一行。采样是 Cold Diffusion 的 x ← x − P_t x̂0 + P_{t−1} x̂0，它能严格保持压缩结果不变。算子定义见 `ops.py` 顶部的 `SIDE_INFO`：

| 键 | 算子 | 附加信息 |
|---|---|---|
| `merge_adj` | 相邻相似合并（主方法） | 约 8 字节 |
| `merge_fixed` | 固定池化 | 无 |
| `merge_global` | 非相邻相似合并 | 约 16 字节 |
| `dct` | 序列低通（DCT 模糊） | 无 |
| `random` | 随机正交投影 | 无 |
| `select_fl` | 句子选择，删一半（facility location） | 2 字节 |
| `svd` | 块内低秩（线性压缩最优参考） | 约 256 字节（1.6%） |

## 硬性规则

1. 沿用 `AGENT.md` 的共享服务器规则：只在 conda 环境 `dmem` 里操作，只用 `CUDA_VISIBLE_DEVICES` 指定的那一张 H800，不碰别人的进程。
2. **不要改动** `ops.py` 的算子定义、`ops_model.py` 的网络和采样、训练超参数（所有算子必须用相同的 `--steps --bs --lr`）。只有在显存不够时可以统一调小 `--bs`，而且 **7 个算子都要用同一个值**。
3. **不要修改或覆盖主实验的文件**：`results/` 下原有的文件、`ckpt/main.pt`、`data/*.npy`。本实验的输出只写到 `data/ops/`、`ckpt/ops/`、`results/ops/`、`logs/ops_*`。
4. 不准编造数值。改动任何代码都要写进 `results/ops/RUNLOG_OPS.md`，并重跑 `python test_ops.py` 和 `bash smoke_ops.sh`。
5. 遇到"停止条件"就停下来报告。

---

## 步骤 1：同步代码并测试

把本地新文件拷到服务器的项目目录：`ops.py ops_model.py test_ops.py make_ops.py train_ops.py eval_ops.py summarize_ops.py smoke_ops.sh AGENT_OPS.md`。同时确认 `entity.py decode_eval.py plot.py` 是最新版本（`decode_eval.py` 里有 `spk_sets`）。

```bash
conda activate dmem && export CUDA_VISIBLE_DEVICES=<卡号>
mkdir -p ckpt/ops results/ops logs
python test_ops.py       # 必须输出 all ops ok
bash smoke_ops.sh        # 必须输出 OPS SMOKE TEST PASSED（假数据，数值没有意义）
```

## 步骤 2：预计算投影矩阵

```bash
python make_ops.py       # 单进程，7 个算子 × 3 个划分，估计几分钟
```

**验收**：`data/ops/{train,valid,test}_P_{op}.npy` 共 21 个文件，形状为 [N,9,16,16]，test 的 N = 846。

## 步骤 3：不还原基线（不需要训练）

```bash
for op in merge_adj merge_fixed merge_global dct random select_fl svd; do
  python eval_ops.py --op $op --mode none
done
```

**验收**：生成 7 个 `results/ops/e1_{op}_none.json`。`merge_adj` 的 top1 应该和主实验 `e1_none_sim.json` 的 0.4834 **完全一致**，因为它们是同一个退化。**不一致就停下来报告**。

## 步骤 4：训练 7 个还原网络

每个模型约占 4–6GB 显存。卡上空闲显存 ≥ 45GB 时，可以 7 个一起跑；不够就分两批（例如先跑前 4 个）。

```bash
for op in merge_adj merge_fixed merge_global dct random select_fl svd; do
  nohup python train_ops.py --op $op --out ckpt/ops/$op.pt > logs/ops_$op.log 2>&1 &
done
```

默认每个训练 15000 步（主实验在第 4500 步就达到最优，所以 15000 步足够），每 500 步验证一次，验证 loss 创新低时自动保存。

**验收**：每个日志最后一行是 `saved best ckpt … (step S, val X, no-restore Y, reduction R%)`。把 7 行都抄进 RUNLOG。某个算子的 X ≥ Y 不算失败，照实记录就行，表示这个算子下还原网络没有学到东西。

## 步骤 5：还原评测

```bash
for op in merge_adj merge_fixed merge_global dct random select_fl svd; do
  python eval_ops.py --op $op --mode cold --ckpt ckpt/ops/$op.pt
done
```

**验收**：生成 7 个 `results/ops/e1_{op}_cold.json`，每个文件里的 `consistency_err` < 1e-3。
**停止条件**：任何一个 `consistency_err` ≥ 1e-3。

## 步骤 6：解码成文本

先解码"还原"的 7 个结果，有时间再解码"不还原"的 7 个：

```bash
for op in merge_adj merge_fixed merge_global dct random select_fl svd; do
  python decode_eval.py --src results/ops/restored_${op}_cold.pt --name ops_${op}_cold --results results/ops
done
# 有时间再做：
for op in merge_adj merge_fixed merge_global dct random select_fl svd; do
  python decode_eval.py --src results/ops/restored_${op}_none.pt --name ops_${op}_none --results results/ops
done
```

## 步骤 7：汇总

```bash
python summarize_ops.py   # → results/ops/summary_ops.md, ops_curve.png
```

在 `RUNLOG_OPS.md` 末尾写"结果摘要"，**只陈述数据**，回答这 5 个问题：

1. 按"还原后 Top-1"从高到低给 7 个算子排序。
2. 每个算子的还原增益（还原 Top-1 − 不还原 Top-1）分别是多少？哪个算子增益最大？
3. `merge_adj`（单层模型）的还原 Top-1 和主实验多层模型的 0.5146 相比如何？
4. `select_fl` 和 `merge_adj` 比：BLEU-4、专名召回、换人率分别是多少？
5. 哪个算子的还原 Top-1 最接近 `svd`？有没有算子在还原之后超过了 `svd` 的"不还原"Top-1？

## 交付

- `results/ops/summary_ops.md`、`ops_curve.png`、`RUNLOG_OPS.md`
- `ckpt/ops/*.pt` 和 `*_log.json`

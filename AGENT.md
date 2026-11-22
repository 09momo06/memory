# AGENT.md：实验执行说明

你是执行实验的 coding agent。按本文件**从上到下**执行，每一步满足"验收"条件后才能进入下一步。背景说明在 `实验指南.md`，有疑问先查那里。

## 目标

在共享 Linux 服务器上，使用**1 张 H800** 和**独立的 conda 环境**，完成 E0 到 E3。交付：`results/summary.md`、`results/e2_curve.png`、`results/e3_depth.png`、`results/samples_*.txt`、`results/RUNLOG.md`。

## 硬性规则

1. **共享服务器礼仪**：
   - 只在 conda 环境 `dmem` 里安装东西，**不要**用 sudo，不要往 base 环境或系统 Python 里装包。
   - 所有 GPU 命令之前先 `export CUDA_VISIBLE_DEVICES=<一张空闲的 H800>`，只占用这一张卡。
   - 不要 kill 不是自己启动的进程。所有文件都只写在本项目目录里。
2. **不要改动方法设计**：块长 16；压缩到 8；退化为每步合并 1 对最相似的相邻段，共 8 步；朴素拼接；Cold Diffusion 采样；只训练一个多层模型。**不要**加摘要、残差或锚点，**不要**改成阈值合并或变长节点。
3. 允许的修改只有三类：真正的 bug、环境或数据字段的适配、超参数（`--steps --bs --lr --workers`）。每处改动都要写进 RUNLOG，说明改了什么、为什么改。
4. 改了 `common.py` 以后，必须重新跑 `python test_common.py` 和 `bash smoke_test.sh`，两个都通过才能继续。
5. **不准编造数值**。表格里的每个数字都必须来自 `results/*.json`。没跑的步骤写"未运行"并说明原因。
6. 遇到"停止条件"就停下来报告，不要自己换方案。只有本文写明的备选方案可以自行启用。
7. 耗时长的命令用 `nohup … > logs/xxx.log 2>&1 &` 放到后台跑，再定期查看日志。

## RUNLOG 格式

每完成一步，就在 `results/RUNLOG.md` 里追加一段：

```
### [步骤号] 名称  (开始时间 → 结束时间)
命令：…
结果：关键数值或输出的最后几行
验收：通过 / 未通过（原因）
改动：无 / 具体描述
```

---

# 阶段 0：环境

## 步骤 0.1：检查 GPU

```bash
nvidia-smi
```

记下：一张显存占用低、没有别人进程在跑的 H800 的编号，以及右上角的 "CUDA Version"。

**停止条件**：没有空闲的 H800。

## 步骤 0.2：建立 conda 环境

```bash
bash setup_env.sh          # 建立 conda 环境 dmem：python3.10 + torch2.6.0(cu124) + fairseq2 + sonar-space + 其他依赖
conda activate dmem
export CUDA_VISIBLE_DEVICES=<步骤 0.1 选定的卡号>
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**验收**：脚本最后打印 `sonar import ok`；torch 版本是 `2.6.0+cu124`，`cuda.is_available()` 为 True。

- 如果 `pip install sonar-space` 之后 torch 版本被改掉了：按 SONAR README（https://github.com/facebookresearch/SONAR），重新安装和 torch 版本**严格匹配**的 fairseq2 和 torch。
- 如果驱动的 CUDA Version 低于 12.4：改用驱动支持的 torch 版本，以及对应的 fairseq2 wheel（地址形如 `https://fair.pkg.atmeta.com/fairseq2/whl/pt<版本>/<cuXXX>`）。修改 `setup_env.sh` 里的这两行，并写进 RUNLOG。
- **备选方案（可以自行启用）**：大约 3 小时还是装不上 SONAR 的话，在 dmem 环境里执行 `pip install torch sentence-transformers`，并把 `datasets sacrebleu rouge-score matplotlib numpy` 装好。之后所有 `encode.py` 命令都加上 `--backend bge`，**跳过所有 `decode_eval.py`**。在 RUNLOG 顶部写明"BGE 备选，无文本解码"。

---

# 阶段 1：数据

## 步骤 1.1：下载并检查

```bash
mkdir -p data/raw ckpt results logs
curl -L -o data/raw/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
python prep_data.py --inspect
```

HF 下载失败的话，先执行 `export HF_ENDPOINT=https://hf-mirror.com`。

**验收**：MSC 的列里有 `dialogue`（话语列表）、`speaker`、`session_id`，以及 `dialoug_id` 或 `dialogue_id`。LoCoMo 共 10 个样本，`conversation` 下有 `session_1` 这类键，每个 turn 都有 `speaker` 和 `text`。字段名不一致的话，只修改 `prep_data.py` 里对应的字段名，写进 RUNLOG。

## 步骤 1.2：预处理

```bash
python prep_data.py
```

**验收**：
- 打印出 train、valid、test 三行统计。test 约 10 段对话、几百个块（估计 500–700 个）；train 的块数 ≥ 10000。实际数字照抄进 RUNLOG。
- 打印出的样例格式是 `Name: sentence`，句子切分看起来合理。

**停止条件**：train 块数 < 5000。

---

# 阶段 2：SONAR 能力测试（关卡）

## 步骤 2.1：编码测试集，并直接解码

```bash
python encode.py --splits test
python decode_eval.py --src ceiling --name ceiling
```

**验收**：生成 `results/dec_ceiling.json` 和 `results/samples_ceiling.txt`。把 json 里的全部数值，以及 samples 里的 **20 对 REF/HYP 原样**抄进 RUNLOG。

**停止条件（必须停下来，等人确认后才能继续）**：完成本步骤后，**无论结果好坏都暂停**，向人汇报 E0 数值和样例，得到"继续"的指令后再进入阶段 3。

---

# 阶段 3：网络就绪

## 步骤 3.1：测试

```bash
python test_common.py      # 必须输出 all ok
bash smoke_test.sh         # 必须输出 SMOKE TEST PASSED（用假数据，数值没有意义）
```

## 步骤 3.2：编码其余数据，标准化

```bash
nohup python encode.py --splits train valid > logs/encode.log 2>&1 &
# 完成后：
python encode.py --normalize
```

**验收**：`data/{train,valid,test}_x.npy` 存在，形状为 [N,16,1024]；`data/mu.npy`、`data/sigma.npy` 存在。

## 步骤 3.3：退化日程和多层数据

```bash
python make_segs.py              # CPU 多进程，默认 16 个，可用 --workers 调整
python make_tree_data.py         # 内存不够的话加 --max_n 15000
```

**验收**：`data/{train,valid,test}_seg_sim_K8.npy` 的形状是 [N,9,16]；`data/train_x_lv{1,2}.npy` 存在。

---

# 阶段 4：训练（只训练 1 个模型）

```bash
nohup python train.py --multilevel --out ckpt/main.pt > logs/main.log 2>&1 &
```

日志每 500 步打印一行：`train … (no-restore …) | val … (no-restore …)`。验证 loss 创新低时会自动保存 checkpoint。

**验收**：日志最后一行 `saved best ckpt … (val X, no-restore Y)` 中 X < Y。把 X、Y 和最优步数写进 RUNLOG。

- 显存不够：`--bs 128`。
- **停止条件**：用 `--lr 1e-4` 重新训练一次之后，仍然 X ≥ Y。此时停下来报告，附上日志最后 20 行。

---

# 阶段 5：测试

## 步骤 5.1：E1 和 E2（向量空间）

```bash
python eval.py --mode none --tag none_sim
python eval.py --ckpt ckpt/main.pt --mode cold --tag main_cold
```

**验收**：生成 `results/e1_none_sim.json` 和 `results/e1_main_cold.json`，后者里有长度为 8 的 `traj`。

## 步骤 5.2：E3 树实验

```bash
python tree_eval.py --tag none
python tree_eval.py --ckpt ckpt/main.pt --tag main
```

**验收**：生成 `results/e3_none.json`、`results/e3_main.json`，以及 `results/restored_tree_main_m{1..4}.pt`。把 json 里的 `n`（测试组数）记进 RUNLOG。

## 步骤 5.3：文本解码

```bash
python decode_eval.py --src results/restored_none_sim.pt     --name none_sim
python decode_eval.py --src results/restored_main_cold.pt    --name main_cold
python decode_eval.py --src results/restored_tree_main_m1.pt --name tree_main_m1
python decode_eval.py --src results/restored_tree_main_m4.pt --name tree_main_m4
```

## 步骤 5.4：汇总

```bash
python plot.py
```

然后在 RUNLOG 末尾写"结果摘要"。**只陈述数据**，回答下面 5 个问题：

1. E0：上限的 BLEU-4、ROUGE-L、数字召回、专名召回各是多少？
2. E1：主方法和不还原相比，Top-1 和 BLEU-4 各差多少？
3. E2：traj 的 Top-1 是否随步数单调上升？
4. E3：主方法下，单层、平衡、链各自的平均 Top-1 是多少？链里 m1 和 m4 差多少？主方法比不还原好多少？
5. 实体：ceiling、none_sim、main_cold 三者的数字召回和专名召回。主方法丢掉的实体里，大约有多少比例在 ceiling 里就已经丢了？

再从 `samples_main_cold.txt` 和 `samples_tree_main_m1.txt` 里各挑 1 个还原得好的例子和 1 个失败的例子，附在后面。

---

# 可选（只在阶段 5 全部完成、并且人明确要求时才做）：固定池化对照

```bash
python make_segs.py --kinds fixed
python eval.py --sched fixed --mode none --tag none_fixed
python decode_eval.py --src results/restored_none_fixed.pt --name none_fixed
python train.py --sched fixed --out ckpt/fixed.pt
python eval.py --sched fixed --ckpt ckpt/fixed.pt --mode cold --tag fixed_cold
python plot.py
```

## 交付清单

- [ ] `results/RUNLOG.md`：每一步的记录 + E0 的 20 条样例 + 结果摘要 + 示例
- [ ] `results/summary.md`、`e2_curve.png`、`e3_depth.png`
- [ ] `results/dec_*.json`、`samples_*.txt`
- [ ] `ckpt/main.pt`、`ckpt/main_log.json`

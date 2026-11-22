"""汇总 7 个算子的多层树实验 → {results}/summary_tree_ops.md, tree_ops_depth.png
例：python summarize_tree_ops.py"""
import argparse, json, os
from ops import OPS, SIDE_INFO

p = argparse.ArgumentParser()
p.add_argument('--results', default='results/ops_tree')
a = p.parse_args()
R = a.results


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


lines = ['## 多层树：各结构 4 个叶子的平均 Top-1（不还原 → 还原，增益为百分点）', '',
         '| 算子 | 单层（压 1 次） | 平衡树（每叶压 2 次） | 不平衡树（原始设计，压 3/3/2/1 次） |', '|---|---|---|---|']
rows = {}
for op in OPS:
    e0, e1 = load(f'{R}/e3_{op}_none.json'), load(f'{R}/e3_{op}_cold.json')
    if not e0 and not e1:
        continue
    rows[op] = (e0, e1)
    cells = []
    for s in ['single', 'balanced', 'chain']:
        n0 = e0[f'{s}_mean_top1'] if e0 else None
        n1 = e1[f'{s}_mean_top1'] if e1 else None
        if n0 is not None and n1 is not None:
            cells.append(f'{n0:.4f} → **{n1:.4f}** ({(n1 - n0) * 100:+.2f})')
        else:
            cells.append(f'{n0 if n0 is not None else "—"} → {n1 if n1 is not None else "—"}')
    lines.append(f'| {SIDE_INFO[op][0]} (`{op}`) | ' + ' | '.join(cells) + ' |')

lines += ['', '## 不平衡树（原始设计：m1,m2→c1；c1,m3→c2；c2,m4→c3）：每个叶子的 Top-1（还原后；括号内为不还原）', '',
          '| 算子 | m1（压 3 次，最旧） | m2（压 3 次） | m3（压 2 次） | m4（压 1 次，最新） |', '|---|---|---|---|---|']
for op, (e0, e1) in rows.items():
    cells = []
    for k in range(1, 5):
        v1 = e1['chain'][f'm{k}'][1] if e1 else '—'
        v0 = e0['chain'][f'm{k}'][1] if e0 else '—'
        cells.append(f'{v1} ({v0})')
    lines.append(f'| `{op}` | ' + ' | '.join(cells) + ' |')

lines += ['', '## 按被压缩次数的平均 Top-1（三种结构合并统计）', '',
          '| 算子 | 模式 | 1 次 | 2 次 | 3 次 |', '|---|---|---|---|---|']
curves = {}
for op, (e0, e1) in rows.items():
    for mode, e in [('不还原', e0), ('还原', e1)]:
        if e:
            d = e['top1_by_times_compressed']
            lines.append(f'| `{op}` | {mode} | ' + ' | '.join(str(d.get(k, '—')) for k in ['1', '2', '3']) + ' |')
            curves[(op, mode)] = [d[k] for k in ['1', '2', '3']]

lines += ['', '## 自检', '', '| 算子 | 模式 | 组数 | 叶子间最小差异（单层/平衡/链） | 一致性误差 |', '|---|---|---|---|---|']
for op, (e0, e1) in rows.items():
    for mode, e in [('none', e0), ('cold', e1)]:
        if e:
            dif = e['leaf_min_pairwise_diff']
            lines.append(f"| `{op}` | {mode} | {e['n_groups']} | {dif['single']} / {dif['balanced']} / {dif['chain']} | "
                         f"{e.get('consistency_err', '—')} |")

dec = []
for op in OPS:
    for tag in ['cold_chain_m1', 'cold_chain_m4', 'cold_balanced_m1']:
        d = load(f'{R}/dec_tree_{op}_{tag}.json')
        if d:
            dec.append((f'`{op}` {tag}', d))
if dec:
    lines += ['', '## 解码后文本（树的叶子）', '',
              '| 设置 | 句数 | BLEU-4 | ROUGE-L | 专名召回 | 换人 |', '|---|---|---|---|---|---|']
    for name, d in dec:
        lines.append(f"| {name} | {d['n_sent']} | {d['BLEU4']} | {d['ROUGE_L']} | {d['name_recall']} | {d.get('speaker_swap', '—')} |")
open(f'{R}/summary_tree_ops.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {}
    for (op, mode), ys in curves.items():
        c = colors.setdefault(op, f'C{len(colors)}')
        ax.plot([1, 2, 3], ys, marker='o', color=c, ls='-' if mode == '还原' else '--',
                label=f'{op} ({"restored" if mode == "还原" else "no restore"})')
    ax.set_xlabel('times compressed'); ax.set_ylabel('Top-1'); ax.set_xticks([1, 2, 3]); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f'{R}/tree_ops_depth.png', dpi=150)
    print('figure saved')
except ImportError:
    print('matplotlib 未安装，跳过作图')

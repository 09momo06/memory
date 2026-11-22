"""汇总表格 + 作图。python plot.py → results/summary.md, results/e2_curve.png, results/e3_depth.png"""
import glob, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
R = sys.argv[1] if len(sys.argv) > 1 else 'results'
rows = [json.load(open(f)) for f in sorted(glob.glob(f'{R}/e1_*.json'))]
lines = ['## E1 单层还原（test 前 n 块；cos / top1）', '',
         '| tag | 退化 | 采样 | project | cos | top1 |', '|---|---|---|---|---|---|']
for r in rows:
    lines.append(f"| {r['tag']} | {r['sched']} | {r['mode']} | {r['project']} | {r['cos']} | {r['top1']} |")

dec = [json.load(open(f)) for f in sorted(glob.glob(f'{R}/dec_*.json'))]
# Auxiliary per-conversation reports also match dec_*.json, but are not rows.
required = {'name', 'n_sent', 'BLEU4', 'ROUGE_L', 'chrF', 'num_recall', 'name_recall'}
dec = [d for d in dec if isinstance(d, dict) and required <= d.keys()]
# Use the ceiling with the same full-block coverage as the compressed runs.
if any(d['name'] == 'ceiling_allblocks' for d in dec):
    dec = [d for d in dec if d['name'] not in {'ceiling', 'ceiling_full'}]
if dec:
    lines += ['', '## 文本解码（E0 上限 / 还原后）', '',
              '| name | 句数 | BLEU-4 | ROUGE-L | chrF | 数字召回 | 专名召回 | 说话人严格正确 | 前缀丢失 | 换人 | 格式差异 | 其他 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for d in dec:
        g = lambda k: d.get(k, '—')
        lines.append(f"| {d['name']} | {d['n_sent']} | {d['BLEU4']} | {d['ROUGE_L']} | {d['chrF']} | {d['num_recall']} | {d['name_recall']} | "
                     f"{g('speaker_acc')} | {g('speaker_missing')} | {g('speaker_swap')} | {g('speaker_format')} | {g('speaker_other')} |")

e3 = [json.load(open(f)) for f in sorted(glob.glob(f'{R}/e3_*.json'))]
if e3:
    lines += ['', '## E3 树实验（每格 cos / top1）', '',
              '| tag | pairing | 结构 | m1 | m2 | m3 | m4 |', '|---|---|---|---|---|---|---|']
    for e in e3:
        for st in ['single', 'balanced', 'chain']:
            if st not in e: continue
            c = e[st]
            lines.append(f"| {e['tag']} | {e['pairing']} | {st} | " + ' | '.join(f'{c[k][0]} / {c[k][1]}' for k in ['m1', 'm2', 'm3', 'm4']) + ' |')
if os.path.exists(f'{R}/e4_merge.json'):
    lines += ['', '## E4 合并相似度诊断', '', '```', open(f'{R}/e4_merge.json').read(), '```']
open(f'{R}/summary.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

# E2：还原步数 vs 质量
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for r in rows:
    if 'traj' in r and len(r['traj']) > 1:
        xs = list(range(0, len(r['traj']) + 1))
        for k, name in enumerate(['cos', 'top1']):
            ax[k].plot(xs, [r['x_K'][k]] + [v[k] for v in r['traj']], marker='o', label=r['tag'])
for k, name in enumerate(['cos', 'top1']):
    ax[k].set_xlabel('restoration steps'); ax[k].set_ylabel(name); ax[k].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{R}/e2_curve.png', dpi=150)

# E3：被压缩次数 vs top1
if e3:
    fig, ax = plt.subplots(figsize=(5, 4))
    for e in e3:
        pts = [(1, e['single'][f'm{k}'][1]) for k in range(1, 5)]
        pts += [(d, e['balanced'][f'm{k + 1}'][1]) for k, d in enumerate(e['balanced_depth'])]
        pts += [(d, e['chain'][f'm{k + 1}'][1]) for k, d in enumerate(e['chain_depth'])]
        agg = {}
        for d, v in pts:
            agg.setdefault(d, []).append(v)
        ds = sorted(agg)
        ax.plot(ds, [sum(agg[d]) / len(agg[d]) for d in ds], marker='o', label=f"{e['tag']} ({e['pairing']})")
    ax.set_xlabel('times compressed'); ax.set_ylabel('top1'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f'{R}/e3_depth.png', dpi=150)
print('figures saved')

"""汇总单层算子对比 → {results}/summary_ops.md, ops_curve.png
例：python summarize_ops.py   （默认读 results/ops，E0 上限读 results/dec_ceiling_allblocks.json）"""
import argparse, json, os
from ops import OPS, SIDE_INFO

p = argparse.ArgumentParser()
p.add_argument('--results', default='results/ops')
p.add_argument('--ceiling', default='results/dec_ceiling_allblocks.json')
p.add_argument('--ckpt_dir', default='ckpt/ops')
a = p.parse_args()
R = a.results


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def ck_info(op):
    log = load(f'{a.ckpt_dir}/{op}_log.json')
    if not log:
        return '—', '—'
    b = min(log, key=lambda r: r['val'])
    return b['step'], f"{(b['val_base'] - b['val']) / b['val_base'] * 100:.2f}%"


lines = ['## 单层算子对比：向量空间（全部 test 块）', '',
         '| 算子 | 附加信息 | 不还原 cos | 不还原 Top-1 | 还原 cos | 还原 Top-1 | Top-1 增益 | 最优步 | 验证 loss 降幅 |',
         '|---|---|---|---|---|---|---|---|---|']
curves = {}
for op in OPS:
    nm, side = SIDE_INFO[op]
    e0, e1 = load(f'{R}/e1_{op}_none.json'), load(f'{R}/e1_{op}_cold.json')
    if not e0 and not e1:
        continue
    f = lambda e, k: f'{e[k]:.4f}' if e else '—'
    gain = f"{(e1['top1'] - e0['top1']) * 100:+.2f}" if e0 and e1 else '—'
    st, red = ck_info(op)
    lines.append(f'| {nm} (`{op}`) | {side} | {f(e0, "cos")} | {f(e0, "top1")} | {f(e1, "cos")} | {f(e1, "top1")} | {gain} | {st} | {red} |')
    if e1 and 'traj' in e1:
        curves[op] = [e1['x_K'][1]] + [v[1] for v in e1['traj']]

dec_rows = []
ceil = load(a.ceiling)
if ceil:
    dec_rows.append(('E0 上限（不压缩）', ceil))
for op in OPS:
    for mode, mname in [('none', '不还原'), ('cold', '还原')]:
        d = load(f'{R}/dec_ops_{op}_{mode}.json')
        if d:
            dec_rows.append((f'{SIDE_INFO[op][0]} · {mname}', d))
if dec_rows:
    lines += ['', '## 单层算子对比：解码后文本', '',
              '| 设置 | 句数 | BLEU-4 | ROUGE-L | chrF | 数字召回 | 专名召回 | 说话人严格正确 | 前缀丢失 | 换人 |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for name, d in dec_rows:
        g = lambda k: d.get(k, '—')
        lines.append(f"| {name} | {d['n_sent']} | {d['BLEU4']} | {d['ROUGE_L']} | {d['chrF']} | {d['num_recall']} | "
                     f"{d['name_recall']} | {g('speaker_acc')} | {g('speaker_missing')} | {g('speaker_swap')} |")
open(f'{R}/summary_ops.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if curves:
        fig, ax = plt.subplots(figsize=(6, 4))
        for op, ys in curves.items():
            ax.plot(range(len(ys)), ys, marker='o', label=op)
        ax.set_xlabel('restoration steps'); ax.set_ylabel('Top-1'); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f'{R}/ops_curve.png', dpi=150)
        print('figure saved')
except ImportError:
    print('matplotlib 未安装，跳过作图')

"""单层算子对比的评测（向量空间：cos / Top-1，同主实验的 Evaluator，在全部 test 块上）。
例：python eval_ops.py --op dct --mode none                         # 不还原：直接用 P_8 x
    python eval_ops.py --op dct --mode cold --ckpt ckpt/ops/dct.pt  # 还原（同时输出 8 步轨迹）
输出 {results}/e1_{op}_{mode}.json，{results}/restored_{op}_{mode}.pt（可直接给 decode_eval.py 解码）"""
import argparse, json, os
import numpy as np
import torch
from common import DEV, Evaluator
from ops import OPS
from ops_model import degrade, load_model_p, restore_p

p = argparse.ArgumentParser()
p.add_argument('--data', default='data')
p.add_argument('--op', required=True, choices=OPS)
p.add_argument('--mode', default='cold', choices=['none', 'cold'])
p.add_argument('--ckpt')
p.add_argument('--results', default='results/ops')
a = p.parse_args()
os.makedirs(a.results, exist_ok=True)

ev = Evaluator('test', a.data)
X = ev.X
n = len(X)
idx = torch.arange(n, device=DEV)
P = torch.tensor(np.load(f'{a.data}/ops/test_P_{a.op}.npy')).to(DEV)
x_K = degrade(X, P[:, -1])
res = {'op': a.op, 'mode': a.mode, 'n_blocks': n, 'x_K': ev.score(x_K, idx)}
if a.mode == 'none':
    xh = x_K
else:
    model = load_model_p(a.ckpt)
    outs, trajs = [], []
    for i in range(0, n, 256):
        xo, tr = restore_p(model, x_K[i:i + 256], P[i:i + 256], return_traj=True)
        outs.append(xo)
        trajs.append(tr)
    xh = torch.cat(outs)
    res['traj'] = [ev.score(torch.cat([tr[j] for tr in trajs]), idx) for j in range(len(trajs[0]))]
    res['consistency_err'] = float((degrade(xh, P[:, -1]) - x_K).abs().max())   # 应接近 0
res['cos'], res['top1'] = ev.score(xh, idx)
print(json.dumps(res))
json.dump(res, open(f'{a.results}/e1_{a.op}_{a.mode}.json', 'w'))
torch.save({'x': xh.half().cpu(), 'blocks': list(range(n))}, f'{a.results}/restored_{a.op}_{a.mode}.pt')

"""E1 单层还原 + E2 还原步数曲线。
例：python eval.py --mode none --tag none_sim                     # 不还原
    python eval.py --ckpt ckpt/main.pt --mode cold --tag main_cold  # 主方法（同时输出 E2 轨迹）
输出 results/e1_{tag}.json，results/restored_{tag}.pt（dict: x 标准化空间 fp16, blocks）"""
import argparse, json
import numpy as np
import torch
from common import DEV, Evaluator, expand, load_model, restore

p = argparse.ArgumentParser()
p.add_argument('--data', default='data')
p.add_argument('--sched', default='sim')
p.add_argument('--K', type=int, default=8)
p.add_argument('--ckpt')
p.add_argument('--mode', default='cold', choices=['none', 'cold', 'naive', 'oneshot'])
p.add_argument('--project', action='store_true')
p.add_argument('--n', type=int, default=100000, help='最多评测多少个 test 块（默认全部）')
p.add_argument('--tag', required=True)
p.add_argument('--results', default='results')
a = p.parse_args()

ev = Evaluator('test', a.data)
n = min(a.n, len(ev.X))
X = ev.X[:n]
idx = torch.arange(n, device=DEV)
S = torch.tensor(np.load(f'{a.data}/test_seg_{a.sched}_K{a.K}.npy')[:n].astype(np.int64), device=DEV)
x_K = expand(X, S[:, -1])

res = {'tag': a.tag, 'sched': a.sched, 'K': a.K, 'mode': a.mode, 'project': a.project, 'n': n}
if a.mode == 'none':
    xh = x_K
    res['cos'], res['top1'] = ev.score(xh, idx)
else:
    model = load_model(a.ckpt)
    assert model.K == a.K
    outs, trajs = [], []
    for i in range(0, n, 256):
        xo, tr = restore(model, x_K[i:i + 256], S[i:i + 256], 0, a.mode, a.project, return_traj=True)
        outs.append(xo)
        trajs.append(tr)
    xh = torch.cat(outs)
    res['cos'], res['top1'] = ev.score(xh, idx)
    res['traj'] = [ev.score(torch.cat([tr[j] for tr in trajs]), idx) for j in range(len(trajs[0]))]
res['x_K'] = ev.score(x_K, idx)
print(json.dumps(res))
json.dump(res, open(f'{a.results}/e1_{a.tag}.json', 'w'))
torch.save({'x': xh.half().cpu(), 'blocks': list(range(n))}, f'{a.results}/restored_{a.tag}.pt')

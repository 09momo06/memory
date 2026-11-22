"""多层树评测（通用投影算子）。测试组的选取和主实验 tree_eval.py 完全相同（同一批 208 组）。
例：python eval_tree_ops.py --op dct --mode none                                  # 不还原：每层直接展开
    python eval_tree_ops.py --op dct --mode cold --ckpt ckpt/ops_tree/dct.pt      # 逐层 cold 还原
输出 {results}/e3_{op}_{mode}.json：每种结构每个叶子的 (cos, top1)，以及按压缩次数的平均
     {results}/restored_{op}_{mode}_{结构}_m{k}.pt：还原出的叶子，可直接给 decode_eval.py 解码"""
import argparse, json, os
import numpy as np
import torch
from common import DEV, Evaluator
from ops import OPS
from ops_model import load_model_p, restore_p
from tree_ops import DEPTH, run_trees, unpack

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--op', required=True, choices=OPS)
    p.add_argument('--mode', default='cold', choices=['none', 'cold'])
    p.add_argument('--ckpt')
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--results', default='results/ops_tree')
    a = p.parse_args()
    os.makedirs(a.results, exist_ok=True)

    ev = Evaluator('test', a.data)
    meta = np.load(f'{a.data}/test_meta.npy')
    rng = np.random.RandomState(0)                                   # 与 tree_eval.py 相同的选组方式
    st = [i for i in range(len(meta) - 3) if meta[i + 3][0] == meta[i][0] and meta[i][1] % 4 == 0]
    st = np.array(st)[rng.permutation(len(st))[:a.n]]
    G = torch.tensor(np.stack([st, st + 1, st + 2, st + 3], 1), device=DEV)
    m = ev.X[G]

    model = load_model_p(a.ckpt) if a.mode == 'cold' else None
    if model is not None:
        assert model.levels >= 3, '树实验需要多层模型（train_ops.py --multilevel）'
    cons = [0.0]

    def rest(z, B, P, lv):
        xk = unpack(z, B)
        if model is None:
            return xk
        xh = restore_p(model, xk, P, lv=lv)
        cons[0] = max(cons[0], float((torch.einsum('nlk,nld->nkd', B, xh) - z).abs().max()))
        return xh

    out = run_trees(a.op, m, rest)
    res = {'op': a.op, 'mode': a.mode, 'ckpt': a.ckpt, 'n_groups': len(G)}
    for s, leaves in out.items():
        res[s] = {f'm{k + 1}': ev.score(leaves[k], G[:, k]) for k in range(4)}
        res[f'{s}_mean_top1'] = round(float(np.mean([res[s][f'm{k + 1}'][1] for k in range(4)])), 4)
        for k in range(4):
            torch.save({'x': leaves[k].half().cpu(), 'blocks': G[:, k].tolist()},
                       f'{a.results}/restored_{a.op}_{a.mode}_{s}_m{k + 1}.pt')
    # 按"被压缩次数"汇总（三种结构合在一起）
    by = {}
    for s, ds in DEPTH.items():
        for k, dpt in enumerate(ds):
            by.setdefault(dpt, []).append(res[s][f'm{k + 1}'][1])
    res['top1_by_times_compressed'] = {str(k): round(float(np.mean(v)), 4) for k, v in sorted(by.items())}
    res['depth'] = DEPTH
    # 自检：不同叶子的还原结果不应完全相同（主实验平衡树 m1/m4 数字完全一致的疑点）
    res['leaf_min_pairwise_diff'] = {s: round(min(float((lv[i] - lv[j]).abs().max())
                                                  for i in range(4) for j in range(i + 1, 4)), 6)
                                     for s, lv in out.items()}
    if model is not None:
        res['consistency_err'] = cons[0]
    print(json.dumps(res))
    json.dump(res, open(f'{a.results}/e3_{a.op}_{a.mode}.json', 'w'))

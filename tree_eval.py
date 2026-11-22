"""E3 树实验：4 个记忆块 m1..m4 → 链 / 平衡树 → 只存根（16 个向量，4 倍压缩）→ 逐层还原到叶子。
例：python tree_eval.py --tag none                         # 不还原，直接展开
    python tree_eval.py --ckpt ckpt/main.pt --tag main
--pairing same：4 块来自同一段对话的连续 64 句（默认）；random：4 块来自 4 段不同对话（可选，跨话题合并）。
同时保存链结构的还原叶子 results/restored_tree_{tag}_m{1..4}.pt，可用 decode_eval.py 解码。"""
import argparse, json
import numpy as np
import torch
from common import DEV, Evaluator, compress, load_model, restore, schedules, unpack

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--ckpt')
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--pairing', default='same', choices=['same', 'random'])
    p.add_argument('--tag', required=True)
    p.add_argument('--results', default='results')
    a = p.parse_args()
    K, r = 8, 1
    ev = Evaluator('test', a.data)
    meta = np.load(f'{a.data}/test_meta.npy')
    rng = np.random.RandomState(0)
    if a.pairing == 'same':
        st = [i for i in range(len(meta) - 3) if meta[i + 3][0] == meta[i][0] and meta[i][1] % 4 == 0]
        st = np.array(st)[rng.permutation(len(st))[:a.n]]
        G = np.stack([st, st + 1, st + 2, st + 3], 1)
    else:
        first = {}
        for i, (art, _) in enumerate(meta):
            first.setdefault(art, i)
        arts = np.array(list(first.values()))
        G = np.stack([arts[rng.permutation(len(arts))[:4]] for _ in range(a.n)])
    G = torch.tensor(G, device=DEV)
    model = load_model(a.ckpt) if a.ckpt else None

    def comp(x):
        S = torch.tensor(schedules(x.cpu().numpy(), 'sim', K, r).astype(np.int64), device=DEV)
        return compress(x, S), S

    def rest(z, S, lv):
        xk = unpack(z, S)
        return restore(model, xk, S, lv, 'cold') if model is not None else xk

    m = ev.X[G]                                                   # [n,4,16,d]
    (z1, S1), (z2, S2), (z3, S3), (z4, S4) = [comp(m[:, k]) for k in range(4)]
    out = {'tag': a.tag, 'ckpt': a.ckpt, 'pairing': a.pairing, 'n': len(G)}

    # 链：c1=[z1;z2], c2=[z(c1);z3], c3=[z(c2);z4]，叶子被压缩次数 3,3,2,1
    zc1, Sc1 = comp(torch.cat([z1, z2], 1))
    zc2, Sc2 = comp(torch.cat([zc1, z3], 1))
    c3 = torch.cat([zc2, z4], 1)
    c2h = rest(c3[:, :8], Sc2, 2)
    c1h = rest(c2h[:, :8], Sc1, 1)
    leaves = [rest(c1h[:, :8], S1, 0), rest(c1h[:, 8:], S2, 0), rest(c2h[:, 8:], S3, 0), rest(c3[:, 8:], S4, 0)]
    out['chain'] = {f'm{k + 1}': ev.score(leaves[k], G[:, k]) for k in range(4)}
    for k in range(4):
        torch.save({'x': leaves[k].half().cpu(), 'blocks': G[:, k].tolist()},
                   f'{a.results}/restored_tree_{a.tag}_m{k + 1}.pt')
    out['chain_depth'] = [3, 3, 2, 1]

    # 平衡：c1=[z1;z2], c1'=[z3;z4], root=[z(c1);z(c1')]，每个叶子被压缩 2 次
    zc1b, Sc1b = comp(torch.cat([z3, z4], 1))
    c1h = rest(zc1, Sc1, 1)
    c1bh = rest(zc1b, Sc1b, 1)
    leaves = [rest(c1h[:, :8], S1, 0), rest(c1h[:, 8:], S2, 0), rest(c1bh[:, :8], S3, 0), rest(c1bh[:, 8:], S4, 0)]
    out['balanced'] = {f'm{k + 1}': ev.score(leaves[k], G[:, k]) for k in range(4)}
    out['balanced_depth'] = [2, 2, 2, 2]

    # 单层参照：每个叶子只压缩 1 次
    leaves = [rest(z, S, 0) for z, S in [(z1, S1), (z2, S2), (z3, S3), (z4, S4)]]
    out['single'] = {f'm{k + 1}': ev.score(leaves[k], G[:, k]) for k in range(4)}
    print(json.dumps(out, ensure_ascii=False))
    json.dump(out, open(f'{a.results}/e3_{a.tag}.json', 'w'), ensure_ascii=False)

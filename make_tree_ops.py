"""某个算子的多层训练数据（同主实验 make_tree_data.py 的构造，只是换成该算子的压缩）。
lv1：c1 = [z(m_i); z(m_{i+1})]；lv2：c2 = [z(c1); z(m_{i+2})]（链结构的第 2 层）。
每个上层节点也要算它自己的投影 P（训练还原网络时用）。
输出 {data}/ops/train_x_lv{1,2}_{op}.npy [n,16,1024] fp16 和 train_P_lv{1,2}_{op}.npy [n,9,16,16]
例：python make_tree_ops.py --op dct"""
import argparse, time
import numpy as np
import torch
from ops import OPS
from tree_ops import comp

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--op', required=True, choices=OPS)
    p.add_argument('--max_n', type=int, default=15000)
    a = p.parse_args()
    t0 = time.time()
    X = torch.tensor(np.load(f'{a.data}/train_x.npy').astype(np.float32))
    meta = np.load(f'{a.data}/train_meta.npy')
    z = torch.cat([comp(a.op, X[i:i + 2048])[0] for i in range(0, len(X), 2048)])
    rng = np.random.RandomState(0)
    starts = np.array([i for i in range(len(X) - 2) if meta[i + 2][0] == meta[i][0]])
    starts = starts[rng.permutation(len(starts))[:a.max_n]]
    c1 = torch.cat([z[starts], z[starts + 1]], 1)
    zc1, _, P1 = comp(a.op, c1)
    c2 = torch.cat([zc1, z[starts + 2]], 1)
    _, _, P2 = comp(a.op, c2)
    for lv, D, P in [(1, c1, P1), (2, c2, P2)]:
        np.save(f'{a.data}/ops/train_x_lv{lv}_{a.op}.npy', D.numpy().astype(np.float16))
        np.save(f'{a.data}/ops/train_P_lv{lv}_{a.op}.npy', P.numpy().astype(np.float32))
        print(a.op, 'lv', lv, tuple(D.shape), f'{time.time() - t0:.0f}s', flush=True)

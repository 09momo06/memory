"""多层训练数据（只用同一篇文章里的连续块）。
lv1：c1 = [z(m_i); z(m_{i+1})]          —— 第 1 层节点
lv2：c2 = [z(c1_i); z(m_{i+2})]          —— 链结构第 2 层节点
输出 {data}/train_x_lv{1,2}.npy, train_seg_lv{1,2}.npy"""
import argparse
import numpy as np
import torch
from common import compress, schedules

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--max_n', type=int, default=30000)
    p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    K, r = 8, 1
    X = torch.tensor(np.load(f'{a.data}/train_x.npy').astype(np.float32))
    S = torch.tensor(np.load(f'{a.data}/train_seg_sim_K8.npy').astype(np.int64))
    meta = np.load(f'{a.data}/train_meta.npy')
    z = torch.cat([compress(X[i:i + 4096], S[i:i + 4096]) for i in range(0, len(X), 4096)])

    rng = np.random.RandomState(0)
    starts = np.array([i for i in range(len(X) - 2) if meta[i + 2][0] == meta[i][0]])
    starts = starts[rng.permutation(len(starts))[:a.max_n]]

    c1 = torch.cat([z[starts], z[starts + 1]], 1)                       # [n,16,d]
    S1 = schedules(c1.numpy(), 'sim', K, r, a.workers)
    zc1 = compress(c1, torch.tensor(S1.astype(np.int64)))
    c2 = torch.cat([zc1, z[starts + 2]], 1)
    S2 = schedules(c2.numpy(), 'sim', K, r, a.workers)
    for lv, D, Sg in [(1, c1, S1), (2, c2, S2)]:
        np.save(f'{a.data}/train_x_lv{lv}.npy', D.numpy().astype(np.float16))
        np.save(f'{a.data}/train_seg_lv{lv}.npy', Sg)
        print('lv', lv, D.shape)

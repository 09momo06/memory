"""预计算退化日程。输出 {data}/{split}_seg_{kind}_K8.npy  [N,9,16] int8
默认只算相似合并（sim）；有时间做固定池化对照时加 --kinds sim fixed"""
import argparse
import numpy as np
from common import schedules

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--kinds', nargs='+', default=['sim'])
    p.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    for name in a.splits:
        X = np.load(f'{a.data}/{name}_x.npy')
        for kind in a.kinds:
            S = schedules(X, kind, 8, 1, a.workers)
            np.save(f'{a.data}/{name}_seg_{kind}_K8.npy', S)
            print(name, kind, S.shape, flush=True)

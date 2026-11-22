"""E4 诊断：被合并的相邻对有多相似？会不会经常被迫把不相关的句子合到一起（Gemini 第 3 条担心的情况）？
统计每次合并时两段均值的余弦相似度（标准化空间，即算法实际使用的空间），
并（可选）看"块内最差一次合并的相似度"与该块还原质量的相关性。
例：python analyze_merge.py --restored results/restored_sim_K8_cold.pt"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--data', default='data')
p.add_argument('--n', type=int, default=2000)
p.add_argument('--restored')
p.add_argument('--results', default='results')
a = p.parse_args()
X = np.load(f'{a.data}/test_x.npy')[:a.n].astype(np.float32)


def merge_sims(x, S):
    out = []
    for t in range(1, len(S)):
        prev, cur = S[t - 1], S[t]
        for j in np.unique(cur):
            parts = np.unique(prev[cur == j])
            if len(parts) == 2:
                u, v = x[prev == parts[0]].mean(0), x[prev == parts[1]].mean(0)
                out.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8)))
    return out


res = {}
per_block_min = None
import os
for kind in ['sim', 'fixed']:
    if not os.path.exists(f'{a.data}/test_seg_{kind}_K8.npy'):
        continue
    S = np.load(f'{a.data}/test_seg_{kind}_K8.npy')[:a.n]
    sims = [merge_sims(x, s) for x, s in zip(X, S)]
    allv = np.concatenate(sims)
    res[kind] = {'quantiles(10,25,50,75,90)': np.percentile(allv, [10, 25, 50, 75, 90]).round(3).tolist(),
                 **{f'frac<{tau}': round(float((allv < tau).mean()), 4) for tau in [0.0, 0.2, 0.4]}}
    if kind == 'sim':
        per_block_min = np.array([min(s) for s in sims])
    # 参照：随机两句（跨文章）的相似度分布，代表"毫不相关"的水平
rand = np.random.RandomState(0)
A = X.reshape(-1, X.shape[-1])
i, j = rand.randint(0, len(A), 5000), rand.randint(0, len(A), 5000)
rs = (A[i] * A[j]).sum(1) / (np.linalg.norm(A[i], axis=1) * np.linalg.norm(A[j], axis=1))
res['random_pair_quantiles(10,50,90)'] = np.percentile(rs, [10, 50, 90]).round(3).tolist()
if a.restored:
    xh = torch.load(a.restored)['x'].float()[:a.n]
    mu, sg = torch.tensor(np.load(f'{a.data}/mu.npy')), torch.tensor(np.load(f'{a.data}/sigma.npy'))
    cos = F.cosine_similarity(xh * sg + mu, torch.tensor(X) * sg + mu, dim=-1).mean(1).numpy()
    res['corr(block_min_merge_sim, block_restore_cos)'] = round(float(np.corrcoef(per_block_min, cos)[0, 1]), 4)
print(json.dumps(res, indent=1, ensure_ascii=False))
json.dump(res, open(f'{a.results}/e4_merge.json', 'w'), indent=1, ensure_ascii=False)

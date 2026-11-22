"""算子对比的单元测试：python test_ops.py，必须输出 all ops ok。"""
import numpy as np
import torch
from ops import OPS, FUNCS, L, K
from ops_model import RestorerP, restore_p, degrade

rng = np.random.RandomState(0)
# 带一点结构的测试块：几段话题 + 噪声，并制造一对几乎相同的句子
base = rng.randn(4, 64)
x = np.repeat(base, 4, axis=0) + 0.5 * rng.randn(L, 64)
x[9] = x[3] + 0.01
x = x.astype(np.float32)

errs = {}
for op in OPS:
    P = FUNCS[op](x)
    assert P.shape == (K + 1, L, L), op
    assert np.allclose(P[0], np.eye(L), atol=1e-6), f'{op}: P_0 应为单位阵'
    for t in range(K + 1):
        assert np.allclose(P[t], P[t].T, atol=1e-6), f'{op}: P_{t} 应对称'
        assert np.allclose(P[t] @ P[t], P[t], atol=1e-5), f'{op}: P_{t} 应幂等'
        assert np.linalg.matrix_rank(P[t], tol=1e-4) == L - t, f'{op}: P_{t} 的秩应为 {L - t}'
        assert np.allclose(P[K] @ P[t], P[K], atol=1e-5), f'{op}: 应嵌套 P_K P_t = P_K'
    errs[op] = float(np.linalg.norm(x - P[K] @ x))

# SVD 是秩 8 投影里重建误差最小的（Eckart–Young），可以作为所有算子"不还原"误差的下界
assert all(errs['svd'] <= e + 1e-4 for e in errs.values()), errs
# 相似合并应该先合并几乎相同的那对句子（非相邻版本能把 3 和 9 合在一起）
Pg = FUNCS['merge_global'](x)
assert Pg[1][3, 9] > 0, 'merge_global 第一步应合并句子 3 和 9'
# 选择算子：第一步应删掉 3 或 9 中的一句（另一句几乎可以代表它）
Ps = FUNCS['select_fl'](x)
assert Ps[1][3, 3] == 0 or Ps[1][9, 9] == 0

# cold 采样：完美模型 → 精确还原；任意模型 → 严格保持 P_K x
X = torch.tensor(np.stack([x, x[::-1].copy()]))
for op in OPS:
    P = torch.tensor(np.stack([FUNCS[op](xi) for xi in X.numpy()]), dtype=torch.float32)
    xK = degrade(X, P[:, K])

    class Oracle(torch.nn.Module):
        def forward(self, xt, t, Pt, lv=None): return X

    class Rand(torch.nn.Module):
        def forward(self, xt, t, Pt, lv=None): return torch.randn_like(xt)

    assert torch.allclose(restore_p(Oracle(), xK, P), X, atol=1e-4), f'{op}: 完美模型应精确还原'
    out = restore_p(Rand(), xK, P)
    assert torch.allclose(degrade(out, P[:, K]), xK, atol=1e-3), f'{op}: cold 采样应保持压缩结果'

m = RestorerP(K=K, d_in=64, d=32, layers=1)
o = m(xK, torch.tensor([K, K]), P[:, K])
assert o.shape == xK.shape and torch.allclose(o, xK), '初始化时输出应等于输入'
print('no-restore error (Frobenius):', {k: round(v, 2) for k, v in errs.items()})
print('all ops ok')

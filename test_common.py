"""单元测试：python test_common.py，必须全部通过才能往下做。"""
import numpy as np
import torch
from common import *

rng = np.random.RandomState(0)
x = rng.randn(16, 64).astype(np.float32)

# 1. 日程的结构性质
for S in [sim_schedule(x, 8, 1), fixed_schedule(16, 8, 1), sim_schedule(x, 4, 2), fixed_schedule(16, 1, 8)]:
    assert S[0].tolist() == list(range(16))
    assert S[-1].max() + 1 == 8, '最后应剩 8 段'
    assert all((np.diff(s) >= 0).all() for s in S), '必须保持顺序'
    assert all(set(s.tolist()) == set(range(s.max() + 1)) for s in S), '段号连续'
    for a, b in zip(S[:-1], S[1:]):
        assert all(len(set(b[a == j])) == 1 for j in np.unique(a)), '只合并不拆分'
        assert b.max() + 1 == a.max() + 1 - (8 // (len(S) - 1)), '每步合并 r 对'

# 1b. 一步压一半 + 只合并相邻 → 唯一可行配对就是固定池化，贪心会凑不满，应报错
try:
    sim_schedule(x, 1, 8); raise SystemExit('应当报错')
except AssertionError:
    pass

# 2. 相似合并第一步合并的确实是最相似的相邻对
y = rng.randn(16, 64).astype(np.float32)
y[5] = y[4] + 0.01
assert sim_schedule(y, 1, 1)[1][4] == sim_schedule(y, 1, 1)[1][5]

# 3. expand / compress / unpack
X = torch.tensor(np.stack([x, y]))
S = torch.tensor(schedules(np.stack([x, y]), 'sim', 8, 1)).long()
e = expand(X, S[:, -1])
seg0 = S[0, -1]
assert torch.allclose(e[0, 0], X[0, seg0 == seg0[0]].mean(0), atol=1e-5)
z = compress(X, S)
assert z.shape == (2, 8, 64)
assert torch.allclose(unpack(z, S), e, atol=1e-6)

# 4. seg_info
size, within = seg_info(torch.tensor([[0, 0, 1, 2, 2, 2]]))
assert size.tolist() == [[2, 2, 1, 3, 3, 3]] and within.tolist() == [[0, 1, 0, 0, 1, 2]]

# 5. 完美模型 + cold 采样 → 精确还原
class Oracle(torch.nn.Module):
    def __init__(self, x0): super().__init__(); self.x0 = x0
    def forward(self, xt, t, lv, seg): return self.x0
out = restore(Oracle(X), e, S, mode='cold')
assert torch.allclose(out, X, atol=1e-5), 'cold + 完美模型应精确还原'

# 6. 任意模型 + cold 采样 → 输出的段均值严格等于存储的 z（一致性，不需要额外锚点）
class Rand(torch.nn.Module):
    def forward(self, xt, t, lv, seg): return torch.randn_like(xt)
out = restore(Rand(), e, S, mode='cold')
assert torch.allclose(compress(out, S), z, atol=1e-4), 'cold 采样应保持段均值'
out = restore(Rand(), e, S, mode='naive', project=True)
assert torch.allclose(compress(out, S), z, atol=1e-4), 'project 应保持段均值'

# 7. 网络前向形状
m = Restorer(K=8, d_in=64, d=32, layers=1)
o = m(e, torch.tensor([8, 8]), torch.tensor([0, 0]), S[:, 8])
assert o.shape == e.shape and torch.allclose(o, e), '初始化时输出应等于输入'
print('all ok')

"""多层树（通用投影算子）的单元测试：python test_tree_ops.py，必须输出 all tree ops ok。"""
import numpy as np
import torch
from ops import OPS, FUNCS, basis
from ops_model import RestorerP, restore_p
from tree_ops import comp, unpack, run_trees

rng = np.random.RandomState(0)
n, d = 3, 64
m = torch.tensor(rng.randn(n, 4, 16, d).astype(np.float32))
m[:, :, 5] = m[:, :, 4] + 0.01                                   # 制造相似句

for op in OPS:
    # 1. 压缩 / 展开一致：B z = P_8 x
    x = m[:, 0]
    z, B, P = comp(op, x)
    assert z.shape == (n, 8, d) and B.shape == (n, 16, 8)
    assert torch.allclose(unpack(z, B), torch.bmm(P[:, -1], x), atol=1e-4), op
    assert torch.allclose(B.transpose(1, 2) @ B, torch.eye(8).expand(n, 8, 8), atol=1e-5), op

    # 2. 完美还原器（查表）→ 所有结构的所有叶子都精确还原
    reg = []
    def hook(x, z, B): reg.append((unpack(z, B), x))
    def oracle(z, B, P, lv):
        xk = unpack(z, B)
        for k, v in reg:
            if k.shape == xk.shape and torch.allclose(k, xk, atol=1e-4):
                return v
        raise AssertionError('oracle: 找不到对应的节点（树的结构或拼接顺序有误）')
    out = run_trees(op, m, oracle, hook)
    for st, leaves in out.items():
        for k in range(4):
            assert torch.allclose(leaves[k], m[:, k], atol=1e-4), (op, st, k)

    # 3. 不还原（直接展开）：每个叶子误差按压缩次数增大，且单层误差 = ||(I − P_8) x||
    none = run_trees(op, m, lambda z, B, P, lv: unpack(z, B))
    for k in range(4):
        e1 = (none['single'][k] - m[:, k]).norm()
        ec = (none['chain'][k] - m[:, k]).norm()
        assert ec >= e1 - 1e-3, (op, k)

    # 4. 任意网络 + cold 采样：每次还原都严格保持该节点存储的 z
    model = RestorerP(K=8, d_in=d, d=32, layers=1, levels=3)
    with torch.no_grad():
        model.out.weight.normal_(0, 0.5)                             # 让网络输出明显偏离输入
    def rest(z, B, P, lv):
        xh = restore_p(model, unpack(z, B), P, lv=lv)
        assert torch.allclose(torch.einsum('nlk,nld->nkd', B, xh), z, atol=1e-3), (op, lv)
        return xh
    run_trees(op, m, rest)

# 5. 单层网络（levels=0）仍然能不带 lv 调用，和旧 checkpoint 兼容
m0 = RestorerP(K=8, d_in=d, d=32, layers=1)
assert not hasattr(m0, 'lv_emb')
restore_p(m0, unpack(z, B), P)
print('all tree ops ok')

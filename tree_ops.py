"""通用投影算子的多层树：压缩、展开、建树、逐层还原（make_tree_ops / eval_tree_ops / test_tree_ops 共用）。

节点：任意一个 16×d 的块（叶子 = 16 句；上层节点 = 两个子节点压缩结果的拼接）。
压缩一个节点：P = compute_P(op, x)，B = basis(op, x, P_8)，z = Bᵀx（8×d）。P 和 B 作为该节点的附加信息存储。
展开：B z = P_8 x，然后用 cold 采样还原，所还原节点的层级 lv 作为网络条件。

和主实验 tree_eval.py 结构一致：
  链：c1=[z1;z2]，c2=[z(c1);z3]，c3=[z(c2);z4]，只存 c3。叶子被压缩次数 3,3,2,1
      还原：c2 ← z(c2)（lv=2），c1 ← z(c1)（lv=1），叶子（lv=0）
  平衡：c1=[z1;z2]，c1'=[z3;z4]，root=[z(c1);z(c1')]。每个叶子被压缩 2 次
      还原：c1、c1' （lv=1），叶子（lv=0）
  单层：每个叶子单独压缩 1 次再还原（参照）"""
import numpy as np
import torch
from ops import basis, compute_P


def comp(op, x):
    """x: torch [n,16,d] → z [n,8,d], B [n,16,8], P [n,9,16,16]（与 x 同设备，float32）"""
    xn = x.detach().float().cpu().numpy()
    P = compute_P(op, xn, workers=1)
    B = np.stack([basis(op, xi, Pi[-1]) for xi, Pi in zip(xn, P)])
    P = torch.tensor(P, device=x.device)
    B = torch.tensor(B, device=x.device)
    z = torch.einsum('nlk,nld->nkd', B, x.float())
    return z, B, P


def unpack(z, B):
    return torch.einsum('nlk,nkd->nld', B, z)


def run_trees(op, m, rest, hook=None):
    """m: [n,4,16,d] 四个叶子。rest(z, B, P, lv) → 还原的 16×d 节点。
    hook(x, z, B)：每压缩一个节点调用一次（测试用）。
    返回 {'single'|'balanced'|'chain': [m1,m2,m3,m4 的还原结果]}"""
    def c(x):
        z, B, P = comp(op, x)
        if hook:
            hook(x, z, B)
        return z, B, P

    leaf = [c(m[:, k]) for k in range(4)]
    (z1, B1, P1), (z2, B2, P2), (z3, B3, P3), (z4, B4, P4) = leaf
    out = {'single': [rest(z, B, P, 0) for z, B, P in leaf]}

    # 平衡
    zc1, Bc1, Pc1 = c(torch.cat([z1, z2], 1))
    zc1b, Bc1b, Pc1b = c(torch.cat([z3, z4], 1))
    c1h = rest(zc1, Bc1, Pc1, 1)
    c1bh = rest(zc1b, Bc1b, Pc1b, 1)
    out['balanced'] = [rest(c1h[:, :8], B1, P1, 0), rest(c1h[:, 8:], B2, P2, 0),
                       rest(c1bh[:, :8], B3, P3, 0), rest(c1bh[:, 8:], B4, P4, 0)]

    # 链（c1 与平衡树的 c1 相同）
    zc2, Bc2, Pc2 = c(torch.cat([zc1, z3], 1))
    c3 = torch.cat([zc2, z4], 1)                                     # 只存这个（+ 各节点附加信息）
    c2h = rest(c3[:, :8], Bc2, Pc2, 2)
    c1h = rest(c2h[:, :8], Bc1, Pc1, 1)
    out['chain'] = [rest(c1h[:, :8], B1, P1, 0), rest(c1h[:, 8:], B2, P2, 0),
                    rest(c2h[:, 8:], B3, P3, 0), rest(c3[:, 8:], B4, P4, 0)]
    return out


DEPTH = {'single': [1, 1, 1, 1], 'balanced': [2, 2, 2, 2], 'chain': [3, 3, 2, 1]}

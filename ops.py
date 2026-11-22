"""单层算子对比：把各种冷扩散退化算子统一写成"嵌套的正交投影"。

每个算子对一个记忆块 x ∈ R^{16×1024} 给出 K+1=9 个 16×16 矩阵 P_0 … P_8：
  - P_0 = I（未退化），P_t 的秩 = 16 − t，所以 P_8 的秩是 8，相当于 8 个向量（和主实验一样压到一半）
  - 退化 D(x, t) = P_t x，只在"句子位置"这个维度上操作，每句的 1024 维特征不被打乱
  - P_t 对称且幂等（正交投影），并且嵌套：P_8 P_t = P_8
    → cold 采样 x ← x − P_t x̂0 + P_{t−1} x̂0 对任意网络都严格保持 P_8 x = 存储的压缩结果
存储的压缩结果都是 8×1024（P_8 的列空间里的 8 个坐标向量），外加少量附加信息（见 SIDE_INFO）。
本文件只依赖 numpy，方便多进程预计算。"""
import numpy as np
from common import sim_schedule, fixed_schedule

L, K = 16, 8

# 存储开销：压缩结果本身都是 8×1024×fp16 = 16384 字节，下面是额外要存的东西
SIDE_INFO = {
    'merge_adj':    ('相邻相似合并（主方法）', '合并日程，约 8 字节'),
    'merge_fixed':  ('固定池化 (1,2)(3,4)…', '无'),
    'merge_global': ('非相邻相似合并（层次聚类）', '合并日程，约 16 字节'),
    'dct':          ('序列低通 / DCT 模糊', '无'),
    'random':       ('随机正交投影（压缩感知）', '无（固定随机种子）'),
    'select_fl':    ('句子选择，删一半（facility location）', '保留位置，2 字节'),
    'svd':          ('块内低秩 / SVD（线性压缩最优参考）', '左奇异向量 16×8，约 256 字节（1.6%）'),
}
OPS = list(SIDE_INFO)


def groups_to_P(seg):
    """分组 → 组内平均矩阵（对称幂等）。seg: [L] 组号，不要求连续或相邻。"""
    same = (seg[:, None] == seg[None, :]).astype(np.float64)
    return same / same.sum(1, keepdims=True)


def _unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def op_merge_adj(x):
    return np.stack([groups_to_P(s) for s in sim_schedule(x, K, 1)])


def op_merge_fixed(x):
    return np.stack([groups_to_P(s) for s in fixed_schedule(L, K, 1)])


def op_merge_global(x):
    """每步合并全局最相似的两组（组均值的余弦相似度），不要求相邻。"""
    seg = np.arange(L)
    Ps = [groups_to_P(seg)]
    for _ in range(K):
        ids = np.unique(seg)
        means = _unit(np.stack([x[seg == g].mean(0) for g in ids]))
        sim = means @ means.T
        np.fill_diagonal(sim, -np.inf)
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        seg = np.where(seg == ids[j], ids[i], seg)
        Ps.append(groups_to_P(seg))
    return np.stack(Ps)


def dct_matrix(n=L):
    """正交 DCT-II 矩阵，第 k 行是第 k 个频率。"""
    k, i = np.arange(n)[:, None], np.arange(n)[None]
    C = np.cos(np.pi * (i + 0.5) * k / n) * np.sqrt(2 / n)
    C[0] /= np.sqrt(2)
    return C


def op_dct(x):
    """每步去掉一个最高频分量（沿句子方向）。"""
    C = dct_matrix()
    return np.stack([C[:L - t].T @ C[:L - t] for t in range(K + 1)])


def op_random(x, seed=0):
    """固定的随机正交基，每步少保留一个方向。"""
    Q, _ = np.linalg.qr(np.random.RandomState(seed).randn(L, L))
    return np.stack([Q[:, :L - t] @ Q[:, :L - t].T for t in range(K + 1)])


def op_select_fl(x):
    """每步删一句：删掉之后 facility location 目标 Σ_i max_{j∈保留} sim(i,j) 下降最少的那句（参考 ToMA）。
    被删的位置在退化状态里是 0（标准化空间里的均值向量）。"""
    s = (_unit(x) @ _unit(x).T + 1) / 2
    keep = np.ones(L, bool)
    Ps = [np.diag(keep.astype(np.float64))]
    for _ in range(K):
        best, bi = -np.inf, -1
        for i in np.where(keep)[0]:
            k2 = keep.copy()
            k2[i] = False
            f = s[:, k2].max(1).sum()
            if f > best:
                best, bi = f, i
        keep[bi] = False
        Ps.append(np.diag(keep.astype(np.float64)))
    return np.stack(Ps)


def op_svd(x):
    """保留前 16−t 个左奇异向量张成的子空间。P_8 x 是 x 在"句子方向秩 8"下的最优近似（Eckart–Young）。"""
    U, _, _ = np.linalg.svd(x.astype(np.float64), full_matrices=False)
    return np.stack([U[:, :L - t] @ U[:, :L - t].T for t in range(K + 1)])


FUNCS = {op: globals()[f'op_{op}'] for op in OPS}
CONTENT_INDEPENDENT = {'merge_fixed', 'dct', 'random'}


def _one(args):
    op, x = args
    return FUNCS[op](x).astype(np.float32)


def compute_P(op, X, workers=16):
    """X: [N,16,d] → [N,9,16,16] float32"""
    X = np.asarray(X, dtype=np.float32)
    if op in CONTENT_INDEPENDENT:
        return np.broadcast_to(FUNCS[op](X[0]).astype(np.float32), (len(X), K + 1, L, L)).copy()
    if workers <= 1 or len(X) < 256:
        return np.stack([_one((op, x)) for x in X])
    from multiprocessing import Pool
    with Pool(workers) as p:
        return np.stack(p.map(_one, [(op, x) for x in X], chunksize=64))


# ---------------- 多层树用：有序正交基 ----------------
def basis(op, x, P8):
    """返回 B ∈ R^{16×8}，列正交、B Bᵀ = P_8，并且列的顺序有意义（拼接成上一层节点时就是这个顺序）。
    压缩：z = Bᵀ x（8×d）；展开：B z = P_8 x。
      合并类：每组一列 = 组内指示向量 / √组大小，按组内最靠前的位置排序（z_j = √s_j · 组均值）
      select_fl：保留的位置按原顺序，每个是单位向量（z 就是保留的原句）
      dct：前 8 个频率（低频在前）；random：固定随机正交基的前 8 列；svd：前 8 个左奇异向量
    合并类和选择类的 B 可以从 P_8 直接读出来，不需要另外存储；svd 需要存 B（就是 SIDE_INFO 里的 256 字节）。"""
    if op in ('merge_adj', 'merge_fixed', 'merge_global', 'select_fl'):
        cols, seen = [], set()
        for i in range(L):
            if P8[i, i] < 1e-6 or i in seen:
                continue
            members = np.where(P8[i] > 1e-6)[0]
            seen.update(members.tolist())
            v = np.zeros(L)
            v[members] = 1 / np.sqrt(len(members))
            cols.append(v)
        B = np.stack(cols, 1)
    elif op == 'dct':
        B = dct_matrix()[:L - K].T
    elif op == 'random':
        Q, _ = np.linalg.qr(np.random.RandomState(0).randn(L, L))
        B = Q[:, :L - K]
    elif op == 'svd':
        U, _, _ = np.linalg.svd(np.asarray(x, dtype=np.float64), full_matrices=False)
        B = U[:, :L - K]
    else:
        raise ValueError(op)
    assert B.shape == (L, L - K), (op, B.shape)
    return B.astype(np.float32)

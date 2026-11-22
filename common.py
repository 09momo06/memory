"""核心组件：相似合并退化、压缩/解包、还原网络、Cold Diffusion 采样、评测器。

约定
- 记忆块 x: [B, L, DIM]，L=16 句，每句一个 SONAR 向量（按维度标准化后）。
- 退化日程 S: [B, K+1, L] long。S[:, t, i] = 第 t 步时位置 i 所属的段号（段号从 0 连续编号、保持顺序）。
  S[:, 0] = 0..L-1（未退化），S[:, K] 共 L/2 段（压到一半）。
- 存储一个压缩块 = z [L/2, DIM]（每段均值）+ 日程 S（int8，约 144 字节，相对 z 可忽略）。
  因为日程被存下来，还原时直接用真实日程，不需要从预测结果重新推算。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
L, DIM = 16, 1024


# ---------------- 分段工具 ----------------
def seg_mean(x, seg):
    """x: [B,n,d], seg: [B,n] long → [B,n,d]，第 j 行是第 j 段的均值（多余行为 0）。"""
    B, n, d = x.shape
    idx = (seg + torch.arange(B, device=x.device)[:, None] * n).reshape(-1)
    s = torch.zeros(B * n, d, device=x.device, dtype=x.dtype).index_add_(0, idx, x.reshape(-1, d))
    c = torch.zeros(B * n, device=x.device, dtype=x.dtype).index_add_(
        0, idx, torch.ones(B * n, device=x.device, dtype=x.dtype))
    return (s / c.clamp(min=1)[:, None]).view(B, n, d)


def expand(x, seg):
    """退化状态：按 seg 分段求均值，再展开回原长度。[B,n,d]"""
    m = seg_mean(x, seg)
    return torch.gather(m, 1, seg[..., None].expand(-1, -1, x.shape[-1]))


def seg_info(seg):
    """每个位置所在段的大小、在段内的序号。[B,n] each"""
    n = seg.shape[1]
    oh = F.one_hot(seg, n)                      # [B,n,n]
    size = oh.sum(1).gather(1, seg)
    within = (oh.cumsum(1) * oh).sum(-1) - 1
    return size, within


# ---------------- 退化日程 ----------------
def sim_schedule(x, K=8, r=1):
    """相似合并（ToMe 式，限相邻）：每步合并 r 对余弦相似度最高、互不重叠的相邻段。
    x: np [n,d] → np [K+1, n]"""
    n0 = len(x)
    seg = np.arange(n0)
    segs = [seg.copy()]
    for _ in range(K):
        n = seg.max() + 1
        means = np.stack([x[seg == j].mean(0) for j in range(n)])
        mn = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-8)
        sim = (mn[:-1] * mn[1:]).sum(1)
        used = np.zeros(n, bool)
        chosen = set()
        for i in np.argsort(-sim, kind='stable'):
            if not used[i] and not used[i + 1]:
                chosen.add(int(i))
                used[i] = used[i + 1] = True
                if len(chosen) == r:
                    break
        assert len(chosen) == r, f'只找到 {len(chosen)} 对不重叠相邻对 (<r={r})，请减小 r、增大 K'
        new = np.zeros(n, int)
        k = -1
        for j in range(n):
            if j - 1 in chosen:
                new[j] = k
            else:
                k += 1
                new[j] = k
        seg = new[seg]
        segs.append(seg.copy())
    return np.stack(segs)


def fixed_schedule(n=L, K=8, r=1, seed=0):
    """固定池化 baseline：最终配对固定为 (0,1)(2,3)…，按固定随机顺序每步合并 r 对，与内容无关。"""
    order = np.random.RandomState(seed).permutation(n // 2)
    p = np.arange(n)
    segs = []
    for t in range(K + 1):
        merged = np.zeros(n // 2, bool)
        merged[order[:t * r]] = True
        key = np.where(merged[p // 2], (p // 2) * 2, p)
        segs.append(np.unique(key, return_inverse=True)[1].reshape(-1))
    return np.stack(segs)


def _sim_star(args):
    return sim_schedule(*args)


def schedules(X, kind='sim', K=8, r=1, workers=16):
    """X: np [B,n,d] → np int8 [B,K+1,n]"""
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[1]
    if kind == 'fixed':
        return np.broadcast_to(fixed_schedule(n, K, r), (len(X), K + 1, n)).astype(np.int8).copy()
    if len(X) < 256 or workers <= 1:
        out = [sim_schedule(x, K, r) for x in X]
    else:
        from multiprocessing import Pool
        with Pool(workers) as p:
            out = p.map(_sim_star, [(x, K, r) for x in X], chunksize=64)
    return np.stack(out).astype(np.int8)


# ---------------- 压缩 / 解包 ----------------
def compress(X, S):
    """X: torch [B,n,d], S: torch long [B,K+1,n] → z [B,n/2,d]"""
    n = X.shape[1]
    assert int(S[:, -1].max()) + 1 == n // 2 and bool((S[:, -1].max(1).values + 1 == n // 2).all())
    return seg_mean(X, S[:, -1])[:, :n // 2]


def unpack(z, S):
    """存储的 z + 日程 → 展开态 x_K [B,n,d]（等于 expand(x0, S[:,K])）"""
    return torch.gather(z, 1, S[:, -1, :, None].expand(-1, -1, z.shape[-1]))


# ---------------- 还原网络 ----------------
class Restorer(nn.Module):
    def __init__(self, K=8, d_in=DIM, d=512, layers=6, n=L, levels=4):
        super().__init__()
        self.K = K
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, n, d))
        self.t_emb = nn.Embedding(K + 1, d)
        self.lv_emb = nn.Embedding(levels, d)
        self.size_emb = nn.Embedding(n + 1, d)
        self.within_emb = nn.Embedding(n, d)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, d_in)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)          # 初始输出 = 输入，即"不还原"基线

    def forward(self, xt, t, lv, seg):
        size, within = seg_info(seg)
        h = (self.inp(xt) + self.pos + self.t_emb(t)[:, None] + self.lv_emb(lv)[:, None]
             + self.size_emb(size) + self.within_emb(within))
        return xt + self.out(self.enc(h))      # 预测 x0


def load_model(path):
    ck = torch.load(path, map_location=DEV)
    m = Restorer(K=ck['K']).to(DEV)
    m.load_state_dict(ck['model'])
    m.eval()
    return m


# ---------------- 采样 ----------------
@torch.no_grad()
def restore(model, x_K, S, lv=0, mode='cold', project=False, return_traj=False):
    """从展开态 x_K 还原到 x0。
    mode: cold = Cold Diffusion 改进采样 x <- x - D(x0p,t) + D(x0p,t-1)
          naive = x <- D(x0p, t-1)
          oneshot = 只调用一次网络（t=K）
    project: 把输出投影回"与存储的 z 一致"（每段均值 = z）。cold 模式天然满足，无需投影。
    """
    K = S.shape[1] - 1
    B = len(x_K)
    x = x_K
    traj = []
    lvv = torch.full((B,), lv, device=x.device, dtype=torch.long)
    for t in range(K, 0, -1):
        tt = torch.full((B,), t, device=x.device, dtype=torch.long)
        x0p = model(x, tt, lvv, S[:, t]).float()
        if mode == 'oneshot':
            x = x0p
            traj.append(x)
            break
        if mode == 'cold':
            x = x - expand(x0p, S[:, t]) + expand(x0p, S[:, t - 1])
        elif mode == 'naive':
            x = expand(x0p, S[:, t - 1])
        else:
            raise ValueError(mode)
        traj.append(x)
    if project:
        x = x - expand(x, S[:, K]) + x_K
    return (x, traj) if return_traj else x


# ---------------- 评测 ----------------
class Evaluator:
    """cos：还原句向量与原句向量的平均余弦（反标准化后的 SONAR 空间）。
    top1：在整个 test 集句子池里，离还原向量最近的是否正是原句。"""

    def __init__(self, split='test', data='data'):
        self.mu = torch.tensor(np.load(f'{data}/mu.npy'), device=DEV).float()
        self.sg = torch.tensor(np.load(f'{data}/sigma.npy'), device=DEV).float()
        self.X = torch.tensor(np.load(f'{data}/{split}_x.npy').astype(np.float32), device=DEV)
        self.pool = F.normalize(self.denorm(self.X.reshape(-1, DIM)), dim=-1)

    def denorm(self, x):
        return x.float() * self.sg + self.mu

    @torch.no_grad()
    def score(self, xh, block_idx):
        """xh: [B,L,DIM]（标准化空间），block_idx: [B] 对应的 test 块号"""
        block_idx = torch.as_tensor(block_idx, device=DEV).long()
        n = xh.shape[1]
        gold = (block_idx[:, None] * n + torch.arange(n, device=DEV)).reshape(-1)
        q = F.normalize(self.denorm(xh.reshape(-1, DIM)), dim=-1)
        cos = (q * self.pool[gold]).sum(-1)
        hit = torch.cat([(q[i:i + 4096] @ self.pool.T).argmax(-1) == gold[i:i + 4096]
                         for i in range(0, len(q), 4096)])
        return round(cos.mean().item(), 4), round(hit.float().mean().item(), 4)

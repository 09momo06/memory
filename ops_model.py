"""算子对比用的通用还原网络和 cold 采样（适用于 ops.py 里的所有投影算子）。
网络结构和主实验的 Restorer 一样（6 层 Transformer，宽度 512，约 2000 万参数），
只是条件信息从"段大小 / 段内序号"换成了投影矩阵 P_t 的每一行（每个位置拿到自己那一行，16 维）。
所以不同算子用的是完全相同的网络和训练设置，对比是公平的。
levels > 0 时额外加一个层级编码（多层树实验用）；levels = 0 时和单层实验的网络完全相同，旧 checkpoint 照常加载。"""
import torch
import torch.nn as nn
from common import DEV, DIM, L


class RestorerP(nn.Module):
    def __init__(self, K=8, d_in=DIM, d=512, layers=6, n=L, dropout=0.1, levels=0):
        super().__init__()
        self.K = K
        self.levels = levels
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, n, d))
        self.t_emb = nn.Embedding(K + 1, d)
        self.p_proj = nn.Linear(n, d)
        if levels:
            self.lv_emb = nn.Embedding(levels, d)
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, d_in)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)          # 初始输出 = 输入，即"不还原"

    def forward(self, xt, t, P, lv=None):
        h = self.inp(xt) + self.pos + self.t_emb(t)[:, None] + self.p_proj(P)
        if self.levels:
            if lv is None:
                lv = torch.zeros_like(t)
            h = h + self.lv_emb(lv)[:, None]
        return xt + self.out(self.enc(h))      # 预测 x0


def degrade(x, P):
    """x: [B,L,d], P: [B,L,L] → P x"""
    return torch.bmm(P, x)


def load_model_p(path):
    ck = torch.load(path, map_location=DEV)
    m = RestorerP(K=ck['K'], levels=ck.get('levels', 0)).to(DEV)
    m.load_state_dict(ck['model'])
    m.eval()
    return m


@torch.no_grad()
def restore_p(model, x_K, P, return_traj=False, lv=None):
    """P: [B,K+1,L,L]。Cold Diffusion 改进采样：x ← x − P_t x̂0 + P_{t−1} x̂0。
    lv: 整数或 None，所还原节点的层级（0 = 叶子）。"""
    K = P.shape[1] - 1
    x, traj = x_K, []
    lvv = None if lv is None else torch.full((len(x),), int(lv), device=x.device, dtype=torch.long)
    for t in range(K, 0, -1):
        tt = torch.full((len(x),), t, device=x.device, dtype=torch.long)
        x0p = model(x, tt, P[:, t], lvv).float()
        x = x - degrade(x0p, P[:, t]) + degrade(x0p, P[:, t - 1])
        traj.append(x)
    return (x, traj) if return_traj else x

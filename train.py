"""训练还原网络 R(x_t, t, level, seg) → x0。
主实验：python train.py --multilevel --out ckpt/main.pt          （第 0/1/2 层数据混合训练）
对照（可选）：python train.py --sched fixed --out ckpt/fixed.pt
验证 loss 在 valid 集第 0 层上计算；每次验证创新低时保存一次（ckpt 取验证最优）。"""
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from common import DEV, Restorer, expand

p = argparse.ArgumentParser()
p.add_argument('--data', default='data')
p.add_argument('--sched', default='sim', choices=['sim', 'fixed'])
p.add_argument('--K', type=int, default=8)
p.add_argument('--multilevel', action='store_true')
p.add_argument('--only_tK', action='store_true', help='只在 t=K 上训练（一步到位还原的对照）')
p.add_argument('--steps', type=int, default=30000)
p.add_argument('--bs', type=int, default=256)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--out', required=True)
a = p.parse_args()
torch.manual_seed(0)
best = float('inf')

Xs = [np.load(f'{a.data}/train_x.npy')]
Ss = [np.load(f'{a.data}/train_seg_{a.sched}_K{a.K}.npy')]
LVs = [np.zeros(len(Xs[0]), np.int64)]
if a.multilevel:
    assert a.sched == 'sim' and a.K == 8
    for lv in [1, 2]:
        Xs.append(np.load(f'{a.data}/train_x_lv{lv}.npy'))
        Ss.append(np.load(f'{a.data}/train_seg_lv{lv}.npy'))
        LVs.append(np.full(len(Xs[-1]), lv, np.int64))
X = torch.tensor(np.concatenate(Xs)).to(DEV)                       # fp16
S = torch.tensor(np.concatenate(Ss).astype(np.int64)).to(DEV)
LV = torch.tensor(np.concatenate(LVs)).to(DEV)
print('train samples', len(X), 'per level', [len(x) for x in Xs])

Xv = torch.tensor(np.load(f'{a.data}/valid_x.npy')[:1024].astype(np.float32)).to(DEV)
Sv = torch.tensor(np.load(f'{a.data}/valid_seg_{a.sched}_K{a.K}.npy')[:1024].astype(np.int64)).to(DEV)
gv = torch.Generator(device=DEV).manual_seed(1)
tv = torch.full((len(Xv),), a.K, device=DEV) if a.only_tK else torch.randint(1, a.K + 1, (len(Xv),), device=DEV, generator=gv)
xtv = expand(Xv, Sv[torch.arange(len(Xv), device=DEV), tv])


def lossf(pred, x0):
    return F.mse_loss(pred, x0) + (1 - F.cosine_similarity(pred, x0, dim=-1)).mean()


@torch.no_grad()
def val_loss():
    model.eval()
    l = lossf(model(xtv, tv, torch.zeros_like(tv), Sv[torch.arange(len(Xv), device=DEV), tv]).float(), Xv).item()
    model.train()
    return l


val_base = lossf(xtv, Xv).item()
model = Restorer(K=a.K).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
use_amp = DEV.type == 'cuda'
t0 = time.time()
log = []
for step in range(a.steps):
    idx = torch.randint(0, len(X), (a.bs,), device=DEV)
    x0 = X[idx].float()
    t = torch.full((a.bs,), a.K, device=DEV) if a.only_tK else torch.randint(1, a.K + 1, (a.bs,), device=DEV)
    seg = S[idx, t]
    xt = expand(x0, seg)
    with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=use_amp):
        pred = model(xt, t, LV[idx], seg)
    pred = pred.float()
    loss = lossf(pred, x0)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 500 == 0 or step == a.steps - 1:
        base = lossf(xt, x0).item()
        vl = val_loss()
        print(f'step {step} train {loss.item():.4f} (no-restore {base:.4f}) | val {vl:.4f} (no-restore {val_base:.4f}) {time.time() - t0:.0f}s', flush=True)
        log.append({'step': step, 'train': loss.item(), 'train_base': base, 'val': vl, 'val_base': val_base})
        if vl < best:
            best = vl
            torch.save({'model': model.state_dict(), 'K': a.K, 'args': vars(a), 'step': step, 'val': vl}, a.out)
json.dump(log, open(a.out.replace('.pt', '_log.json'), 'w'))
print(f'saved best ckpt to {a.out} (val {best:.4f}, no-restore {val_base:.4f})')

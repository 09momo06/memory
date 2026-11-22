"""训练某个算子的还原网络。
单层：python train_ops.py --op dct --out ckpt/ops/dct.pt                         （只用第 0 层数据）
多层：python train_ops.py --op dct --multilevel --out ckpt/ops_tree/dct.pt       （第 0/1/2 层混合，加层级编码）
验证 loss 在 valid 集第 0 层上计算；验证创新低时保存（ckpt 取验证最优）。"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from common import DEV
from ops import OPS
from ops_model import RestorerP, degrade

p = argparse.ArgumentParser()
p.add_argument('--data', default='data')
p.add_argument('--op', required=True, choices=OPS)
p.add_argument('--steps', type=int, default=15000)
p.add_argument('--bs', type=int, default=256)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--eval_every', type=int, default=500)
p.add_argument('--multilevel', action='store_true')
p.add_argument('--out', required=True)
a = p.parse_args()
torch.manual_seed(0)
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
K = 8

Xs = [np.load(f'{a.data}/train_x.npy')]
Ps = [np.load(f'{a.data}/ops/train_P_{a.op}.npy')]
if a.multilevel:
    for lv in [1, 2]:
        Xs.append(np.load(f'{a.data}/ops/train_x_lv{lv}_{a.op}.npy'))
        Ps.append(np.load(f'{a.data}/ops/train_P_lv{lv}_{a.op}.npy'))
LV = torch.tensor(np.concatenate([np.full(len(x), lv) for lv, x in enumerate(Xs)])).long().to(DEV)
X = torch.tensor(np.concatenate(Xs).astype(np.float16)).to(DEV)                  # fp16
P = torch.tensor(np.concatenate(Ps).astype(np.float32)).to(DEV)                  # fp32 [N,9,16,16]
print('samples per level', [len(x) for x in Xs])
del Xs, Ps
LEVELS = 3 if a.multilevel else 0
Xv = torch.tensor(np.load(f'{a.data}/valid_x.npy')[:1024].astype(np.float32)).to(DEV)
Pv = torch.tensor(np.load(f'{a.data}/ops/valid_P_{a.op}.npy')[:1024]).to(DEV)
gv = torch.Generator(device=DEV).manual_seed(1)
tv = torch.randint(1, K + 1, (len(Xv),), device=DEV, generator=gv)
ar = torch.arange(len(Xv), device=DEV)
xtv = degrade(Xv, Pv[ar, tv])
print('op', a.op, 'train', tuple(X.shape), 'valid', tuple(Xv.shape))


def lossf(pred, x0):
    return F.mse_loss(pred, x0) + (1 - F.cosine_similarity(pred, x0, dim=-1)).mean()


model = RestorerP(K=K, levels=LEVELS).to(DEV)


@torch.no_grad()
def val_loss():
    model.eval()
    l = lossf(model(xtv, tv, Pv[ar, tv], torch.zeros_like(tv) if LEVELS else None).float(), Xv).item()
    model.train()
    return l


val_base = lossf(xtv, Xv).item()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=max(0.05, 2 / a.steps))
use_amp = DEV.type == 'cuda'
best, log, t0 = float('inf'), [], time.time()
for step in range(a.steps):
    idx = torch.randint(0, len(X), (a.bs,), device=DEV)
    x0 = X[idx].float()
    t = torch.randint(1, K + 1, (a.bs,), device=DEV)
    Pt = P[idx, t]
    xt = degrade(x0, Pt)
    with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=use_amp):
        pred = model(xt, t, Pt, LV[idx] if LEVELS else None)
    loss = lossf(pred.float(), x0)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % a.eval_every == 0 or step == a.steps - 1:
        vl = val_loss()
        print(f'step {step} train {loss.item():.4f} | val {vl:.4f} (no-restore {val_base:.4f}) {time.time() - t0:.0f}s', flush=True)
        log.append({'step': step, 'train': loss.item(), 'val': vl, 'val_base': val_base})
        if vl < best:
            best = vl
            torch.save({'model': model.state_dict(), 'K': K, 'levels': LEVELS, 'op': a.op, 'args': vars(a), 'step': step, 'val': vl}, a.out)
json.dump(log, open(a.out.replace('.pt', '_log.json'), 'w'))
best_step = min(log, key=lambda r: r['val'])['step']
print(f'saved best ckpt to {a.out} (step {best_step}, val {best:.4f}, no-restore {val_base:.4f}, '
      f'reduction {(val_base - best) / val_base * 100:.2f}%)')

"""Validate actual stage-3 data and record measured dimensions and invariants."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['encoding','complete'],required=True)
a=p.parse_args()
root=Path(__file__).resolve().parent
data=root/'data'
results=root/'results'
report={'mode':a.mode,'arrays':{}}
mu=np.load(data/'mu.npy')
sigma=np.load(data/'sigma.npy')
assert mu.shape==sigma.shape==(1024,)
assert np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma>0).all()
report['mu_shape']=list(mu.shape)
report['sigma_shape']=list(sigma.shape)
report['sigma_min']=float(sigma.min())
report['sigma_max']=float(sigma.max())
blocks={}
for split in ['train','valid','test']:
    convs=[json.loads(line) for line in (data/f'{split}_convs.jsonl').read_text().splitlines()]
    n=sum(len(c)//16 for c in convs)
    blocks[split]=n
    emb=np.load(data/f'{split}_emb.npy',mmap_mode='r')
    x=np.load(data/f'{split}_x.npy',mmap_mode='r')
    meta=np.load(data/f'{split}_meta.npy')
    sents=json.loads((data/f'{split}_sents.json').read_text())
    assert emb.shape==(n*16,1024) and x.shape==(n,16,1024)
    assert x.dtype==np.float16 and meta.shape==(n,2)
    assert sents==[s for c in convs for s in c]
    assert np.array_equal(meta,np.array([(ci,bi) for ci,c in enumerate(convs) for bi in range(len(c)//16)]))
    for i in range(0,n,1024):
        assert np.isfinite(x[i:i+1024]).all()
    for bi in sorted({0,n//2,n-1}):
        expected=((emb[bi*16:(bi+1)*16]-mu)/sigma).astype(np.float16)
        assert np.array_equal(x[bi],expected)
    report['arrays'][f'{split}_x.npy']={'shape':list(x.shape),'dtype':str(x.dtype),'conversations':len(convs),'sentences':len(sents)}
    report['arrays'][f'{split}_emb.npy']={'shape':list(emb.shape),'dtype':str(emb.dtype)}
    report['arrays'][f'{split}_meta.npy']={'shape':list(meta.shape),'dtype':str(meta.dtype)}
# Check every training vector for mean/std scale after fp16 normalization.
x=np.load(data/'train_x.npy',mmap_mode='r')
sums=np.zeros(1024,np.float64)
sq=np.zeros(1024,np.float64)
for i in range(0,len(x),512):
    part=x[i:i+512].reshape(-1,1024).astype(np.float64)
    sums+=part.sum(0)
    sq+=np.square(part).sum(0)
mean=sums/(len(x)*16)
std=np.sqrt(sq/(len(x)*16)-np.square(mean))
source=np.load(data/'train_emb.npy',mmap_mode='r')
reference_mean=source.mean(0,dtype=np.float64)
reference_std=source.std(0,dtype=np.float64)
assert np.array_equal(mu,reference_mean.astype(np.float32))
assert np.array_equal(sigma,(reference_std+1e-6).astype(np.float32))
expected_mean=(reference_mean-mu)/sigma
expected_std=reference_std/sigma
assert np.max(np.abs(mean))<1e-3
assert np.max(np.abs(mean-expected_mean))<1e-3
# sigma includes the existing +1e-6 stabilizer, so low-variance dimensions
# intentionally have standard deviation below 1; check against that formula.
assert np.max(np.abs(std-expected_std))<1e-3
report['normalized_train_mean_abs_max']=float(np.max(np.abs(mean)))
report['normalized_train_std_min']=float(std.min())
report['normalized_train_std_max']=float(std.max())
report['expected_normalized_std_min']=float(expected_std.min())
report['normalized_train_mean_error_abs_max']=float(np.max(np.abs(mean-expected_mean)))
report['normalized_train_std_error_abs_max']=float(np.max(np.abs(std-expected_std)))
report['statistics_accumulation_dtype']='float64'
report['statistics_storage_dtype']='float32'

def validate_schedule(name,n):
    seg=np.load(data/name,mmap_mode='r')
    assert seg.shape==(n,9,16) and seg.dtype==np.int8
    assert (seg[:,0]==np.arange(16)).all()
    assert (seg[:,:,0]==0).all()
    assert (seg.max(-1)+1==16-np.arange(9)).all()
    diff=np.diff(seg,axis=-1)
    assert ((diff>=0)&(diff<=1)).all()
    # Positions already merged at t cannot split at t+1.
    assert ((seg[:,:-1,1:]!=seg[:,:-1,:-1]) | (seg[:,1:,1:]==seg[:,1:,:-1])).all()
    report['arrays'][name]={'shape':list(seg.shape),'dtype':str(seg.dtype)}
    return seg

if a.mode=='complete':
    import torch
    from common import compress,sim_schedule
    for split,n in blocks.items():
        seg=validate_schedule(f'{split}_seg_sim_K8.npy',n)
        x=np.load(data/f'{split}_x.npy',mmap_mode='r')
        for i in sorted({0,n//2,n-1}):
            assert np.array_equal(seg[i],sim_schedule(x[i].astype(np.float32),8,1))
    train=np.load(data/'train_x.npy',mmap_mode='r')
    meta=np.load(data/'train_meta.npy')
    starts=np.array([i for i in range(len(train)-2) if meta[i+2,0]==meta[i,0]])
    starts=starts[np.random.RandomState(0).permutation(len(starts))[:30000]]
    report['multilevel_samples']=len(starts)
    for lv in [1,2]:
        x=np.load(data/f'train_x_lv{lv}.npy',mmap_mode='r')
        assert x.shape==(len(starts),16,1024) and x.dtype==np.float16 and len(x)>0
        for i in range(0,len(x),1024):
            assert np.isfinite(x[i:i+1024]).all()
        report['arrays'][f'train_x_lv{lv}.npy']={'shape':list(x.shape),'dtype':str(x.dtype)}
        validate_schedule(f'train_seg_lv{lv}.npy',len(starts))
    s=np.load(data/'train_seg_sim_K8.npy',mmap_mode='r')
    lv1=np.load(data/'train_x_lv1.npy',mmap_mode='r')
    lv2=np.load(data/'train_x_lv2.npy',mmap_mode='r')
    s1=np.load(data/'train_seg_lv1.npy',mmap_mode='r')
    for j in sorted({0,len(starts)//2,len(starts)-1}):
        start=starts[j]
        assert meta[start,0]==meta[start+1,0]==meta[start+2,0]
        tx=torch.tensor(np.array(train[start:start+3]),dtype=torch.float32)
        ts=torch.tensor(np.array(s[start:start+3]),dtype=torch.long)
        z=compress(tx,ts)
        c1=torch.cat([z[0],z[1]],0)[None]
        assert np.allclose(lv1[j],c1[0].numpy(),rtol=1e-3,atol=2e-3)
        zc1=compress(c1,torch.tensor(np.array(s1[j:j+1]),dtype=torch.long))
        c2=torch.cat([zc1,z[2:3]],1)
        assert np.allclose(lv2[j],c2[0].numpy(),rtol=1e-3,atol=2e-3)
    report['unit_test_passed']='all ok' in (root/'logs/test_common_stage3.log').read_text()
    report['smoke_test_passed']='SMOKE TEST PASSED' in (root/'logs/smoke_stage3.log').read_text()
    assert report['unit_test_passed'] and report['smoke_test_passed']
report['passed']=True
name='stage3_encoding_validation.json' if a.mode=='encoding' else 'stage3_validation.json'
(results/name).write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False),flush=True)

"""Validate completion, the best checkpoint, and the required loss comparison."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import torch
from common import Restorer

root=Path(__file__).resolve().parent
preflight=json.loads((root/'results/stage4_preflight.json').read_text())
logs=json.loads((root/'ckpt/main_log.json').read_text())
ck=torch.load(root/'ckpt/main.pt',map_location='cpu',weights_only=True)
args=ck['args']
assert args['multilevel'] and args['sched']=='sim' and args['K']==8 and not args['only_tK']
assert args['steps']==30000
assert logs and logs[-1]['step']==args['steps']-1
expected_steps=list(range(0,args['steps'],500))+[args['steps']-1]
assert [entry['step'] for entry in logs]==expected_steps
for entry in logs:
    assert all(math.isfinite(entry[key]) for key in ['train','train_base','val','val_base'])
best=min(logs,key=lambda entry:entry['val'])
base=logs[0]['val_base']
assert all(entry['val_base']==base for entry in logs)
assert ck['step']==best['step'] and ck['val']==best['val']
assert ck['K']==8
model=Restorer(K=ck['K'])
model.load_state_dict(ck['model'],strict=True)
assert all(torch.isfinite(value).all() for value in model.state_dict().values())
text=(root/'logs/main.log').read_text()
assert 'saved best ckpt' in text.splitlines()[-1]
samples=re.search(r'train samples (\d+) per level (\[[^\n]+\])',text)
assert samples is not None
assert int(samples.group(1))==preflight['train_samples_total']
assert ast.literal_eval(samples.group(2))==preflight['train_samples_by_level']
for name,sha in preflight['source_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
passed=best['val']<base
report={'passed':passed,'checkpoint_valid':True,'finite_parameters':True,
        'checkpoint':'ckpt/main.pt','training_log':'ckpt/main_log.json',
        'completed_steps':args['steps'],'last_logged_step':logs[-1]['step'],
        'best_step':ck['step'],'step_indexing':'zero-based, as in train.py log',
        'best_validation_loss':best['val'],'no_restore_validation_loss':base,
        'absolute_loss_reduction':base-best['val'],
        'relative_loss_reduction':(base-best['val'])/base if base else None,
        'train_samples_total':preflight['train_samples_total'],
        'train_samples_by_level':preflight['train_samples_by_level'],
        'train_args':args,'log_entries':len(logs),
        'parameter_count':sum(parameter.numel() for parameter in model.parameters())}
(root/'results/stage4_training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False),flush=True)
if passed:
    (root/'results/STAGE4_COMPLETE').touch()
else:
    print('VALIDATION NOT BETTER THAN BASELINE: follow the documented learning-rate retry.',flush=True)
sys.exit(0 if passed else 2)

"""Precompute the existing seven operators, single-process, without changing data."""
import argparse
from pathlib import Path
import numpy as np
from ops import OPS, compute_P

p = argparse.ArgumentParser()
p.add_argument('--data', default='data')
a = p.parse_args()
data = Path(a.data)
out = data / 'ops'
out.mkdir(parents=True, exist_ok=True)
for split in ('train', 'valid', 'test'):
    X = np.load(data / f'{split}_x.npy', mmap_mode='r')
    assert X.ndim == 3 and X.shape[1:] == (16, 1024), X.shape
    for op in OPS:
        path = out / f'{split}_P_{op}.npy'
        assert not path.exists(), f'Refusing to overwrite {path}'
        P = compute_P(op, X, workers=1)
        assert P.shape == (len(X), 9, 16, 16)
        assert np.isfinite(P).all()
        np.save(path, P)
        print(split, op, P.shape, P.dtype, flush=True)

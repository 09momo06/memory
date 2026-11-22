"""冒烟测试用假数据（不需要 SONAR）。句向量 = 文章主题 + AR(1) 漂移 + 偶发话题突变。
用法：python make_fake_data.py --out data_fake"""
import argparse, json, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--out', default='data_fake')
p.add_argument('--n_train_art', type=int, default=60)
p.add_argument('--n_test_art', type=int, default=15)
p.add_argument('--n_valid_art', type=int, default=10)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.RandomState(0)
L, D = 16, 1024


def article(n_blocks):
    topic = rng.randn(D)
    v = rng.randn(D)
    out = []
    for _ in range(n_blocks * L):
        if rng.rand() < 0.1:
            v = rng.randn(D)                      # 话题突变
        v = 0.8 * v + 0.6 * rng.randn(D)
        out.append(topic + v)
    return np.array(out)


for name, n_art in [('train', a.n_train_art), ('valid', a.n_valid_art), ('test', a.n_test_art)]:
    arts = [article(rng.randint(4, 8)) for _ in range(n_art)]
    E = np.concatenate(arts)
    if name == 'train':
        mu, sg = E.mean(0), E.std(0) + 1e-6
        np.save(f'{a.out}/mu.npy', mu.astype(np.float32))
        np.save(f'{a.out}/sigma.npy', sg.astype(np.float32))
    X = ((E - mu) / sg).reshape(-1, L, D).astype(np.float16)
    meta = np.array([(ai, bi) for ai, art in enumerate(arts) for bi in range(len(art) // L)])
    np.save(f'{a.out}/{name}_x.npy', X)
    np.save(f'{a.out}/{name}_emb.npy', E.astype(np.float32))
    np.save(f'{a.out}/{name}_meta.npy', meta)
    json.dump([f'假句子{i}' for i in range(len(E))], open(f'{a.out}/{name}_sents.json', 'w'), ensure_ascii=False)
    print(name, X.shape)

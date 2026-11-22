"""第 1 步：句子编码，保存原始向量 data/{split}_emb.npy（fp32，未标准化）。
第 2 步（--normalize）：用 train 的 mu/sigma 按维度标准化，得到 data/{split}_x.npy [N,16,1024] fp16。
用法：
  python encode.py --splits test               # 先只编码测试集，用来测 SONAR 上限（E0）
  python encode.py --splits train valid        # E0 过关后再编码其余
  python encode.py --normalize                 # 生成 *_x.npy、*_meta.npy、*_sents.json、mu.npy、sigma.npy
备选：--backend bge（BAAI/bge-m3，不能解码，只做向量空间评测）"""
import argparse, json, os
import numpy as np

L = 16
SPLITS = ['train', 'valid', 'test']
p = argparse.ArgumentParser()
p.add_argument('--splits', nargs='+', default=SPLITS)
p.add_argument('--normalize', action='store_true')
p.add_argument('--backend', default='sonar', choices=['sonar', 'bge'])
p.add_argument('--bs', type=int, default=256)
a = p.parse_args()


def load_convs(name):
    return [json.loads(l) for l in open(f'data/{name}_convs.jsonl')]


if a.normalize:
    E = np.load('data/train_emb.npy')
    # Accumulate large-corpus statistics in float64 to avoid float32 mean drift
    # in low-variance dimensions; use the stored float32 values for normalization.
    mu = E.mean(0, dtype=np.float64).astype(np.float32)
    sg = (E.std(0, dtype=np.float64) + 1e-6).astype(np.float32)
    np.save('data/mu.npy', mu.astype(np.float32))
    np.save('data/sigma.npy', sg.astype(np.float32))
    for name in SPLITS:
        convs = load_convs(name)
        E = np.load(f'data/{name}_emb.npy')
        X = ((E - mu) / sg).reshape(-1, L, E.shape[1]).astype(np.float16)
        meta = np.array([(ci, bi) for ci, c in enumerate(convs) for bi in range(len(c) // L)])
        assert len(meta) == len(X)
        np.save(f'data/{name}_x.npy', X)
        np.save(f'data/{name}_meta.npy', meta)
        json.dump([s for c in convs for s in c], open(f'data/{name}_sents.json', 'w'), ensure_ascii=False)
        print(name, X.shape)
    raise SystemExit

import torch
if a.backend == 'sonar':
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    enc = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                       tokenizer="text_sonar_basic_encoder", device=torch.device("cuda"))
    encode = lambda ss: enc.predict(ss, source_lang="eng_Latn", batch_size=a.bs).float().cpu().numpy()
else:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer('BAAI/bge-m3', device='cuda')
    encode = lambda ss: st.encode(ss, batch_size=a.bs, convert_to_numpy=True).astype(np.float32)

for name in a.splits:
    flat = [s for c in load_convs(name) for s in c]
    chunks = []
    for i in range(0, len(flat), 20000):
        chunks.append(encode(flat[i:i + 20000]))
        print(name, i + len(chunks[-1]), '/', len(flat), flush=True)
    E = np.concatenate(chunks)
    assert E.shape == (len(flat), 1024), E.shape
    np.save(f'data/{name}_emb.npy', E)
    # E0 decodes before --normalize, so its reference sentences must already exist.
    with open(f'data/{name}_sents.json', 'w') as f:
        json.dump(flat, f, ensure_ascii=False)
open('data/BACKEND', 'w').write(a.backend)

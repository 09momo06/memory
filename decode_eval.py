"""把向量解码回文本，计算 BLEU-4 / ROUGE-L / chrF / 实体召回，输出对照示例。
--src ceiling             不压缩，test 原始 SONAR 向量直接解码（E0：系统上限）
--src results/xxx.pt      解码 eval.py / tree_eval.py 保存的还原结果（dict: x[标准化空间], blocks）
例：python decode_eval.py --src ceiling --name ceiling
    python decode_eval.py --src results/restored_main_cold.pt --name main_cold"""
import argparse, json
import numpy as np
import sacrebleu
import torch
from rouge_score import rouge_scorer
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from entity import entity_recall, split_speaker

p = argparse.ArgumentParser()
p.add_argument('--data', default='data')
p.add_argument('--src', required=True)
p.add_argument('--name', required=True)
p.add_argument('--nblk', type=int, default=100000, help='最多解码多少个块（每块 16 句；默认全部，与压缩实验的 846 块一致）')
p.add_argument('--bs', type=int, default=32, help='解码 batch size')
p.add_argument('--results', default='results')
a = p.parse_args()
L = 16
dec = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder",
                                   tokenizer="text_sonar_basic_encoder", device=torch.device("cuda"))
sents = json.load(open(f'{a.data}/test_sents.json'))
if a.src == 'ceiling':
    E = np.load(f'{a.data}/test_emb.npy')[:a.nblk * L]
    blocks = np.arange(len(E) // L)
else:
    d = torch.load(a.src)
    X, blocks = d['x'].float().numpy()[:a.nblk], np.asarray(d['blocks'])[:a.nblk]
    mu, sg = np.load(f'{a.data}/mu.npy'), np.load(f'{a.data}/sigma.npy')
    E = X.reshape(-1, X.shape[-1]) * sg + mu
refs = [sents[b * L + i] for b in blocks for i in range(L)]
# 每句所在对话的全部说话人（用于判断"换人"）
meta = np.load(f'{a.data}/test_meta.npy')
conv_spk = {}
for b, (ci, _) in enumerate(meta):
    conv_spk.setdefault(int(ci), set()).update(split_speaker(sents[b * L + i])[0] for i in range(L))
spk_sets = [conv_spk[int(meta[b][0])] for b in blocks for _ in range(L)]
hyp = dec.predict(torch.tensor(E, dtype=torch.float32), target_lang="eng_Latn", max_seq_len=256, batch_size=a.bs)
assert len(hyp) == len(refs), (len(hyp), len(refs))
# Preserve all aligned outputs for entity-loss overlap and reproducible examples.
with open(f'{a.results}/predictions_{a.name}.jsonl', 'w') as f:
    for i, (ref, text) in enumerate(zip(refs, hyp)):
        block = int(blocks[i // L])
        f.write(json.dumps({'block': block, 'position': i % L,
                            'sentence_index': block * L + i % L,
                            'ref': ref, 'hyp': text}, ensure_ascii=False) + '\n')
sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rl = np.mean([sc.score(r, h)['rougeL'].fmeasure for r, h in zip(refs, hyp)])
res = {'name': a.name, 'n_sent': len(refs),
       'BLEU4': round(sacrebleu.corpus_bleu(hyp, [refs]).score, 2),
       'ROUGE_L': round(100 * float(rl), 2),
       'chrF': round(sacrebleu.corpus_chrf(hyp, [refs]).score, 2),
       **entity_recall(refs, hyp, spk_sets)}
print(json.dumps(res))
json.dump(res, open(f'{a.results}/dec_{a.name}.json', 'w'))
with open(f'{a.results}/samples_{a.name}.txt', 'w') as f:
    for k in range(min(5, len(blocks))):
        f.write(f'===== block {blocks[k]} =====\n')
        for i in range(k * L, (k + 1) * L):
            f.write(f'REF: {refs[i]}\nHYP: {hyp[i]}\n\n')

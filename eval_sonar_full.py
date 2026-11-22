"""E0 over every LoCoMo text sentence, including incomplete-block tails."""
import argparse
from collections import Counter
import datetime
import gc
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import sacrebleu
import torch
from rouge_score import rouge_scorer
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from prep_data import load_locomo
from entity import entity_recall

p=argparse.ArgumentParser()
p.add_argument('--bs', type=int, default=32)
a=p.parse_args()
root=Path(__file__).resolve().parent
os.chdir(root)
results=root/'results'
def stamp():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def save(name, value):
    path=results/name
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4'
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert Path('data/BACKEND').read_text().strip()=='sonar'
full=load_locomo()
old=[json.loads(line) for line in open('data/test_convs.jsonl')]
old_sents=json.loads(Path('data/test_sents.json').read_text())
old_emb=np.load('data/test_emb.npy')
assert len(full)==len(old)==10
assert old_sents==[s for c in old for s in c]
assert old_emb.shape==(len(old_sents),1024) and old_emb.dtype==np.float32
for original, processed in zip(full,old):
    assert original[:len(processed)]==processed
assert not (results/'dec_ceiling_full.json').exists(), 'Full result already exists; preserve it before rerunning'
tails=[s for c,b in zip(full,old) for s in c[len(b):]]
coverage={'n_conversations':len(full),'n_sent':sum(map(len,full)),
          'n_existing_block_sentences':len(old_sents),'n_added_tail_sentences':len(tails),
          'decoder_batch_size':a.bs,'max_seq_len':256,
          'conversations':[{'conversation_index':i,'n_sent':len(c),'n_existing_block_sentences':len(b),
                            'n_added_tail_sentences':len(c)-len(b)} for i,(c,b) in enumerate(zip(full,old))]}
save('ceiling_full_coverage.json',coverage)
save('ceiling_full_progress.json',{'stage':'encoding_tails','updated_at':stamp(),**coverage})
print('Coverage:',json.dumps(coverage),flush=True)
if tails:
    enc=TextToEmbeddingModelPipeline(encoder='text_sonar_basic_encoder',tokenizer='text_sonar_basic_encoder',device=torch.device('cuda'))
    tail_emb=enc.predict(tails,source_lang='eng_Latn',batch_size=256).float().cpu().numpy()
    assert tail_emb.shape==(len(tails),1024)
    del enc
    gc.collect()
    torch.cuda.empty_cache()
else:
    tail_emb=np.empty((0,1024),dtype=np.float32)
embeddings=[]
offset=tail_offset=0
for c,b in zip(full,old):
    n_tail=len(c)-len(b)
    embeddings.append(np.concatenate([old_emb[offset:offset+len(b)],tail_emb[tail_offset:tail_offset+n_tail]]))
    offset+=len(b)
    tail_offset+=n_tail
np.save('data/test_sonar_full_emb.npy',np.concatenate(embeddings))
Path('data/test_sonar_full_sents.json').write_text(json.dumps([s for c in full for s in c],ensure_ascii=False))
print('Tail encoding complete; starting full decoding.',flush=True)
dec=EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder',tokenizer='text_sonar_basic_encoder',device=torch.device('cuda'))
sc=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
def score(refs, hyps, rouge_values):
    return {'n_sent':len(refs),'BLEU4':round(sacrebleu.corpus_bleu(hyps,[refs]).score,2),
            'ROUGE_L':round(100*float(np.mean(rouge_values)),2),
            'chrF':round(sacrebleu.corpus_chrf(hyps,[refs]).score,2),**entity_recall(refs,hyps)}
refs_all=[]
hyps_all=[]
rouge_all=[]
block_mask=[]
per_conv=[]
spk=lambda s:s.split(': ',1)[0] if ': ' in s[:40] else '<无前缀>'
normalize=lambda s:s.strip().strip('"\'“”‘’').strip().casefold()
all_speakers={spk(s) for c in full for s in c}
errors=[]
counts=Counter()
with (results/'predictions_ceiling_full.jsonl').open('w') as predictions, (results/'samples_ceiling_full.txt').open('w') as samples:
    for ci,(refs,emb,b) in enumerate(zip(full,embeddings,old)):
        save('ceiling_full_progress.json',{'stage':'decoding','updated_at':stamp(),
              'done_conversations':ci,'total_conversations':len(full),'done_sentences':len(refs_all),'total_sentences':coverage['n_sent']})
        hyps=list(dec.predict(torch.tensor(emb,dtype=torch.float32),target_lang='eng_Latn',batch_size=a.bs,max_seq_len=256))
        assert len(hyps)==len(refs)
        rouge=[sc.score(r,h)['rougeL'].fmeasure for r,h in zip(refs,hyps)]
        conv_metrics={'conversation_index':ci,**score(refs,hyps,rouge)}
        per_conv.append(conv_metrics)
        speakers={spk(s) for s in refs}
        samples.write(f'===== conversation {ci} =====\n')
        for si,(ref,hyp) in enumerate(zip(refs,hyps)):
            record={'global_index':len(refs_all)+si,'conversation_index':ci,'sentence_index':si,
                    'in_existing_block_test':si<len(b),'ref':ref,'hyp':hyp}
            predictions.write(json.dumps(record,ensure_ascii=False)+'\n')
            samples.write(f'REF: {ref}\nHYP: {hyp}\n\n')
            rsp,hsp=spk(ref),spk(hyp)
            if rsp!=hsp:
                if hsp=='<无前缀>':
                    kind='missing_or_nonstandard_prefix'
                elif normalize(rsp)==normalize(hsp):
                    kind='prefix_format'
                elif hsp in speakers:
                    kind='another_speaker_same_conversation'
                elif hsp in all_speakers:
                    kind='speaker_from_another_conversation'
                else:
                    kind='other_prefix_needs_review'
                counts[kind]+=1
                errors.append({**record,'ref_speaker':rsp,'hyp_speaker':hsp,'kind':kind})
        predictions.flush()
        samples.flush()
        refs_all.extend(refs)
        hyps_all.extend(hyps)
        rouge_all.extend(rouge)
        block_mask.extend([si<len(b) for si in range(len(refs))])
        save('dec_ceiling_full_by_conversation.json',per_conv)
        print(f'Completed conversation {ci+1}/{len(full)}: {len(refs_all)}/{coverage["n_sent"]} sentences; '+json.dumps(conv_metrics),flush=True)
assert len(refs_all)==coverage['n_sent']==len(hyps_all)
metrics={'name':'ceiling_full','scope':'all 10 LoCoMo conversations, including incomplete-block tails',
         'n_conversations':len(full),**score(refs_all,hyps_all,rouge_all)}
block_refs=[r for r,keep in zip(refs_all,block_mask) if keep]
block_hyps=[h for h,keep in zip(hyps_all,block_mask) if keep]
block_rouge=[r for r,keep in zip(rouge_all,block_mask) if keep]
block_metrics={'name':'ceiling_allblocks','scope':'all 846 blocks from the original preprocessing',**score(block_refs,block_hyps,block_rouge)}
save('dec_ceiling_allblocks.json',block_metrics)
save('speaker_prefix_check_full.json',{'n_pairs':len(refs_all),'n_mismatches':len(errors),
     'counts':dict(counts),'other_prefix_counts':dict(Counter(e['hyp_speaker'] for e in errors if e['kind']=='other_prefix_needs_review')),
     'mismatches':errors})
save('dec_ceiling_full.json',metrics)
save('ceiling_full_progress.json',{'stage':'completed','updated_at':stamp(),'done_conversations':len(full),
     'total_conversations':len(full),'done_sentences':len(refs_all),'total_sentences':len(refs_all)})
print('FULL_METRICS',json.dumps(metrics),flush=True)
print('ALL_BLOCK_METRICS',json.dumps(block_metrics),flush=True)
print('SPEAKER_ERROR_COUNTS',json.dumps(dict(counts)),flush=True)

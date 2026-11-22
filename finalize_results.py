"""Audit decoded artifacts and write the required data-only experiment summary."""
import datetime
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from entity import entities, entity_recall, split_speaker

root = Path(__file__).resolve().parent
R = root / 'results'
assert (root / 'logs/stage5_3_4.exit').read_text().strip() == '0'
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
sents = json.loads((root / 'data/test_sents.json').read_text())
meta = np.load(root / 'data/test_meta.npy')
vector = json.loads((R / 'stage5_1_2_validation.json').read_text())
conv_speakers = {}
for block, (conv, _) in enumerate(meta):
    conv_speakers.setdefault(int(conv), set()).update(split_speaker(sents[block * 16 + i])[0] for i in range(16))

names = ['ceiling_allblocks', 'none_sim', 'main_cold', 'tree_main_m1', 'tree_main_m4']
metrics, predictions, checks = {}, {}, {}
for name in names:
    print('Auditing', name, flush=True)
    d = json.loads((R / f'dec_{name}.json').read_text())
    rows = [json.loads(line) for line in (R / f'predictions_{name}.jsonl').read_text().splitlines()]
    expected = vector['test_sentence_pool'] if name in names[:3] else vector['tree_groups'] * 16
    assert d['n_sent'] == len(rows) == expected, (name, len(rows), expected)
    assert len({row['sentence_index'] for row in rows}) == expected
    for row in rows:
        assert row['sentence_index'] == row['block'] * 16 + row['position']
        assert row['ref'] == sents[row['sentence_index']]
        assert 0 <= row['position'] < 16
        assert isinstance(row['hyp'], str)
    refs = [row['ref'] for row in rows]
    hyps = [row['hyp'] for row in rows]
    speakers = [conv_speakers[int(meta[row['block'], 0])] for row in rows]
    recomputed = {
        'BLEU4': round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        'ROUGE_L': round(100 * float(np.mean([scorer.score(r, h)['rougeL'].fmeasure for r, h in zip(refs, hyps)])), 2),
        'chrF': round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
        **entity_recall(refs, hyps, speakers),
    }
    for key, value in recomputed.items():
        assert math.isfinite(value)
        assert d[key] == value, (name, key, d[key], value)
    assert abs(sum(d[f'speaker_{k}'] for k in ['acc', 'format', 'missing', 'swap', 'other']) - 1) <= 0.0003
    assert (R / f'samples_{name}.txt').stat().st_size > 0
    metrics[name], predictions[name] = d, rows
    checks[name] = {'n_sent': expected, 'sentence_alignment_passed': True, 'all_text_and_entity_metrics_recomputed': True}

for name in names[1:3]:
    assert [r['sentence_index'] for r in predictions[name]] == [r['sentence_index'] for r in predictions['ceiling_allblocks']]

entity_overlap = {}
for kind, offset in [('numbers', 0), ('proper_names', 1)]:
    counts = {'reference_occurrences': 0, 'ceiling_lost': 0, 'main_lost': 0, 'lost_in_both': 0, 'main_lost_ceiling_kept': 0, 'main_kept_ceiling_lost': 0}
    for c, m in zip(predictions['ceiling_allblocks'], predictions['main_cold']):
        assert c['sentence_index'] == m['sentence_index'] and c['ref'] == m['ref']
        ct, mt = split_speaker(c['hyp'])[1], split_speaker(m['hyp'])[1]
        for item in entities(c['ref'])[offset]:
            cl, ml = item not in ct, item not in mt
            counts['reference_occurrences'] += 1
            counts['ceiling_lost'] += cl
            counts['main_lost'] += ml
            counts['lost_in_both'] += cl and ml
            counts['main_lost_ceiling_kept'] += ml and not cl
            counts['main_kept_ceiling_lost'] += cl and not ml
    counts['fraction_of_main_losses_already_lost_in_ceiling'] = counts['lost_in_both'] / counts['main_lost'] if counts['main_lost'] else None
    entity_overlap[kind] = counts
keys = ['reference_occurrences', 'ceiling_lost', 'main_lost', 'lost_in_both', 'main_lost_ceiling_kept', 'main_kept_ceiling_lost']
combined = {key: sum(entity_overlap[k][key] for k in ['numbers', 'proper_names']) for key in keys}
combined['fraction_of_main_losses_already_lost_in_ceiling'] = combined['lost_in_both'] / combined['main_lost'] if combined['main_lost'] else None
entity_overlap['combined'] = combined
entity_overlap['definition'] = 'Reference entity occurrences; original entities() extraction and case-sensitive substring hit rule, same sentence index. Overlap is measured against the uncompressed encode/decode ceiling, not a causal attribution to the encoder alone.'

examples = []
for name in ['main_cold', 'tree_main_m1']:
    available = {(r['ref'], r['hyp']): r for r in predictions[name]}
    pairs = []
    ref = None
    for line in (R / f'samples_{name}.txt').read_text().splitlines():
        if line.startswith('REF: '):
            ref = line[5:]
        elif line.startswith('HYP: ') and ref is not None:
            pair = (ref, line[5:])
            if pair in available:
                r = available[pair].copy()
                r['rougeL'] = scorer.score(*pair)['rougeL'].fmeasure
                r['missing_entities'] = [item for values in entities(r['ref']) for item in values if item not in split_speaker(r['hyp'])[1]]
                pairs.append(r)
            ref = None
    assert len(pairs) >= 2
    substantial = [r for r in pairs if len(split_speaker(r['ref'])[1].split()) >= 6] or pairs
    good = max(substantial, key=lambda r: (r['rougeL'], -len(r['missing_entities']), len(r['ref'])))
    failed_pool = [r for r in substantial if r['missing_entities']] or substantial
    bad = min(failed_pool, key=lambda r: r['rougeL'])
    assert (good['ref'], good['hyp']) != (bad['ref'], bad['hyp'])
    examples += [dict(good, setting=name, selection='good'), dict(bad, setting=name, selection='failure')]

backup = Path((root / 'logs/ceiling_entity_alignment_backup.txt').read_text())
old = json.loads((backup / 'dec_ceiling_allblocks.json').read_text())
differences = {k: {'previous': old.get(k), 'current': v} for k, v in metrics['ceiling_allblocks'].items() if old.get(k) != v}
report = {
    'passed': True,
    'timestamp': datetime.datetime.now().astimezone().isoformat(),
    'scope': {'test_blocks': vector['test_blocks'], 'single_sentences': vector['test_sentence_pool'], 'tree_groups': vector['tree_groups'], 'tree_leaf_sentences': vector['tree_groups'] * 16},
    'decode_validation': checks,
    'e0_reference': 'dec_ceiling_allblocks.json',
    'ceiling_changes_from_previous_run': differences,
    'text_metrics': metrics,
    'e1': {**vector['e1'], 'none_BLEU4': metrics['none_sim']['BLEU4'], 'main_BLEU4': metrics['main_cold']['BLEU4'], 'BLEU4_change': round(metrics['main_cold']['BLEU4'] - metrics['none_sim']['BLEU4'], 2)},
    'e2': vector['e2'],
    'e3': vector['e3'],
    'chain_main_m1_top1': vector['chain_main_m1_top1'],
    'chain_main_m4_top1': vector['chain_main_m4_top1'],
    'chain_main_m4_minus_m1_percentage_points': vector['chain_main_m4_minus_m1_percentage_points'],
    'entity_loss_overlap': entity_overlap,
    'examples': examples,
    'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ['decode_eval.py', 'entity.py', 'plot.py', 'common.py']},
    'fixed_pooling_executed': False,
}
(R / 'final_analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
# All displayed experiment numbers below come from saved result JSON.
report = json.loads((R / 'final_analysis.json').read_text())
metrics = report['text_metrics']

def pct(value):
    return str((Decimal(str(value)) * 100).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)) + '%'

def pp(value):
    return str(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

ceiling, none, main = [metrics[k] for k in ['ceiling_allblocks', 'none_sim', 'main_cold']]
lines = ['## 结果摘要', '',
    f"比较口径：E0 与 E1 使用同一批 {report['scope']['test_blocks']} 块、{report['scope']['single_sentences']} 句；E3 为 {report['scope']['tree_groups']} 组，单个叶子的文本评测为 {report['scope']['tree_leaf_sentences']} 句。历史的3200句 E0 与未截断到完整块的全句 E0 不用于本轮主比较。", '',
    f"1. **E0**：BLEU-4 {ceiling['BLEU4']}，ROUGE-L {ceiling['ROUGE_L']}，数字召回 {pct(ceiling['num_recall'])}，专名召回 {pct(ceiling['name_recall'])}。", '',
    f"2. **E1**：Top-1 从 {pct(report['e1']['none_top1'])} 到 {pct(report['e1']['main_top1'])}，提高 {pp(report['e1']['top1_gain_percentage_points'])} 个百分点；BLEU-4 从 {none['BLEU4']} 到 {main['BLEU4']}，变化 {report['e1']['BLEU4_change']:+.2f}。cos 从 {report['e1']['none_cos']} 到 {report['e1']['main_cos']}，变化 {report['e1']['cos_change']:+.4f}。", '',
    '3. **E2**：8步 Top-1 ' + ('单调上升' if report['e2']['top1_monotonic_nondecreasing'] else '不单调') + '：' + ' → '.join(pct(x) for x in report['e2']['top1']) + '。', '',
    '4. **E3**：四个叶子的平均 Top-1：', '', '| 结构 | 不还原 | 主方法 | 主方法增益（百分点） |', '|---|---:|---:|---:|']
for name, label in [('single', '单层'), ('balanced', '平衡树'), ('chain', '链')]:
    d = report['e3'][name]
    lines.append(f"| {label} | {pct(d['none_top1_mean'])} | {pct(d['main_top1_mean'])} | {pp(d['top1_gain_percentage_points'])} |")
lines += ['', f"主方法链结构 m1 的 Top-1 为 {pct(report['chain_main_m1_top1'])}，m4 为 {pct(report['chain_main_m4_top1'])}，相差 {pp(report['chain_main_m4_minus_m1_percentage_points'])} 个百分点。", '',
    '5. **实体召回与丢失交集**：', '', '| 设置 | 数字召回 | 专名召回 |', '|---|---:|---:|']
for name, label in [('ceiling_allblocks', 'E0 上限'), ('none_sim', '不还原'), ('main_cold', '主方法')]:
    d = metrics[name]
    lines.append(f"| {label} | {pct(d['num_recall'])} | {pct(d['name_recall'])} |")
lines += ['', '按原实体抽取和命中规则，逐句、逐参考实体出现位置统计：', '', '| 实体 | 主方法丢失 | 其中 E0 也丢失 | 占主方法丢失的比例 |', '|---|---:|---:|---:|']
for name, label in [('numbers', '数字'), ('proper_names', '专名'), ('combined', '合计')]:
    d = report['entity_loss_overlap'][name]
    fraction = d['fraction_of_main_losses_already_lost_in_ceiling']
    lines.append(f"| {label} | {d['main_lost']} | {d['lost_in_both']} | {pct(fraction) if fraction is not None else '无丢失'} |")
lines += ['', '这里统计的是与不压缩编解码上限的丢失交集，不能单独归因于编码器，也不以汇总召回率之比冒充实体交集。', '', '## 原样示例', '']
for example in report['examples']:
    label = '较好' if example['selection'] == 'good' else '失败'
    lines += [f"### {example['setting']}：{label}（block {example['block']}，位置 {example['position']}）", '', '```text', f"REF: {example['ref']}", f"HYP: {example['hyp']}", '```', '']
    if example['missing_entities']:
        lines += ['未召回的参考实体：' + '、'.join(example['missing_entities']) + '。', '']
summary = '\n'.join(lines) + '\n'
(R / 'final_review.md').write_text(summary)
with (R / 'summary.md').open('a') as f:
    f.write('\n' + summary)
with (R / 'RUNLOG.md').open('a') as f:
    f.write(f"\n### [5.3/5.4 最终验收] {report['timestamp']}\n\n命令：`python finalize_results.py`。\n\n结果：五组逐句预测均已与原句索引匹配；BLEU、ROUGE、chrF、实体和说话人统计全部从保存的 REF/HYP 重新核算，与各自 JSON 一致。所需20条E0样例沿用前文；本轮对比使用同846块E0，新增完整预测用于实际实体交集。E0重跑与原结果的变化见 final_analysis.json。\n\n验收：解码输出、结果摘要及示例通过；两张图还需视觉检查。改动：输出与汇总修复已记录，方法、模型及 common.py 未改。\n\n" + summary)
print(json.dumps({k: report[k] for k in ['passed', 'scope', 'ceiling_changes_from_previous_run', 'e1', 'entity_loss_overlap', 'examples']}, ensure_ascii=False, indent=2), flush=True)

"""对话数据 → 句子序列。
训练/验证：MSC（Multi-Session Chat，HF: nayohan/multi_session_chat），同一对话的所有 session 按顺序拼接。
测试：LoCoMo（data/raw/locomo10.json，来自 github.com/snap-research/locomo）。
每句格式："Name: sentence"。MSC 没有人名，给每个对话的两位说话人随机分配英文名（与 LoCoMo 的格式对齐）。
少于 MIN_WORDS 个词的句子并入同一轮里相邻的句子。
输出 data/{train,valid,test}_convs.jsonl，每行一个对话的句子列表（截断为 16 的整数倍）。
用法：python prep_data.py            # 处理全部
     python prep_data.py --inspect  # 只打印两个数据集的原始样例，检查字段名"""
import argparse, json, os, random, re
from collections import defaultdict

L, MIN_WORDS = 16, 4
NAMES = ("James Mary John Linda Robert Susan Michael Karen David Lisa William Nancy Richard Betty Joseph Sandra "
         "Thomas Ashley Charles Emily Daniel Donna Matthew Michelle Anthony Carol Mark Amanda Steven Melissa Paul "
         "Deborah Andrew Laura Joshua Rebecca Kevin Sharon Brian Cynthia George Kathleen Edward Amy Ronald Angela "
         "Timothy Anna Jason Brenda Jeffrey Pamela Ryan Emma Jacob Nicole Gary Helen Nicholas Samantha Eric Katherine "
         "Jonathan Christine Stephen Rachel Larry Carolyn Justin Janet Scott Maria Brandon Olivia Benjamin Heather "
         "Samuel Diane Frank Julie Gregory Joyce Raymond Victoria Alexander Kelly Patrick Christina Jack Lauren "
         "Dennis Joan Jerry Evelyn Tyler Judith Aaron Megan Jose Andrea Adam Cheryl Nathan Hannah Henry Jacqueline "
         "Zachary Martha Douglas Gloria Peter Teresa Kyle Ann Noah Sara Ethan Madison Jeremy Frances Walter Kathryn").split()
SPLIT_RE = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=[A-Z0-9"\'(\[])')


def split_turn(text):
    text = re.sub(r'\s+', ' ', text or '').strip()
    if not text:
        return []
    parts = [p.strip() for p in SPLIT_RE.split(text) if p.strip()]
    out = []
    for p in parts:                                   # 短句并入前一句
        if out and len(p.split()) < MIN_WORDS:
            out[-1] = out[-1] + ' ' + p
        else:
            out.append(p)
    if len(out) > 1 and len(out[0].split()) < MIN_WORDS:   # 开头的短句并入后一句
        out[1] = out[0] + ' ' + out[1]
        out = out[1:]
    return out


def conv_to_sents(turns):
    """turns: [(speaker_name, text)] → ['Name: sentence', ...]"""
    return [f'{spk}: {s}' for spk, text in turns for s in split_turn(text)]


def load_msc(inspect=False):
    from datasets import load_dataset
    ds = load_dataset("nayohan/multi_session_chat")
    if inspect:
        for sp in ds:
            print('== MSC split', sp, len(ds[sp]), 'rows; columns', ds[sp].column_names)
            print({k: (v[:3] if isinstance(v, list) else v) for k, v in ds[sp][0].items()})
        return None
    sessions = defaultdict(list)                      # (split, dialog_id) -> [(session_id, row)]
    for sp in ds:
        cols = ds[sp].column_names
        did = 'dialoug_id' if 'dialoug_id' in cols else 'dialogue_id'
        for row in ds[sp]:
            sessions[(sp, row[did])].append((row.get('session_id', 0), row))
    rng = random.Random(0)
    convs = []
    for key in sorted(sessions, key=str):
        a, b = rng.sample(NAMES, 2)
        turns = []
        for _, row in sorted(sessions[key], key=lambda x: x[0]):
            utts = row['dialogue']
            spks = row.get('speaker') or [None] * len(utts)
            first = spks[0] if spks else None
            for i, (u, s) in enumerate(zip(utts, spks)):
                is_a = (s == first) if s is not None else (i % 2 == 0)
                turns.append((a if is_a else b, u))
        convs.append(conv_to_sents(turns))
    return convs


def load_locomo(path='data/raw/locomo10.json', inspect=False):
    data = json.load(open(path))
    if inspect:
        c = data[0]['conversation']
        print('== LoCoMo', len(data), 'samples; conversation keys', list(c.keys())[:8])
        print(c['session_1'][:2])
        return None
    convs = []
    for sample in data:
        c = sample['conversation']
        turns, n = [], 1
        while f'session_{n}' in c:
            for t in c[f'session_{n}']:
                turns.append((t['speaker'], t.get('text', '')))      # 只用 text，不用图片描述
            n += 1
        convs.append(conv_to_sents(turns))
    return convs


def dump(name, convs, min_blocks):
    kept = [c[:(len(c) // L) * L] for c in convs if len(c) >= min_blocks * L]
    with open(f'data/{name}_convs.jsonl', 'w') as f:
        for c in kept:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    nb = sum(len(c) // L for c in kept)
    print(f'{name}: {len(kept)} convs (dropped {len(convs) - len(kept)} short), {nb} blocks, {nb * L} sents')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--n_valid', type=int, default=300)
    a = p.parse_args()
    os.makedirs('data', exist_ok=True)
    if a.inspect:
        load_msc(True)
        load_locomo(inspect=True)
        raise SystemExit
    test = load_locomo()
    msc = load_msc()
    random.Random(1).shuffle(msc)
    dump('test', test, 4)
    dump('valid', msc[:a.n_valid], 1)
    dump('train', msc[a.n_valid:], 1)
    ex = test[0][:6]
    print('LoCoMo 样例：', *ex, sep='\n  ')
    print('MSC 样例：', *msc[0][:6], sep='\n  ')

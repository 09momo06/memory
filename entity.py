"""英文轻量实体抽取 + 召回率，用来量化"事实错误"。
句子格式为 "Name: text"，抽取前先去掉说话人前缀（说话人单独统计，不算进专名）。
数字：阿拉伯数字（含 3.5 / 10:30 / 2,000）。专名：句中（非句首）大写开头的词，排除 I / I'm 等。

说话人指标（每句恰好落入一类，四类相加 = 1）：
  speaker_acc      严格一致（与旧版定义相同，保留以便对比）
  speaker_format   只差大小写 / 引号 / 首尾空白
  speaker_missing  解码结果没有可解析的前缀（': ' 不在前 40 个字符内，或前缀超过 3 个词）
  speaker_swap     前缀是同一段对话里另一位说话人 —— 对记忆系统真正有害的错误
  speaker_other    其余（拼写变化如 John→Jon、缩写、其他名字等）
没有提供 conv_speakers 时，无法判断换人，swap 记为 0，相应句子归入 other。"""
import re

NUM = re.compile(r'\d+(?:[.,:]\d+)*')
CAP = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-zA-Z\-]+")
STOP = {"I", "I'm", "I've", "I'll", "I'd", "OK", "Okay"}


def split_speaker(s):
    if ': ' in s[:40]:
        spk, text = s.split(': ', 1)
        return spk, text
    return '', s


def parse_prefix(s):
    """可解析的说话人前缀（最多 3 个词），否则返回 None。"""
    spk, _ = split_speaker(s)
    spk = spk.strip()
    if not spk or len(spk.split()) > 3:
        return None
    return spk


def norm_name(s):
    return re.sub(r'["\'`“”‘’]', '', s).strip().lower()


def entities(s):
    _, text = split_speaker(s)
    nums = NUM.findall(text)
    names = [w for w in CAP.findall(text) if w not in STOP]
    return nums, names


def speaker_stats(refs, hyps, conv_speakers=None):
    """conv_speakers: 与 refs 等长的列表，每项是该句所在对话的全部说话人名字集合。"""
    c = {'acc': 0, 'format': 0, 'missing': 0, 'swap': 0, 'other': 0}
    for i, (r, h) in enumerate(zip(refs, hyps)):
        rs, hs = split_speaker(r)[0], parse_prefix(h)
        if split_speaker(h)[0] == rs:
            c['acc'] += 1
        elif hs is None:
            c['missing'] += 1
        elif norm_name(hs) == norm_name(rs):
            c['format'] += 1
        elif conv_speakers is not None and norm_name(hs) in {norm_name(x) for x in conv_speakers[i]} - {norm_name(rs)}:
            c['swap'] += 1
        else:
            c['other'] += 1
    n = max(len(refs), 1)
    return {f'speaker_{k}': round(v / n, 4) for k, v in c.items()} | {'n_speaker_swap': c['swap']}


def entity_recall(refs, hyps, conv_speakers=None):
    nh = nt = mh = mt = 0
    for r, h in zip(refs, hyps):
        nums, names = entities(r)
        _, htext = split_speaker(h)
        nt += len(nums); nh += sum(x in htext for x in nums)
        mt += len(names); mh += sum(x in htext for x in names)
    return {'num_recall': round(nh / max(nt, 1), 4), 'n_num': nt,
            'name_recall': round(mh / max(mt, 1), 4), 'n_name': mt,
            **speaker_stats(refs, hyps, conv_speakers)}


if __name__ == '__main__':
    s = "Caroline: I went to the LGBTQ support group on May 7 with Melanie, it started at 10:30."
    print(entities(s))
    refs = ["Caroline: Hey Mel! Good to see you!", "Nate: Sorry to hear that, Joanna.",
            "John: I want to ask you, what motivates you?", "Maria: Oh John, that sounds tough.",
            "Gina: Sorry about your job Jon.", "Tim: Last night I joined a forum."]
    hyps = ["Hey Mel! Good to see you!", "Joanna: Sorry to hear that, Joanna.",
            "Jon: I want to ask you, what motivates you?", "maria: Oh John, that sounds hard.",
            "Gina: Sorry about your job Jon.", "Hey there, buddy and friend: Last night I joined a forum."]
    spk = [{'Caroline', 'Melanie'}, {'Nate', 'Joanna'}, {'John', 'James'}, {'Maria', 'John'},
           {'Gina', 'Jon'}, {'Tim', 'John'}]
    out = entity_recall(refs, hyps, spk)
    print(out)
    expect = {'speaker_acc': 1, 'speaker_format': 1, 'speaker_missing': 2, 'speaker_swap': 1, 'speaker_other': 1}
    for k, v in expect.items():
        assert round(out[k] * 6) == v, (k, out[k])
    assert out['n_speaker_swap'] == 1
    print('entity ok')

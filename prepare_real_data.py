# -*- coding: utf-8 -*-
"""从 HuggingFace 镜像 dhbloo/TuSimple 抢救真实 TuSimple 数据。

产出（供实验一 Ultra-Fast 直接使用）：
  <REAL>/train_set/clips/<session>/<clip>/<frame>.jpg
  <REAL>/train_set/label_data_0313.json | 0531 | 0601      （convert_tusimple.py 写死这三个文件名）
  <REAL>/test_set/clips/.../*.jpg
  <REAL>/test_set/test_label.json

用法：
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python prepare_real_data.py
"""
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

REAL = os.environ.get('REAL_DATA', '/home/ma-user/work/ch3_demo/data/TusimpleReal')
HF_API = 'https://huggingface.co/api/datasets/dhbloo/TuSimple/tree/main/'
HF_RES = 'https://huggingface.co/datasets/dhbloo/TuSimple/resolve/main/'
SESSIONS = ['0313-1', '0313-2', '0530', '0531', '0601']

# convert_tusimple.py 硬编码读取这三个文件
TRAIN_LABELS = ['label_data_0313.json', 'label_data_0531.json', 'label_data_0601.json']
# 额外标注（用于扩充可下载图片的交集）
EXTRA_LABELS = ['seg_label/train_val.json', 'seg_label/test.json', 'test_label.json', 'test_baseline.json']

os.makedirs(os.path.join(REAL, 'train_set', 'clips'), exist_ok=True)
os.makedirs(os.path.join(REAL, 'test_set', 'clips'), exist_ok=True)


def curl(url, dest=None, timeout=120):
    cmd = ['curl', '-sL', '-m', str(timeout), url]
    if dest:
        cmd += ['-o', dest]
        subprocess.run(cmd, capture_output=True)
        return os.path.exists(dest) and os.path.getsize(dest) > 1000
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.stdout


# ---------- 1) 下载标注文件 ----------
local = {}
for name in TRAIN_LABELS + ['test_label.json']:
    dest = os.path.join(REAL, os.path.basename(name))
    ok = curl(HF_RES + name, dest)
    print('[label] %-24s %s' % (name, 'OK' if ok else 'FAIL'))
    if ok:
        local[name] = dest

for name in EXTRA_LABELS:
    dest = os.path.join(REAL, name.replace('/', '_'))
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        local[name] = dest
        continue
    if curl(HF_RES + name, dest):
        local[name] = dest
        print('[label] %-24s OK' % name)


def read_jsonl(path):
    out = []
    try:
        for line in open(path, encoding='utf-8').read().strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    except Exception:
        pass
    return out


# ---------- 2) 枚举 HF 上真实存在的 clip 目录 ----------
existing = set()
for s in SESSIONS:
    raw = curl(HF_API + 'clips/%s?limit=1000' % s, timeout=90)
    try:
        data = json.loads(raw)
    except Exception:
        data = None
    if isinstance(data, list):
        for x in data:
            if x.get('type') == 'directory':
                existing.add((s, x['path'].split('/')[-1]))
print('[HF] 现存 clip 目录 = %d' % len(existing))


def matched(rec):
    rf = rec.get('raw_file') or ''
    p = rf.split('/')
    return len(p) == 4 and p[0] == 'clips' and (p[1], p[2]) in existing


# ---------- 3) 收集有图可用的标注 ----------
train_entries, test_entries = {}, {}
for name in TRAIN_LABELS:
    p = local.get(name)
    if not p:
        continue
    for r in read_jsonl(p):
        if matched(r):
            train_entries[r['raw_file']] = r
for name in EXTRA_LABELS:
    p = local.get(name)
    if not p:
        continue
    for r in read_jsonl(p):
        if matched(r):
            train_entries.setdefault(r['raw_file'], r)
p = local.get('test_label.json')
if p:
    for r in read_jsonl(p):
        if matched(r):
            test_entries[r['raw_file']] = r

print('[match] 训练可用 = %d   测试可用 = %d' % (len(train_entries), len(test_entries)))
print('[match] 帧号分布 =', Counter(k.split('/')[-1] for k in train_entries).most_common(3))


# ---------- 4) 并发下载图片 ----------
def dl(args):
    root, rf = args
    dest = os.path.join(root, rf)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return 1
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if curl(HF_RES + rf, dest, timeout=120):
        return 1
    if os.path.exists(dest):
        try:
            os.remove(dest)
        except OSError:
            pass
    return 0


def download_all(root, entries, tag):
    ok = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for n in ex.map(dl, [(root, rf) for rf in sorted(entries)]):
            ok += n
    print('[%s] 下载成功 %d / %d' % (tag, ok, len(entries)))
    return ok


download_all(os.path.join(REAL, 'train_set'), train_entries, 'train')
download_all(os.path.join(REAL, 'test_set'), test_entries, 'test')


# ---------- 5) 只保留图确实在盘上的标注，按原文件名写回 ----------
def write_kept(root, entries, out_name):
    kept = [r for r in entries.values()
            if os.path.exists(os.path.join(root, r['raw_file']))
            and os.path.getsize(os.path.join(root, r['raw_file'])) > 1000]
    with open(os.path.join(root, out_name), 'w') as f:
        for r in kept:
            f.write(json.dumps(r) + '\n')
    print('[write] %-22s %d 条' % (out_name, len(kept)))
    return len(kept)


# 训练集：按 raw_file 所属 session 分流到三个标注文件（convert_tusimple.py 会读全部三个）
def session_of(rf):
    p = rf.split('/')
    return p[1] if len(p) > 1 else '0313-1'


sess2file = {'0313-1': 'label_data_0313.json', '0313-2': 'label_data_0313.json',
             '0530': 'label_data_0531.json', '0531': 'label_data_0531.json',
             '0601': 'label_data_0601.json'}
train_root = os.path.join(REAL, 'train_set')
kept_train = [r for r in train_entries.values()
              if os.path.exists(os.path.join(train_root, r['raw_file']))
              and os.path.getsize(os.path.join(train_root, r['raw_file'])) > 1000]
buckets = {n: [] for n in TRAIN_LABELS}
for r in kept_train:
    buckets[sess2file.get(session_of(r['raw_file']), 'label_data_0313.json')].append(r)
for name, rows in buckets.items():
    with open(os.path.join(train_root, name), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print('[write] %-22s %d 条' % (name, len(rows)))

write_kept(os.path.join(REAL, 'test_set'), test_entries, 'test_label.json')

n_train = sum(1 for _ in open(os.path.join(train_root, 'label_data_0313.json'))) + \
          sum(1 for _ in open(os.path.join(train_root, 'label_data_0531.json'))) + \
          sum(1 for _ in open(os.path.join(train_root, 'label_data_0601.json')))
print('===== 真实数据准备完成：训练 %d 张 / 测试 %d 张 =====' % (n_train, len(test_entries)))
print('目录:', REAL)

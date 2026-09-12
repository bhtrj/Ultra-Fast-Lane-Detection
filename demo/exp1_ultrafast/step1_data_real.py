# -*- coding: utf-8 -*-
"""实验一 第 1 步（真实数据版）：从抢救到的真实 TuSimple 中抽小样本，生成可训练的数据集。

数据来源：HuggingFace 镜像 dhbloo/TuSimple（官方 S3 已 404，OpenDataLab 需 AK/SK）。
先用 prepare_real_data.py 把可用图片抓到 data/TusimpleReal（默认 644 张，140MB），
这里再从中抽 N_TRAIN / N_TEST 张，输出到 data/Tusimple（与合成版同路径，后续步骤不变）。

用法：
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python step1_真实数据.py                 # 默认 60 训练 / 20 测试
    python step1_真实数据.py --n_train 120 --n_test 40
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('CH3_DEMO', os.path.abspath(os.path.join(HERE, '..', '..')))
UF = os.path.join(BASE, 'code_uf')
REAL = os.path.join(BASE, 'data', 'tusimple_full')      # 用 --more_data 拉取全量时的存放位置
DST = os.path.join(BASE, 'data', 'tusimple_sample')     # 仓库自带的 sample 数据

TRAIN_LABELS = ['label_data_0313.json', 'label_data_0531.json', 'label_data_0601.json']
SESS2FILE = {'0313-1': 'label_data_0313.json', '0313-2': 'label_data_0313.json',
             '0530': 'label_data_0531.json', '0531': 'label_data_0531.json',
             '0601': 'label_data_0601.json'}


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding='utf-8').read().strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def has_img(root, rf):
    p = os.path.join(root, rf)
    return os.path.exists(p) and os.path.getsize(p) > 1000


def build(split, n, src_root, dst_root, label_files, seed=1234):
    """src_root -> dst_root，抽 n 条（图复制 + 标注重写）"""
    pool = []
    for lf in label_files:
        for r in read_jsonl(os.path.join(src_root, lf)):
            rf = r.get('raw_file')
            if rf and has_img(src_root, rf):
                pool.append(r)
    if not pool:
        print('[ERR] %s 没有可用样本：%s' % (split, src_root))
        sys.exit(1)
    random.seed(seed)
    random.shuffle(pool)
    pick = pool[:n] if n < len(pool) else pool

    os.makedirs(os.path.join(dst_root, 'clips'), exist_ok=True)
    for r in pick:
        rf = r['raw_file']
        src, dst = os.path.join(src_root, rf), os.path.join(dst_root, rf)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    if split == 'train':
        buckets = {n_: [] for n_ in TRAIN_LABELS}
        for r in pick:
            sess = r['raw_file'].split('/')[1]
            buckets[SESS2FILE.get(sess, 'label_data_0313.json')].append(r)
        for name, rows in buckets.items():
            with open(os.path.join(dst_root, name), 'w') as f:
                for r in rows:
                    f.write(json.dumps(r) + '\n')
    else:
        with open(os.path.join(dst_root, 'test_label.json'), 'w') as f:
            for r in pick:
                f.write(json.dumps(r) + '\n')
    print('[OK] %-9s 池中 %d 张 -> 取 %d 张  -> %s' % (split, len(pool), len(pick), dst_root))
    return len(pick)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_train', type=int, default=60)
    ap.add_argument('--n_test', type=int, default=20)
    ap.add_argument('--real', default=REAL)
    args = ap.parse_args()

    # 仓库自带 sample：若图片已就位，只重跑 convert_tusimple.py（幂等，几秒）
    have = glob.glob(os.path.join(DST, 'train_set', 'clips', '*', '*', '*.jpg'))
    if len(have) >= args.n_train:
        print('[OK] sample 数据已就位（%d 张训练图），跳过下载' % len(have))
    else:
        if not os.path.isdir(os.path.join(args.real, 'train_set')):
            print('[!] 需要更多真实数据：调用 prepare_real_data.py 下载（约 3 分钟，140MB）')
            r = subprocess.call([sys.executable, os.path.join(BASE, 'prepare_real_data.py')])
            if r != 0:
                sys.exit(r)
        for sub in ('train_set', 'test_set'):
            p = os.path.join(DST, sub)
            if os.path.exists(p):
                shutil.rmtree(p)
            os.makedirs(p)
        build('train', args.n_train,
              os.path.join(args.real, 'train_set'), os.path.join(DST, 'train_set'), TRAIN_LABELS)
        build('test', args.n_test,
              os.path.join(args.real, 'test_set'), os.path.join(DST, 'test_set'),
              ['test_label.json'], seed=4321)

    # 生成 train_gt.txt + 分割掩码
    conv = os.path.join(UF, 'convert_tusimple.py')
    cmd = [sys.executable, conv, '--root', os.path.join(DST, 'train_set')]
    print('[RUN]', ' '.join(cmd))
    r = subprocess.call(cmd, cwd=UF)
    if r != 0:
        print('[ERR] convert_tusimple.py 退出码', r)
        sys.exit(r)
    gt = os.path.join(DST, 'train_set', 'train_gt.txt')
    print('[OK] train_gt.txt 行数 =', sum(1 for _ in open(gt)) if os.path.exists(gt) else 0)
    print('===== 实验一 真实数据准备完成 =====')

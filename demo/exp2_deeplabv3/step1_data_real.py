# -*- coding: utf-8 -*-
"""实验二 第 1 步（真实数据版）：从真实 TuSimple 抽小样本，生成 图+掩码 分割数据集。

来源：HuggingFace 镜像 dhbloo/TuSimple。
  图   = clips/<session>/<clip>/20.jpg     （prepare_real_data.py 已下载 438 张）
  掩码 = seg_label/<session>/<clip>/20.png （像素值 0=背景，>=1=车道线编号，下载后原样保存）
产出：data/LaneSegReal/{images,masks,train.txt,val.txt}，默认 80 训练 / 20 验证。

用法：
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python step1_真实数据.py
"""
import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('CH3_DEMO', os.path.abspath(os.path.join(HERE, '..', '..')))
REAL = os.path.join(BASE, 'data', 'tusimple_full')
DST = os.path.join(BASE, 'data', 'laneseg_sample')      # 仓库自带的 sample 数据
HF_RES = 'https://huggingface.co/datasets/dhbloo/TuSimple/resolve/main/'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_train', type=int, default=80)
    ap.add_argument('--n_val', type=int, default=20)
    args = ap.parse_args()

    # 仓库自带 sample：若图+掩码已就位，直接跳过
    import glob as _g
    _have = _g.glob(os.path.join(DST, 'images', '*.jpg'))
    if os.path.exists(os.path.join(DST, 'train.txt')) and len(_have) >= args.n_train:
        print('[OK] sample 数据已就位（%d 张图 + 掩码），跳过下载' % len(_have))
        print('===== 实验二 真实数据准备完成 =====')
        raise SystemExit(0)

    anno = os.path.join(REAL, 'seg_label_train_val.json')
    if not os.path.exists(anno):
        print('[ERR] 找不到', anno, ' —— 先运行 prepare_real_data.py')
        raise SystemExit(1)
    rows = [json.loads(l) for l in open(anno, encoding='utf-8').read().strip().split('\n') if l.strip()]
    print('[OK] seg 标注池:', len(rows), '条')

    for sub in ('images', 'masks'):
        os.makedirs(os.path.join(DST, sub), exist_ok=True)

    img_root = os.path.join(REAL, 'train_set')

    jobs = []          # (name, img_src, mask_url)
    for r in rows:
        rf = r['raw_file']                          # clips/0313-1/10000/20.jpg
        parts = rf.split('/')                       # clips, session, clip, frame
        if len(parts) != 4:
            continue
        img_src = os.path.join(img_root, rf)
        if not (os.path.exists(img_src) and os.path.getsize(img_src) > 1000):
            continue
        mask_url = HF_RES + 'seg_label/%s/%s/%s' % (parts[1], parts[2], parts[3].replace('.jpg', '.png'))
        jobs.append((img_src, mask_url))
    print('[OK] 有图可配对的样本:', len(jobs), '张')

    def fetch(job):
        img_src, mask_url = job
        tmp = '/tmp/_mask_%d.png' % abs(hash(mask_url))
        subprocess.run(['curl', '-sL', '-m', '60', mask_url, '-o', tmp], capture_output=True)
        if not (os.path.exists(tmp) and os.path.getsize(tmp) > 1000):
            return None
        m = cv2.imread(tmp, 0)
        os.remove(tmp)
        if m is None:
            return None
        return img_src, m

    got = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(fetch, jobs[:400]):
            if res:
                got.append(res)
            if len(got) >= args.n_train + args.n_val:
                break
    print('[OK] 图+掩码配对成功:', len(got), '张')

    if len(got) < 10:
        print('[ERR] 配对样本过少，中止')
        raise SystemExit(1)

    lines = []
    for i, (img_src, mask) in enumerate(got):
        name = '%04d' % i
        shutil.copy2(img_src, os.path.join(DST, 'images', name + '.jpg'))
        cv2.imwrite(os.path.join(DST, 'masks', name + '.png'), mask)
        lines.append(name)
    tr, va = lines[:args.n_train], lines[args.n_train:args.n_train + args.n_val]
    open(os.path.join(DST, 'train.txt'), 'w').write('\n'.join(tr) + '\n')
    open(os.path.join(DST, 'val.txt'), 'w').write('\n'.join(va) + '\n')

    m = cv2.imread(os.path.join(DST, 'masks', tr[0] + '.png'), 0)
    print('[OK]', DST)
    print('    train=%d val=%d  掩码尺寸=%s  掩码取值=%s'
          % (len(tr), len(va), m.shape, sorted(set(m.ravel().tolist()))[:8]))
    print('===== 实验二 真实数据准备完成 =====')

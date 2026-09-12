# -*- coding: utf-8 -*-
"""实验一（Ultra-Fast-Lane-Detection）第 2 步：训练

对仓库 train.py 做 2 处演示用补丁（不改动算法），然后调用它训练：
  补丁 A：验证精度达标才存盘的阈值 0.9582 -> 0.0（合成数据达不到，否则一个 ckpt 都不存）
  补丁 B：训练结束额外存一份纯网络权重 uf_lane_final.ckpt，供第 3 步推理加载

用法：
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python step2_训练.py             # 默认 5 个 epoch，约 2 分钟
    python step2_训练.py --epochs 10
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('CH3_DEMO', os.path.abspath(os.path.join(HERE, '..', '..')))
UF = os.path.join(BASE, 'code_uf')
DATA = os.path.join(BASE, 'data', 'tusimple_sample')
OUT = os.path.join(BASE, 'out', 'exp1')

PATCHES = [
    # (必须存在于原文, 替换为)
    ("if acc >= 0.9582:", "if acc >= 0.0:"),
    ("""                save_checkpoint(cb_params.train_network, ckpt_file)""",
     """                save_checkpoint(cb_params.train_network, ckpt_file)
                try:
                    save_checkpoint(self.model.predict_network,
                                    ckpt_file.replace('.ckpt', '_net.ckpt'))
                except Exception as e:
                    print('[DEMO] save net ckpt failed:', e)"""),
    ("""    if dataset == 'CULane':
        if cfg.train_url.startswith('s3://')""",
     """    # === DEMO PATCH: 存一份纯网络权重 ===
    _demo_out = cfg.train_url
    os.makedirs(_demo_out, exist_ok=True)
    _demo_ckpt = os.path.join(_demo_out, 'uf_lane_final.ckpt')
    save_checkpoint(net, _demo_ckpt)
    print('[DEMO] final ckpt saved ->', _demo_ckpt)

    if dataset == 'CULane':
        if cfg.train_url.startswith('s3://')"""),
]


def make_train_demo():
    src = os.path.join(UF, 'train.py')
    txt = open(src, encoding='utf-8').read()
    dst = os.path.join(UF, 'train_demo.py')
    changed = 0
    for old, new in PATCHES:
        if old not in txt:
            print('[WARN] 补丁未命中（跳过）:', old.strip().splitlines()[0][:60])
            continue
        if new in txt:
            continue                     # 已打过
        txt = txt.replace(old, new, 1)
        changed += 1
    if changed:
        open(dst, 'w', encoding='utf-8').write(txt)
    print('[OK] 生成', dst, ' 本次打补丁', changed, '处')
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--warmup', type=int, default=2)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    script = make_train_demo()

    cmd = [sys.executable, script,
           '--config_path', './config/tusimple_resnet18.yaml',
           '--data_url', DATA + os.sep,
           '--backbone_pretrain', '',
           '--epochs', str(args.epochs),
           '--warmup', str(args.warmup),
           '--batch_size', str(args.batch_size),
           '--train_url', OUT]
    print('[RUN]', ' '.join(cmd))
    r = subprocess.call(cmd, cwd=UF)
    if r != 0:
        print('[ERR] 训练退出码', r)
        sys.exit(r)

    cks = sorted(f for f in os.listdir(OUT) if f.endswith('.ckpt'))
    print('[OK] 产出权重:', cks)
    print('===== 实验一 训练完成 =====')


if __name__ == '__main__':
    main()

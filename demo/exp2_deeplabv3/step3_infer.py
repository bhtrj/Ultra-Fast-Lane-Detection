# -*- coding: utf-8 -*-
"""实验二（DeepLabV3 车道线分割）第 3 步：推理 + 分割结果可视化（演示用）

用法：
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python step3_推理演示.py

产出（<BASE>/out/exp2/）：
    seg_000.png ...   三联图：原图 | 真值 | 预测
    demo_grid.jpg     4 张三联图拼一张，便于投屏
"""
import os
import sys

import cv2
import numpy as np
from mindspore import Tensor, context, load_checkpoint, load_param_into_net

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('CH3_DEMO', os.path.abspath(os.path.join(HERE, '..', '..')))
DL = os.path.join(BASE, 'code_deeplab')
ROOT = os.path.join(BASE, 'data', 'laneseg_sample')
OUT = os.path.join(BASE, 'out', 'exp2')
sys.path.insert(0, DL)

context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')

from src.deeplab_v3 import DeepLabV3          # noqa: E402
from step2_train import SegData, read_list       # noqa: E402

W = H = 320


def pick_ckpt():
    for n in ('deeplab_best.ckpt', 'deeplab_last.ckpt'):
        p = os.path.join(OUT, n)
        if os.path.exists(p):
            return p
    print('[ERR] 没有找到权重，请先运行 step2_训练.py')
    sys.exit(1)


def to_vis(mask):
    """0/1 掩码 -> 彩色图（背景深灰，车道线青色）"""
    out = np.full((H, W, 3), (60, 62, 68), np.uint8)
    out[mask == 1] = (0, 220, 220)
    return out


def miou(pred, gt):
    ious = []
    for c in (0, 1):
        p, g = (pred == c), (gt == c)
        u = (p | g).sum()
        if u == 0:
            continue
        ious.append((p & g).sum() / float(u))
    return float(np.mean(ious))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=None, help='数据根目录；默认合成数据，真实数据传 data/LaneSegReal')
    args = ap.parse_args()
    data_root = args.data_dir or ROOT

    ckpt = pick_ckpt()
    print('[CKPT]', ckpt)
    net = DeepLabV3('eval', num_classes=2, output_stride=8)
    load_param_into_net(net, load_checkpoint(ckpt))
    net.set_train(False)

    names = read_list('val.txt', data_root)[:8]
    ds = SegData(names, data_root)
    ious, accs, tiles = [], [], []

    for i, n in enumerate(names):
        img, mask = ds[i]
        logits = net(Tensor(img[None]))
        pred = logits.asnumpy().argmax(1)[0]
        raw = cv2.imread(os.path.join(data_root, 'images', n + '.jpg'))
        raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
        ious.append(miou(pred, mask))
        accs.append(float((pred == mask).mean()))

        tile = np.hstack([raw, to_vis(mask), to_vis(pred)])
        cv2.putText(tile, 'orig', (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        cv2.putText(tile, 'gt', (W + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        cv2.putText(tile, 'pred', (2 * W + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 220), 2)
        p = os.path.join(OUT, 'seg_%03d.png' % int(n))
        cv2.imwrite(p, tile)
        if len(tiles) < 4:
            tiles.append(cv2.resize(tile, (480, 160)))
        print('  [%s] mIoU=%.4f  pixel_acc=%.4f -> %s' % (n, ious[-1], accs[-1], p))

    print('[RESULT] 平均 mIoU = %.4f   平均像素准确率 = %.4f  (%d 张)'
          % (float(np.mean(ious)), float(np.mean(accs)), len(names)))

    if tiles:
        while len(tiles) < 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
        gp = os.path.join(OUT, 'demo_grid.jpg')
        cv2.imwrite(gp, grid)
        print('[OK] 拼图 ->', gp)

    print('===== 实验二 推理演示完成 =====')


if __name__ == '__main__':
    main()

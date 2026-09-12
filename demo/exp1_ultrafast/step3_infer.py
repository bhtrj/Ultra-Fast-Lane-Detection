# -*- coding: utf-8 -*-
"""实验一（Ultra-Fast-Lane-Detection）第 3 步：推理 + 车道线可视化（演示用）

用法：
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python step3_推理演示.py

产出（<BASE>/out/exp1/）：
    pred_000.jpg ...     原图 + 绿线(真值) + 红线(预测)
    demo_grid.jpg        4 张结果拼成一张，便于投屏
    控制台打印每张图的 TuSimple accuracy
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
from mindspore import Tensor, context, load_checkpoint, load_param_into_net

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('CH3_DEMO', os.path.abspath(os.path.join(HERE, '..', '..')))
UF = os.path.join(BASE, 'code_uf')
DATA = os.path.join(BASE, 'data', 'tusimple_sample')
OUT = os.path.join(BASE, 'out', 'exp1')
sys.path.insert(0, UF)

context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')

from src.dataset import create_lane_test_dataset          # noqa: E402
from src.network import ParsingNet                        # noqa: E402
from src.resnet import get_resnet                         # noqa: E402
from src.utils import TusimpleAccEval                     # noqa: E402

GRIDING_NUM = 100
NUM_LANES = 4
ROW_ANCHOR = list(range(64, 285, 4))       # 56 个行锚点，与 config/tusimple_resnet18.yaml 一致
# 行锚点在 288 高的网络输入坐标系里，画回 720 高的原图要乘 720/288 = 2.5
Y_DRAW = [int(r * 720 / 288) for r in ROW_ANCHOR]


def pick_ckpt():
    """优先取“最近一次训练”产出的纯网络权重（按修改时间，避免挑到历史残留的旧权重）"""
    cands = [p for p in glob.glob(os.path.join(OUT, '*_net.ckpt'))]
    if cands:
        return max(cands, key=os.path.getmtime)
    final = os.path.join(OUT, 'uf_lane_final.ckpt')
    if os.path.exists(final):
        return final
    cands = sorted(glob.glob(os.path.join(OUT, '*.ckpt')), key=os.path.getmtime)
    if cands:
        return cands[-1]
    print('[ERR] 没有找到权重，请先运行 step2_训练.py')
    sys.exit(1)


def draw(img, lanes, color, width=4):
    for lane in lanes:
        pts = [(int(x), y) for x, y in zip(lane, Y_DRAW) if x > 0]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, color, width)
    return img


def main():
    ckpt = pick_ckpt()
    print('[CKPT]', ckpt)

    net = ParsingNet('18', get_resnet('18'),
                     cls_dim=(GRIDING_NUM + 1, len(ROW_ANCHOR), NUM_LANES),
                     use_aux=True)
    load_param_into_net(net, load_checkpoint(ckpt))
    net.set_train(False)
    print('[OK] 模型加载完成')

    test_root = os.path.join(DATA, 'test_set')
    ds = create_lane_test_dataset('Tusimple', test_root, 'test_label.json', 1)
    with open(os.path.join(test_root, 'test_label.json')) as f:
        label_info = [json.loads(l) for l in f.readlines()]

    acc_eval = TusimpleAccEval()
    accs, tiles = [], []

    for data in ds.create_dict_iterator():
        imgs = data['image']
        idx = int(data['index'].asnumpy()[0])
        out = net(imgs).asnumpy()[0]                       # (101, 56, 4)

        pred_lanes = acc_eval.generate_tusimple_lines(out, (288, 800), GRIDING_NUM)
        info = label_info[idx]
        gt_lanes = np.array(info['lanes'])
        y_samples = np.array(info['h_samples'])
        acc = acc_eval.bench(pred_lanes, gt_lanes, y_samples)
        accs.append(acc)

        img = cv2.imread(os.path.join(test_root, info['raw_file']))
        draw(img, gt_lanes, (0, 200, 0), 3)                # 真值：绿
        draw(img, pred_lanes, (0, 60, 255), 4)             # 预测：红
        cv2.putText(img, 'acc=%.3f' % acc, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        p = os.path.join(OUT, 'pred_%03d.jpg' % idx)
        cv2.imwrite(p, img)
        if len(tiles) < 4:
            tiles.append(cv2.resize(img, (640, 360)))
        print('  [%d] acc=%.4f -> %s' % (idx, acc, p))

    print('[RESULT] 平均 accuracy = %.4f  (%d 张)' % (float(np.mean(accs)), len(accs)))

    if tiles:
        while len(tiles) < 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
        gp = os.path.join(OUT, 'demo_grid.jpg')
        cv2.imwrite(gp, grid)
        print('[OK] 拼图 ->', gp)

    print('===== 实验一 推理演示完成 =====')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""实验二（DeepLabV3 车道线分割）第 2 步：训练

用法：
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python step2_训练.py                    # 默认 10 epoch，约 5 分钟
    python step2_训练.py --epochs 20 --batch_size 4

产出：<BASE>/out/exp2/deeplab_best.ckpt、deeplab_last.ckpt
"""
import argparse
import os
import sys

import numpy as np
from mindspore import Tensor, context, nn, save_checkpoint
from mindspore.dataset import GeneratorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('CH3_DEMO', os.path.abspath(os.path.join(HERE, '..', '..')))
DL = os.path.join(BASE, 'code_deeplab')
ROOT = os.path.join(BASE, 'data', 'laneseg_sample')     # 默认用仓库自带的真实 sample
OUT = os.path.join(BASE, 'out', 'exp2')
sys.path.insert(0, DL)

context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')

from src.deeplab_v3 import DeepLabV3          # noqa: E402
from src.loss import SoftmaxCrossEntropyLoss  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


def read_list(txt, root=None):
    return [l.strip() for l in open(os.path.join(root or ROOT, txt)) if l.strip()]


IMG_SIZE = (320, 320)


class SegData:
    """PyNative 下最省事的数据源：直接吐 CHW float32 图 + HW int32 掩码。
    兼容两种数据：合成（320x320，0/1）与真实 TuSimple（1280x720，像素值=车道线编号），
    统一 resize 到 320x320、掩码二值化（>0 视为车道线）。"""

    def __init__(self, names, root=None):
        import cv2
        self.cv2 = cv2
        self.names = names
        self.root = root or ROOT

    def __getitem__(self, i):
        cv2 = self.cv2
        n = self.names[i]
        img = cv2.imread(os.path.join(self.root, 'images', n + '.jpg'))
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = img.transpose(2, 0, 1).astype(np.float32)
        mask = cv2.imread(os.path.join(self.root, 'masks', n + '.png'), 0)
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.int32)
        return img, mask

    def __len__(self):
        return len(self.names)


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
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--opt', default='momentum', choices=['momentum', 'adam'])
    ap.add_argument('--data_dir', default=None,
                    help='数据根目录（含 images/ masks/ train.txt val.txt）。默认合成数据；真实数据传 data/LaneSegReal')
    args = ap.parse_args()
    data_root = args.data_dir or ROOT
    os.makedirs(OUT, exist_ok=True)

    tr = GeneratorDataset(SegData(read_list('train.txt', data_root), data_root), ['image', 'mask'],
                          shuffle=True).batch(args.batch_size, drop_remainder=True)
    va_names = read_list('val.txt', data_root)
    va_ds = SegData(va_names, data_root)
    va_idx = list(range(min(8, len(va_names))))

    net = DeepLabV3('train', num_classes=2, output_stride=8)
    net.set_train()
    loss_fn = SoftmaxCrossEntropyLoss(num_cls=2)
    net_with_loss = nn.WithLossCell(net, loss_fn)
    # 实测：Adam 在本任务（从头训、batch=2、BN）上第 3 步就发散成 nan；
    # Momentum(lr=1e-3, momentum=0.9) 稳定下降，故默认用它。
    if args.opt == 'adam':
        opt = nn.Adam(net.trainable_params(), learning_rate=args.lr)
    else:
        opt = nn.Momentum(net.trainable_params(), learning_rate=args.lr, momentum=0.9)
    train_net = nn.TrainOneStepCell(net_with_loss, opt)
    train_net.set_train()

    best = 0.0
    for ep in range(1, args.epochs + 1):
        losses = []
        for d in tr.create_dict_iterator():
            loss = train_net(d['image'], d['mask'])
            losses.append(float(loss.asnumpy()))
        # 每个 epoch 在验证集上算 mIoU
        net.set_train(False)
        ious = []
        for i in va_idx:
            img, mask = va_ds[i]
            logits = net(Tensor(img[None]))
            pred = logits.asnumpy().argmax(1)[0]
            ious.append(miou(pred, mask))
        net.set_train()
        m = float(np.mean(ious))
        print('[EPOCH %2d] loss=%.4f  val mIoU=%.4f' % (ep, float(np.mean(losses)), m))
        if m > best:
            best = m
            save_checkpoint(net, os.path.join(OUT, 'deeplab_best.ckpt'))
            print('           -> best saved (%.4f)' % m)
    save_checkpoint(net, os.path.join(OUT, 'deeplab_last.ckpt'))
    print('[OK] best mIoU = %.4f' % best)
    print('===== 实验二 训练完成 =====')


if __name__ == '__main__':
    main()

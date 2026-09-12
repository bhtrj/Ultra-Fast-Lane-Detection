# -*- coding: utf-8 -*-
"""演示前环境自检：确认 MindSpore / Ascend / 依赖都就绪"""
import os
import subprocess
import sys

print('python      :', sys.version.split()[0])
try:
    import mindspore as ms
    print('mindspore   :', ms.__version__)
except Exception as e:
    print('mindspore   : 导入失败 ->', e)
    sys.exit(1)

print('ASCEND_HOME :', os.environ.get('ASCEND_HOME_PATH') or os.environ.get('ASCEND_HOME') or '(未设置)')
for k in ('LD_LIBRARY_PATH', 'PYTHONPATH'):
    v = os.environ.get(k, '')
    print('%-13s: %s' % (k, (v[:90] + '...') if len(v) > 90 else (v or '(空)')))

for m in ('numpy', 'cv2', 'scipy', 'sklearn', 'yaml', 'PIL'):
    try:
        __import__(m)
        print('  [OK]', m)
    except Exception as e:
        print('  [!!]', m, '缺失 ->', e)

try:
    import mindspore as ms
    from mindspore import Tensor
    import numpy as np
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='Ascend')
    a = Tensor(np.ones((2, 2), np.float32))
    print('Ascend 张量运算:', (a + a).asnumpy().tolist())
except Exception as e:
    print('Ascend 张量运算: 失败 ->', type(e).__name__, str(e)[:160])
    print('>>> 请先执行 source /usr/local/Ascend/ascend-toolkit/set_env.sh')

try:
    out = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=20)
    for line in out.stdout.splitlines():
        if 'NPU' in line or 'Chip' in line or 'Health' in line:
            print('npu-smi:', line.strip())
except Exception:
    print('npu-smi: 不可用（不影响训练）')

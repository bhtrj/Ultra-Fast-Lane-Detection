#!/bin/bash
# 一键跑通两个实验（在仓库根目录执行）
#   bash run_all.sh        全部
#   bash run_all.sh 1      只跑实验一
#   bash run_all.sh 2      只跑实验二
set -o pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd "$(dirname "$0")" || exit 1
mkdir -p out/exp1 out/exp2 log

ONLY=${1:-all}

echo "########## 0. 环境自检 ##########"
python 0_env_check.py 2>&1 | tee log/0_env.log

if [ "$ONLY" = "all" ] || [ "$ONLY" = "1" ]; then
echo "########## 实验一 Ultra-Fast-Lane-Detection ##########"
echo "--- 1/3 数据 ---"
python demo/exp1_ultrafast/step1_data_real.py 2>&1 | tee log/exp1_step1.log
echo "--- 2/3 训练 ---"
python demo/exp1_ultrafast/step2_train.py --epochs 8 --batch_size 2 2>&1 | tee log/exp1_step2.log
echo "--- 3/3 推理 ---"
python demo/exp1_ultrafast/step3_infer.py 2>&1 | tee log/exp1_step3.log
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "2" ]; then
echo "########## 实验二 DeepLabV3 车道线分割 ##########"
echo "--- 1/3 数据 ---"
python demo/exp2_deeplabv3/step1_data_real.py 2>&1 | tee log/exp2_step1.log
echo "--- 2/3 训练 ---"
python demo/exp2_deeplabv3/step2_train.py --epochs 20 --batch_size 2 2>&1 | tee log/exp2_step2.log
echo "--- 3/3 推理 ---"
python demo/exp2_deeplabv3/step3_infer.py 2>&1 | tee log/exp2_step3.log
fi

echo "########## 完成 ##########"
echo "实验一结果图: $(pwd)/out/exp1/demo_grid.jpg"
echo "实验二结果图: $(pwd)/out/exp2/demo_grid.jpg"

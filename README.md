# Ascend-debug
python version:3.7  CANN version：5.1.RC2

# 1. 安装Paddle主框架
直接安装提供的编译好的paddle whl包
```
pip install paddlepaddle-0.0.0-cp37-cp37m-linux_aarch64.whl
```
# 2. 安装Custom NPU插件
直接安装提供的编译好的Custom NPU包
```
pip install paddle_custom_npu-0.0.0-cp37-cp37m-linux_aarch64.whl
```
# 3. 准备代码与数据
下载PARL代码
```
git clone -b develop https://github.com/PaddlePaddle/PARL.git
```
安装PARL与依赖
```
cd PARL
pip install -e . 
pip install atari-py==0.2.6 gym==0.18.0
pip install pillow==9.3.0
```
修改train.py以支持NPU
```
diff --git a/examples/PPO/train.py b/examples/PPO/train.py
index fafa86f..299215d 100644
--- a/examples/PPO/train.py
+++ b/examples/PPO/train.py
@@ -14,6 +14,8 @@
 
 import argparse
 import numpy as np
+import os
+import paddle
 from parl.utils import logger, tensorboard
 
 from mujoco_config import mujoco_config
@@ -43,6 +45,8 @@ def run_evaluate_episodes(agent, eval_env, eval_episodes):
 
 
 def main():
+    paddle.set_device('ascend:{0}'.format(
+            os.getenv('FLAGS_selected_ascends', 0)))
     config = mujoco_config if args.continuous_action else atari_config
     if args.env_num:
         config['env_num'] = args.env_num
```
# 4. 训练模型
```
GLOG_v=4 python train.py
```

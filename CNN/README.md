CNN
====

# Overview
tensorflowの練習用
使用するデータはmnistやcifarを想定

# Description
- [2018年風TensorFlowでの学習処理の記述方法](http://ksksksks2.hatenadiary.jp/entry/20181008/1538994843)にある```MoniteredTrainedSession```や```tf.data```などの使い方を学ぶため、mnistやcifarを用いて、プログラムを作成する


# Requirement
新たにinstallするものはないため、[READMR.md](../README.md)を参照

# Usage
## Build Network
```train.py```内にある```set_model```関数に作成したいモデルを定義する  
listである**model_set**にlayerごとの情報をlistとしてappendする  
定義の方法は以下の通り  
- 基本設計：['layer種類', 'args']
- argsの設計
    - **fc**:[units, activation]
    - **dropout**:[units, activation, rate]
    - **ReLU**:なし
    - **conv**:[kernel, filter, strides]
    - **max_pool**:[pool_size, strides]
    - **avg_pool**:[pool_size, strides]
    - **BN**:なし

## Learning
```bash
$ python CNN/main.py --data (データ名) \
                     --n_epoch (学習回数) \
                     --batch_size (batch size) \
                     --lr (学習率) \
                     --opt (optimizer) \
                     --checkpoints_to_keep \
                     --keep_checkpoint_every_n_hours \
                     --save_checkpoint_steps
```
## Tensorboard
```
$ tensorboard --logdir=/path/to/logdir
```

## 
# Sample Result
```bash
---Start Learning------
data : mnistepoch : 1000
batch_size : 32
learning rate : 0.1
Optimizer : SGD
-----------------------
WARNING:tensorflow:From /home/rl/Desktop/tensorflow/CNN/data_load.py:42: map_and_batch (from tensorflow.contrib.data.python.ops.batching) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.experimental.map_and_batch(...)`.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
2018-12-18 10:57:16.400469: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-12-18 10:57:16.553101: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-12-18 10:57:16.554453: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: GeForce GTX 1060 6GB major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:06:00.0
totalMemory: 5.92GiB freeMemory: 5.15GiB
2018-12-18 10:57:16.554474: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2018-12-18 10:57:16.817170: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-12-18 10:57:16.817205: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
2018-12-18 10:57:16.817229: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
2018-12-18 10:57:16.817473: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4906 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:06:00.0, compute capability: 6.1)
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into ./results/181218_105715/model/model.ckpt.
INFO:tensorflow:accuracy = 0.09375, global_step = 0, loss = 2.7242298
INFO:tensorflow:global_step/sec: 270.383
INFO:tensorflow:accuracy = 0.65625, global_step = 100, loss = 0.9412532 (0.370 sec)
INFO:tensorflow:global_step/sec: 509.65
INFO:tensorflow:accuracy = 0.875, global_step = 200, loss = 0.4556799 (0.196 sec)
INFO:tensorflow:global_step/sec: 590.016
INFO:tensorflow:accuracy = 0.9375, global_step = 300, loss = 0.3789385 (0.169 sec)
INFO:tensorflow:global_step/sec: 610.722
INFO:tensorflow:accuracy = 0.71875, global_step = 400, loss = 1.0471249 (0.164 sec)
INFO:tensorflow:global_step/sec: 613.563
INFO:tensorflow:accuracy = 0.75, global_step = 500, loss = 0.52988124 (0.163 sec)
INFO:tensorflow:global_step/sec: 601.447
INFO:tensorflow:accuracy = 0.90625, global_step = 600, loss = 0.38489723 (0.166 sec)
INFO:tensorflow:global_step/sec: 623.636
INFO:tensorflow:accuracy = 0.90625, global_step = 700, loss = 0.36388448 (0.160 sec)
INFO:tensorflow:global_step/sec: 632.068
INFO:tensorflow:accuracy = 0.90625, global_step = 800, loss = 0.41966242 (0.158 sec)
INFO:tensorflow:global_step/sec: 641.423
INFO:tensorflow:accuracy = 0.90625, global_step = 900, loss = 0.1906437 (0.156 sec)
INFO:tensorflow:Saving checkpoints for 1000 into ./results/181218_105715/model/model.ckpt.
```
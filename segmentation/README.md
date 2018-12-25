Semantic Segmentation
==
[こちらのソース](https://qiita.com/tktktks10/items/0f551aea27d2f62ef708)を基に自分用に変更した

# Overview
tensorflowでSegmentationを行う

# Description
U-Netを用いてPascal VOC 2012のデータをSegmentationする

# Usage
```
$ python segmentation/train.py --n_epoch (学習回数) \
                               --batch_size (batch size) \
                               --lr (学習率) \
                               --opt (optimizer) \
                               --checkpoints_to_keep \
                               --keep_checkpoint_every_n_hours \
                               --save_checkpoint_steps
```
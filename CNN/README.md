CNN
====

# Overview
tensorflowの練習用
使用するデータはmnistやcifarを想定

# Description
- [2018年風TensorFlowでの学習処理の記述方法](http://ksksksks2.hatenadiary.jp/entry/20181008/1538994843)にある```MoniteredTrainedSession```や```tf.data```などの使い方を学ぶため、mnistやcifarを用いて、プログラムを作成する


# Requirement
```
$ brew install pyenv
$ pyenv install 3.6.7
$ pyenv global 3.6.7
$ pip install numpy
$ pip install matplotlib
$ pip install requests
$ pip install tensorflowß
```

# Usage
```bash
python main.py --data (データ名) \
               --n_epoch (学習回数) \
               --opt (optimizer) \
               --test
```
# Sample Result
```bash

```
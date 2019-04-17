"""Utilities for parsing PTB text files."""
# based on https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import tensorflow as tf


Py3 = sys.version_info[0] == 3

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n", "<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    
    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary

def ptb_producer(raw_data, num_steps=35, name=None):
    data_len = len(raw_data)
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    # raw_dataから入力シーケンスと正解データに分けて、学習データを作成する
    data_num = data_len - num_steps     # 作成できるデータ個数
    raw_data = tf.reshape(raw_data, (-1, 1))
    i = list(range(data_num)) # [0, 1, .., epoch_size-1] という整数を順ぐりに無限生成するイテレータ
    x = tf.strided_slice(raw_data[:data_num], i * num_steps, list(map(lambda x: x+num_steps, i)), strides=1)
    x.set_shape([data_num, num_steps])
    y = tf.strided_slice(raw_data, i * num_steps + 1, list(map(lambda x: x+num_steps + 1, i)), strides=1)  # 正解 y は x の次に来る単語なので、1を足してスライスを右に一つずらす
    y.set_shape([data_num, num_steps])
    return x, y



def sptb_producer(raw_data, batch_size, num_steps=35, name=None):
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)  # 扱いやすいようにビルトインの List から Tensor に変換

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size  # 行列の列数
        data = tf.reshape(raw_data[0 : batch_size * batch_len],
                        [batch_size, batch_len])  # 1次元リストだった raw_data を、batch_size x batch_len の行列に整形

        epoch_size = (batch_len - 1) // num_steps  # 1エポック (データの一回り) の大きさ
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()  # [0, 1, .., epoch_size-1] という整数を順ぐりに無限生成するイテレータ
        x = tf.strided_slice(data, [0, i * num_steps],
                            [batch_size, (i + 1) * num_steps])  # この使われ方の strided_slice は、data[0:batch_size, i*num_steps:(i+1)*num_steps] だと思って良い
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                            [batch_size, (i + 1) * num_steps + 1])  # 正解 y は x の次に来る単語なので、1を足してスライスを右に一つずらす
        y.set_shape([batch_size, num_steps])
        return x, y
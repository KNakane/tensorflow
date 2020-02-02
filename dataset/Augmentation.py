import os, sys
import numpy as np
import tensorflow as tf
import scipy.ndimage as ndimage

class Augmentation():
    def __init__(self, func):
        self.func = func

    def __call__(self, image, label):
        im_shape = image.shape
        [image, label] = tf.py_function(eval('self.' + self.func), inp=[image, label], Tout=[tf.float32, tf.float32])
        image.set_shape(im_shape)
        return image, label

    def shift(self, img, label, v=2, h=2):
        image = []
        for i in range(img.shape[0]):
            vertical = 2 * v * np.random.rand() - v
            horizontal = 2 * h * np.random.rand() - h
            if len(img[i].shape) == 2:
                image.append(ndimage.shift(img[i], [vertical, horizontal], cval=0))
            elif len(img[i].shape) == 3:
                image.append(ndimage.shift(img[i], [vertical, horizontal, 0], cval=0))
            else:
                print('Image data_shape -> {}'.format(img[i].shape))
                raise NotImplementedError()

        image = np.vstack((img, np.asarray(image)))
        label = np.hstack((label, label))
        return image, label

    def mirror(self, img, label):
        image = img[:, :, ::-1]
        image = np.vstack((img, np.asarray(image)))
        label = np.hstack((label, label))
        return image, label

    def rotate(self, img, label, angle_range=[-20, 20]):
        image = ndimage.rotate(img, np.random.uniform(angle_range[0], angle_range[1]), reshape=False)
        image = np.vstack((img, np.asarray(image)))
        label = np.hstack((label, label))
        return image, label

    def shift_rotate(self, img, label, v=2, h=2, angle_range=(-10, 10)):
        image = ndimage.shift(img, [v, h], cval=0)
        image = ndimage.rotate(image, np.random.uniform(angle_range[0], angle_range[1]), reshape=False)
        image = np.vstack((img, np.asarray(image)))
        label = np.hstack((label, label))
        return image, label


    def cutout(self, mask_size=7):
        # reference : https://arxiv.org/abs/1708.04552 or http://kenbo.hatenablog.com/entry/2017/11/28/211932
        # 実装 : https://www.kumilog.net/entry/numpy-data-augmentation#Cutout
        aug = []
        for i in trange(self.img.shape[0], desc="Augmentation -> Cutout"):
            origin_img = np.copy(self.img[i])
            mask_value = np.mean(origin_img)

            h, w = origin_img.shape[0], origin_img.shape[1]
            top = np.random.randint(0 - mask_size // 2, h - mask_size)
            left = np.random.randint(0 - mask_size // 2, w - mask_size)
            bottom = top + mask_size
            right = left + mask_size

            # はみ出した場合の処理
            if top < 0:
                top = 0
            if left < 0:
                left = 0

            # マスク部分の画素値を平均値で埋める
            origin_img[top:bottom, left:right].fill(mask_value)
            aug.append(origin_img)
        self.aug_img = np.vstack((self.aug_img, np.asarray(aug)))
        self.aug_label = np.hstack((self.aug_label, self.label))
        return self.aug_img, self.aug_label

    def random_erace(self, p=0.8, s=(0.02, 0.4), r=(0.3, 3)):
        aug = []
        for i in trange(self.img.shape[0], desc="Augmentation -> Random Erace"):
            # マスクするかしないか
            if np.random.rand() > p:
                aug.append(self.img[i])
                continue
            image = np.copy(self.img[i])

            # マスクする画素値をランダムで決める
            mask_value = np.random.randint(0, 256)
            h, w = image.shape[0], image.shape[1]
            # マスクのサイズを元画像のs(0.02~0.4)倍の範囲からランダムに決める
            mask_area = np.random.randint(h * w * s[0], h * w * s[1])

            # マスクのアスペクト比をr(0.3~3)の範囲からランダムに決める
            mask_aspect_ratio = np.random.rand() * (r[1] - r[0]) + r[0]

            # マスクのサイズとアスペクト比からマスクの高さと幅を決める
            # 算出した高さと幅(のどちらか)が元画像より大きくなることがあるので修正する
            mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
            if mask_height > h - 1:
                mask_height = h - 1
            mask_width = int(mask_aspect_ratio * mask_height)
            if mask_width > w - 1:
                mask_width = w - 1

            top = np.random.randint(0, h - mask_height)
            left = np.random.randint(0, w - mask_width)
            bottom = top + mask_height
            right = left + mask_width
            image[top:bottom, left:right].fill(mask_value)
            aug.append(image)
        self.aug_img = np.vstack((self.aug_img, np.asarray(aug)))
        self.aug_label = np.hstack((self.aug_label, self.label))
        return self.aug_img, self.aug_label
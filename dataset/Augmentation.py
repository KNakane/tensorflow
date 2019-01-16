import os, sys
import numpy as np
from tqdm import trange
from scipy.misc import imresize
from scipy.ndimage.interpolation import shift, rotate

class Augment():
    def __init__(self, images, labels):
        self.img = images
        self.label = labels
        self.aug_img = self.img.copy()
        self.aug_label = self.label.copy() 

    def shift(self, v=3, h=3):
        aug = []
        for i in trange(self.img.shape[0], desc="Augmentation -> Shift"):
            vertical = 2*v * np.random.rand() - v
            horizontal = 2*h * np.random.rand() - h
            aug.append(shift(self.img[i], [vertical, horizontal], cval=0))
        self.aug_img = np.vstack((self.aug_img, np.asarray(aug)))
        self.aug_label = np.hstack((self.aug_label, self.label))
        return self.aug_img, self.aug_label

    def mirror(self):
        aug = []
        for i in trange(self.img.shape[0], desc="Augmentation -> Mirror"):
            aug.append(self.img[i, :, ::-1, :])
        self.aug_img = np.vstack((self.aug_img, np.asarray(aug)))
        self.aug_label = np.hstack((self.aug_label, self.label))
        return self.aug_img, self.aug_label

    def rotate(self, angle_range=(-20, 20)):
        aug = []
        for i in trange(self.img.shape[0], desc="Augmentation -> Rotate"):
            h, w = self.img[i].shape[0], self.img[i].shape[1]
            angle = np.random.randint(*angle_range)
            image = rotate(self.img[i], angle)
            aug.append(imresize(image, (h, w)))
        self.aug_img = np.vstack((self.aug_img, np.asarray(aug)))
        self.aug_label = np.hstack((self.aug_label, self.label))
        return self.aug_img, self.aug_label

    def shift_rotate(self, v=3, h=3, angle_range=(-20, 20)):
        aug = []
        for i in trange(self.img.shape[0], desc="Augmentation -> Shift & Rotate"):
            vertical = 2*v * np.random.rand() - v
            horizontal = 2*h * np.random.rand() - h
            self.img[i] = shift(self.img[i], [vertical, horizontal], cval=0)
            h, w = self.img[i].shape[0], self.img[i].shape[1]
            angle = np.random.randint(*angle_range)
            image = rotate(self.img[i], angle)
            aug.append(imresize(image, (h, w)))
        self.aug_img = np.vstack((self.aug_img, np.asarray(aug)))
        self.aug_label = np.hstack((self.aug_label, self.label))
        return self.aug_img, self.aug_label

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

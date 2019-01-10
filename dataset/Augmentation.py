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

    def shift(self, v, h):
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
            h, w, _ = self.img[i].shape
            angle = np.random.randint(*angle_range)
            image = rotate(self.img[i], angle)
            aug.append(imresize(image, (h, w)))
        self.aug_img = np.vstack((self.aug_img, np.asarray(aug)))
        self.aug_label = np.hstack((self.aug_label, self.label))
        return self.aug_img, self.aug_label

    def shift_rotate(self, v, h, angle_range=(-20, 20)):   
        pass
        return

    def cutout(self):
        #reference : https://arxiv.org/abs/1708.04552 or http://kenbo.hatenablog.com/entry/2017/11/28/211932
        pass
        return

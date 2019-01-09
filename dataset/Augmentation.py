import os, sys
import numpy as np
from tqdm import trange
from scipy.ndimage.interpolation import shift

class Augment():
    def __init__(self, images, labels):
        self.img = images
        self.label = labels
        self.aug_img = self.img.copy()
        self.aug_label = self.label.copy() 

    def shift(self, v, h):
        aug = []
        for i in trange(self.img.shape[0], desc="Augmentation -> shift"):
            vertical = 2*v * np.random.rand() - v
            horizontal = 2*h * np.random.rand() - h
            aug.append(shift(self.img[i], [vertical, horizontal], cval=0))
        self.aug_img = np.vstack((self.aug_img, np.asarray(aug)))
        self.aug_label = np.hstack((self.aug_label, self.label))
        return self.aug_img, self.aug_label

    def mirror(self):
        pass
        #return np.array(self.aug_img), np.array(self.aug_label)
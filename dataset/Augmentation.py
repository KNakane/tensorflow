import os, sys
import numpy as np
from scipy.ndimage.interpolation import shift

class Augment():
    def __init__(self, images, labels):
        self.img = images
        self.label = labels
        self.aug_img = self.img.copy()
        self.aug_label = self.label.copy() 

    def shift(self, v, h):
        aug = shift(self.img, [0, v, h], cval=0)
        self.aug_img = np.vstack((self.aug_img,self.img))
        self.aug_label = np.vstack((self.aug_label,self.label))
        return self.aug_img, self.aug_label

    def mirror(self):
        pass
        #return np.array(self.aug_img), np.array(self.aug_label)
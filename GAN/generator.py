import os,sys
sys.path.append('./network')
import tensorflow as tf
from model import DNN

class Generator(DNN):
    def __init__(self):
        super().__init__()

    def inference(self):
        return
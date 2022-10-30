import os,sys,wget
import glob
import tarfile
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.datasets import *
from keras.utils import np_utils
from util import image_augmenter as ia

class Load():
    def __init__(self, init_size=(128, 128), one_hot=True):
        self.init_size = init_size
        self.one_hot = one_hot
        #self.dataset_dir = "segmentation/data_set"
        self.dataset_dir = "/home/rl/Desktop/models/research/deeplab/datasets/pascal_voc_seg"
        self.dir_original = self.dataset_dir + "/VOCdevkit/VOC2012/JPEGImages"
        self.dir_segmented = self.dataset_dir + "/VOCdevkit/VOC2012/SegmentationClass"
        self.deploy()
        self.category = DataSet.CATEGORY

    def deploy(self): # Download and Deploy
        if not os.path.exists(self.dataset_dir + "/VOCdevkit"):
            os.makedirs(self.dataset_dir,exist_ok=True)
            if not os.path.exists(self.dataset_dir + "/VOCtrainval_11-May-2012.tar"):
                print('DownLoad Pascal VOC 2012')
                wget.download(url="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
                              out=self.dataset_dir)
            with tarfile.open(self.dataset_dir+'/VOCtrainval_11-May-2012.tar', mode='r:*') as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, self.dataset_dir)
        return 

    def get_data(self):
        # Generate paths of images to load
        # 読み込むファイルのパスリストを作成
        paths_original, paths_segmented = self.generate_paths()

        # Extract images to ndarray using paths
        # 画像データをndarrayに展開
        return self.extract_images(paths_original, paths_segmented)

    def import_data(self):
        images_original, images_segmented = self.extract_images(paths_original, paths_segmented)
        # Get a color palette
        # カラーパレットを取得
        image_sample_palette = Image.open(paths_segmented[0])
        palette = image_sample_palette.getpalette()

        return DataSet(images_original, images_segmented, palette,
                       augmenter=ia.ImageAugmenter(size=self.init_size, class_count=len(DataSet.CATEGORY)))
    
    def generate_paths(self):
        # Get FILE name
        paths_original = glob.glob(self.dir_original + "/*")
        paths_segmented = glob.glob(self.dir_segmented + "/*")
        if len(paths_original) == 0 or len(paths_segmented) == 0:
            raise FileNotFoundError("Could not load images.")
        filenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], paths_segmented))
        paths_original = list(map(lambda filename: self.dir_original + "/" + filename + ".jpg", filenames))
        return paths_original, paths_segmented
    
    def extract_images(self, paths_original, paths_segmented):
        images_original, images_segmented = [], []

        # Load images from directory_path using generator
        print("Loading original images", end="", flush=True)
        for image in self.image_generator(paths_original, antialias=True):
            images_original.append(image)
            if len(images_original) % 200 == 0:
                print(".", end="", flush=True)
        print(" Completed", flush=True)
        print("Loading segmented images", end="", flush=True)
        for image in self.image_generator(paths_segmented, normalization=False):
            images_segmented.append(image)
            if len(images_segmented) % 200 == 0:
                print(".", end="", flush=True)
        print(" Completed")
        assert len(images_original) == len(images_segmented)

        # Cast to ndarray
        images_original = np.asarray(images_original, dtype=np.float32)
        images_segmented = np.asarray(images_segmented, dtype=np.uint8)

        # Change indices which correspond to "void" from 255
        images_segmented = np.where(images_segmented == 255, len(DataSet.CATEGORY)-1, images_segmented)

        # One hot encoding using identity matrix.
        if self.one_hot:
            print("Casting to one-hot encoding... ", end="", flush=True)
            identity = np.identity(len(DataSet.CATEGORY), dtype=np.uint8)
            images_segmented = identity[images_segmented]
            print("Done")
        else:
            pass

        return images_original, images_segmented

    def image_generator(self, file_paths, normalization=True, antialias=False):
        for file_path in file_paths:
            if file_path.endswith(".png") or file_path.endswith(".jpg"):
                # open a image
                image = Image.open(file_path)
                # to square
                image = self.crop_to_square(image)
                # resize by init_size
                if self.init_size is not None and self.init_size != image.size:
                    if antialias:
                        image = image.resize(self.init_size, Image.ANTIALIAS)
                    else:
                        image = image.resize(self.init_size)
                # delete alpha channel
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                image = np.asarray(image)
                if normalization:
                    image = image / 255.0
                yield image
        return

    def crop_to_square(self, image):
        size = min(image.size)
        left, upper = (image.width - size) // 2, (image.height - size) // 2
        right, bottom = (image.width + size) // 2, (image.height + size) // 2
        return image.crop((left, upper, right, bottom))



    def load(self, images, labels, batch_size, buffer_size=1000, is_training=False):
        def preprocess_fn(image, label):
            '''A transformation function to preprocess raw data
            into trainable input. '''
            x = tf.reshape(tf.cast(image, tf.float32), (self.init_size[0], self.init_size[1], 3))
            y = tf.reshape(tf.cast(label, tf.uint8), (self.init_size[0], self.init_size[1], len(DataSet.CATEGORY)))
            return x, y

        self.features_placeholder = tf.placeholder(images.dtype, images.shape, name='input_images')
        self.labels_placeholder = tf.placeholder(labels.dtype, labels.shape, name='correct_data')
        dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))

        # Transform and batch data at the same time
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            preprocess_fn, batch_size,
            num_parallel_batches=4,  # cpu cores
            drop_remainder=True if is_training else False))

        if is_training:
            dataset = dataset.shuffle(buffer_size).repeat()  # depends on sample size
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

class DataSet(object):
    CATEGORY = (
        "ground",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
        "void"
    )

    def __init__(self, images_original, images_segmented, image_palette, augmenter=None):
        assert len(images_original) == len(images_segmented), "images and labels must have same length."
        self._images_original = images_original
        self._images_segmented = images_segmented
        self._image_palette = image_palette
        self._augmenter = augmenter

    @property
    def images_original(self):
        return self._images_original

    @property
    def images_segmented(self):
        return self._images_segmented

    @property
    def palette(self):
        return self._image_palette

    @property
    def length(self):
        return len(self._images_original)

    @staticmethod
    def length_category():
        return len(DataSet.CATEGORY)

    def print_information(self):
        print("****** Dataset Information ******")
        print("[Number of Images]", len(self._images_original))

    def __add__(self, other):
        images_original = np.concatenate([self.images_original, other.images_original])
        images_segmented = np.concatenate([self.images_segmented, other.images_segmented])
        return DataSet(images_original, images_segmented, self._image_palette, self._augmenter)

    def shuffle(self):
        idx = np.arange(self._images_original.shape[0])
        np.random.shuffle(idx)
        self._images_original, self._images_segmented = self._images_original[idx], self._images_segmented[idx]

    def transpose_by_color(self):
        self._images_original = self._images_original.transpose(0, 3, 1, 2)
        self._images_segmented = self._images_segmented.transpose(0, 3, 1, 2)

    def perm(self, start, end):
        end = min(end, len(self._images_original))
        return DataSet(self._images_original[start:end], self._images_segmented[start:end], self._image_palette,
                       self._augmenter)

    def __call__(self, batch_size=20, shuffle=True, augment=True):
        """
        `A generator which yields a batch. The batch is shuffled as default.
         バッチを返すジェネレータです。 デフォルトでバッチはシャッフルされます。
        Args:
            batch_size (int): batch size.
            shuffle (bool): If True, randomize batch datas.
        Yields:
            batch (ndarray[][][]): A batch data.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be more than 1.")
        if shuffle:
            self.shuffle()

        for start in range(0, self.length, batch_size):
            batch = self.perm(start, start+batch_size)
            if augment:
                assert self._augmenter is not None, "you have to set an augmenter."
                yield self._augmenter.augment_dataset(batch, method=[ia.ImageAugmenter.NONE, ia.ImageAugmenter.FLIP])
            else:
                yield batch


if __name__ == "__main__":
    load = Load()
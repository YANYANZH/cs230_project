"""
Written by Yanyan Zhao.
Image folder contains all the images while label folder contains all the masks.
Image and its mask are with same name.
"""


from torch.utils.data import Dataset
import os
import random
from PIL import Image
import numpy as np

def data_split(image_dir, label_dir):

    imagenames = os.listdir(image_dir)
    filenames = imagenames
    filenames.sort()
    random.seed(230)
    random.shuffle(filenames)
    # split the dataset into train/dev/test(80/10/10).
    split_1 = int(0.8 * len(filenames))
    split_2 = int(0.9 * len(filenames))
    train = filenames[:split_1]
    dev = filenames[split_1:split_2]
    test = filenames[split_2:]
    train_image = [os.path.join(image_dir, f) for f in train]
    train_label = [os.path.join(label_dir, f) for f in train]
    dev_image = [os.path.join(image_dir, f) for f in dev]
    dev_label = [os.path.join(label_dir, f) for f in dev]
    test_image = [os.path.join(image_dir, f) for f in test]
    test_label = [os.path.join(label_dir, f) for f in test]

    return train_image, train_label, dev_image, dev_label, test_image, test_label


class EMDataset(Dataset):
    def __init__(self, image, label):
        self.imagenames = image
        self.labelnames = label

    def __len__(self):
        return len(self.imagenames)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.imagenames[idx]))
        label = np.array(Image.open(self.labelnames[idx]))
        return image, label

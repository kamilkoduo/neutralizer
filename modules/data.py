import numpy as np
import pandas as pd

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

from sklearn.model_selection import StratifiedShuffleSplit

from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from os.path import basename

from pathlib import Path
from torch.utils.data import Dataset
import enum
from collections import defaultdict
from PIL import Image
from skimage import io

RANDOM_STATE = 42
# Data

DATA_RAW_ROOT = '../data/raw'

TEST_SIZE = 0.1
TRAIN_BATCH = 64
VALID_BATCH = 1024


RAW_JAFFE_DATA_DIR = Path('../data/raw/jaffe')

# IMG SIZES consts
IMG_SIZE_DEEP_EMOTION = (48, 48)
IMG_SIZE_VGG16 = (224, 224)
IMG_SIZE_DEFAULT = (64,64)

# todo: ? add randomized augmentation as transform for train samples ?

def make_transforms(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


class Expression(enum.Enum):
    NEUTRAL = 'NE'
    ANGRY = 'AN'
    DISGUST = 'DI'
    FEAR = 'FE'
    HAPPY = 'HA'
    SAD = 'SA'
    SURPRISE = 'SU'


ExpEnc = {exp: idx for idx, exp in enumerate(Expression)}
ExpDec = {idx: exp for idx, exp in enumerate(Expression)}


class JAFFEDataset(Dataset):
    def __init__(self, data_dir=RAW_JAFFE_DATA_DIR, expression=None, img_size=IMG_SIZE_DEFAULT):
        """
        :param data_dir (Path)         : Path to images
        :param expression (Expression) : Enum member, expression label
        """

        self.data_dir = Path(data_dir)
        self.samples = self.build_dict()

        self.transform = make_transforms(img_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        desc = self.samples[idx]
        image_path = desc['path']
        image = Image.fromarray(io.imread(image_path, as_gray=True))

        if self.transform:
            image = self.transform(image)

        return image, desc

    def build_dict(self):
        samples = []
        for entry in self.data_dir.iterdir():
            tokens = entry.name.split('.')
            if tokens[-1] != 'tiff':
                continue

            exp = tokens[1][:2]
            iden = tokens[0]

            samples.append({
                'path': str(entry),
                'exp': ExpEnc[Expression(exp)],
                'iden': iden,
            })

        return samples

    def exp_split(self, test_size=0.1, random_state=42):
        exps = []
        for s in self.samples:
            exps.append(s['exp'])

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        splits = splitter.split(X=np.arange(len(exps), dtype=np.int), y=exps)

        train_idx, test_idx = next(splits)

        train_ds = Subset(self, train_idx)
        test_ds = Subset(self, test_idx)

        return train_ds, test_ds


class JAFFEDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, img_size=IMG_SIZE_DEFAULT):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size

        self.dataset = None
        self.train_split = None
        self.test_split = None

        self.num_workers = 10

    def prepare_data(self):
        if not RAW_JAFFE_DATA_DIR.exists():
            print('JAFFE data not found')
        else:
            print('JAFFE data found')

    def setup(self, scenario=None, stage=None):
        self.dataset = JAFFEDataset(img_size=self.img_size)

        if scenario == 'exp' or scenario is None:
            self.train_split, self.test_split = self.dataset.exp_split()

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    # def val_dataloader(self):
    #     return DataLoader(self.jaffe_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == '__main__':
    # test
    jaffe = JAFFEDataset()
    samples = jaffe.samples
    # print('\n'.join([f'{s}' for s in samples]))
    one, two = jaffe.exp_split()

    print(jaffe[0])
    print(one[0])

    dm = JAFFEDataModule()
    dm.prepare_data()
    dm.setup()
    dl = dm.train_dataloader()

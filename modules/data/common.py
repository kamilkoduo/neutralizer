import skimage
from PIL import Image
from torchvision.transforms import transforms
import enum
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader, Subset

# todo: ? add randomized augmentation as transform for train samples ?
from modules.data import config, common

import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from skimage import io

from modules.data import config


class Expression(enum.IntEnum):
    NEUTRAL = 0
    ANGRY = 1
    DISGUST = 2
    FEAR = 3
    HAPPY = 4
    SAD = 5
    SURPRISE = 6


def make_transform(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def neutral_extension(descriptions):
    # make list of descs instead

    neutrals = defaultdict(list)
    for idx, desc in enumerate(descriptions):
        if desc['exp'] == Expression.NEUTRAL.value:
            neutrals[desc['iden']].append(idx)

    for desc in descriptions:
        if desc['exp'] != Expression.NEUTRAL.value:
            desc['neutral'] = neutrals[desc['iden']]

    return descriptions


class CommonDataset(Dataset):
    def __init__(self, descriptions, neutral=False, img_size=config.IMG_SIZE_DEFAULT):
        # drop None values
        self.descriptions = [e for e in descriptions if e]

        if neutral:
            self.descriptions = neutral_extension(self.descriptions)

        self.transform = make_transform(img_size)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        desc = self.descriptions[idx]

        image = Image.fromarray(skimage.io.imread(desc['path'], as_gray=True))

        if self.transform:
            image = self.transform(image)

        result = {
            'image': image,
            'desc': desc,
        }

        if 'neutral' in desc:
            result['neutral'] = [self.__getitem__(idx) for idx in desc['neutral']]

        return result

    # todo: how to distribute identities, split them between test train?

    def exp_split(self, test_size=0.1, random_state=42):
        exps = [desc['exp'] for desc in self.descriptions]

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        splits = splitter.split(X=np.arange(len(exps), dtype=np.int), y=exps)

        train_idx, test_idx = next(splits)

        train_ds = Subset(self, train_idx)
        test_ds = Subset(self, test_idx)

        return train_ds, test_ds

    def exp_subset(self, expression):
        indexes = [idx for idx, desc in enumerate(self.descriptions) if desc['exp'] == expression.value]
        return Subset(self, indexes)


class CommonDataModule(pl.LightningDataModule):
    def __init__(self, dataset_class, batch_size=32, img_size=config.IMG_SIZE_DEFAULT, num_workers=16):
        super().__init__()
        self.dataset_class = dataset_class

        self.batch_size = batch_size
        self.img_size = img_size

        self.dataset = None
        self.train_split = None
        self.test_split = None

        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, expression=None, neutral=False, scenario=None, stage=None):
        self.dataset = self.dataset_class(img_size=self.img_size, neutral=neutral)

        if expression:
            self.dataset = self.dataset.exp_subset(expression)

        if scenario == 'exp' or scenario is None:
            self.train_split, self.test_split = self.dataset.exp_split()

        if scenario == 'train':
            self.train_split = self.dataset

        if scenario == 'test':
            self.test_split = self.dataset

    def train_dataloader(self):
        if self.train_split:
            return DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    # def val_dataloader(self):
    #     return DataLoader(self.jaffe_val, batch_size=self.batch_size)

    def test_dataloader(self):
        if self.test_split:
            return DataLoader(self.test_split, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

import random

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

    def read_image(self, idx):
        path = self.descriptions[idx]['path']

        image = Image.fromarray(skimage.io.imread(path, as_gray=True))
        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.read_image(idx)
        desc = self.descriptions[idx]

        result = image, desc

        if 'neutral' in desc:
            neutrals = [self.read_image(idx) for idx in desc['neutral']]
            result = *result, neutrals

        return result

    # todo: how to distribute identities, split them between test train?

    def exp_split_ind(self, test_size=0.1, random_state=42):
        exps = [desc['exp'] for desc in self.descriptions]

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        splits = splitter.split(X=np.arange(len(exps), dtype=np.int), y=exps)

        return next(splits)

    def exp_subset_ind(self, expression):
        indexes = [idx for idx, desc in enumerate(self.descriptions) if desc['exp'] == expression.value]
        return indexes

    def exp_indices(self, expression: Expression, subset_ind):
        ind = [idx for idx, desc in enumerate(self.descriptions) if desc['exp'] == expression.value]
        if subset_ind is not None:
            ind = [idx for idx in ind if idx in subset_ind]

        return ind


class CommonDataModule(pl.LightningDataModule):
    def __init__(self, dataset_class, batch_aligned=False, neutral=False, batch_size=1,
                 img_size=config.IMG_SIZE_DEFAULT, num_workers=16):
        super().__init__()
        self.dataset_class = dataset_class

        self.batch_aligned = batch_aligned
        self.batch_size = batch_size
        self.img_size = img_size

        self.dataset = self.dataset_class(img_size=self.img_size, neutral=neutral)
        self.train_split_ind = None
        self.test_split_ind = None

        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, scenario=None, stage=None):

        if scenario == 'exp' or scenario is None:
            self.train_split_ind, self.test_split_ind = self.dataset.exp_split_ind()

        elif scenario == 'train':
            self.train_split_ind = range(len(self.dataset))

        elif scenario == 'test':
            self.test_split_ind = range(len(self.dataset))

    def split_ind_dataloader(self, split_ind):
        if split_ind is not None:
            if self.batch_aligned:
                sampler = ExpressionBatchSampler(self.dataset, subset_ind=split_ind, batch_size=self.batch_size)
                return DataLoader(dataset=self.dataset, batch_sampler=sampler, num_workers=self.num_workers)
            else:
                return DataLoader(Subset(self.dataset, split_ind), batch_size=self.batch_size, num_workers=self.num_workers)

        return None

    def train_dataloader(self):
        return self.split_ind_dataloader(self.train_split_ind)

    def val_dataloader(self):
        return self.split_ind_dataloader(self.test_split_ind)

    def test_dataloader(self):
        return DataLoader(Subset(self.dataset, self.test_split_ind), num_workers=self.num_workers)


from torch.utils.data.sampler import SequentialSampler, BatchSampler, Sampler


class IndexSampler(Sampler):
    def __init__(self, indices):
        super().__init__(())
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class SingleExpressionBatchSampler(Sampler):
    def __init__(self, dataset: CommonDataset, expression: Expression, batch_size, subset_ind=None, drop_last=False):
        super().__init__(dataset)

        indices = dataset.exp_indices(expression=expression, subset_ind=subset_ind)

        self.sampler = BatchSampler(IndexSampler(indices), batch_size, drop_last)

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        return iter(self.sampler)


class ExpressionBatchSampler(Sampler):
    def __init__(self, dataset: CommonDataset, batch_size, subset_ind=None, drop_last=False):
        super().__init__(dataset)
        self.samplers = [
            SingleExpressionBatchSampler(dataset=dataset, expression=exp, batch_size=batch_size, subset_ind=subset_ind,
                                         drop_last=drop_last)
            for exp in Expression
        ]

    def __iter__(self):
        iters = {
            exp: iter(self.samplers[exp])
            for exp in Expression
        }
        while True:
            # Feel free to use the sequential strategy
            current_exp, current_iter = random.choice(list(iters.items()))
            try:
                batch = next(current_iter)
                yield batch
            except StopIteration:
                del iters[current_exp]

            if len(iters) == 0:
                break

    def __len__(self):
        return sum(len(s) for s in self.samplers)

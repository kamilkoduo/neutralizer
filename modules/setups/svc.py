from modules.setups.base_neutralizer import *

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import pandas as pd

logdir = 'lightning_logs/version_85'

if __name__ == '__main__':

    model = BaseNeutralizer.load_from_checkpoint(f'{logdir}/checkpoints/epoch=163.ckpt')
    model.eval()

    jaffe_dm = JAFFEDataModule(neutral=True,
                               img_size=data.config.IMG_SIZE_DEFAULT)
    jaffe_dm.setup(scenario='exp')

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    x_train = []
    y_train = []

    for sample in jaffe_dm.train_dataloader():
        l, n, exp = model.forward(sample)

        x_train.append(l.view(-1).detach().numpy())
        y_train.append(sample[1]['exp'])

    clf.fit(x_train, y_train)

    x_test = []
    y_test = []

    for sample in jaffe_dm.test_dataloader():
        l, n, exp = model.forward(sample)

        x_test.append(l.view(-1).detach().numpy())
        y_test.append(sample[1]['exp'])

    accuracy = clf.score(x_test, y_test)
    # accuracy = clf.score(x_train, y_train)

    print(accuracy)
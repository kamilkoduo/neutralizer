import torch
import pytorch_lightning as pl

from modules import data
from modules.data.common import Expression
from modules.data.jaffe import JAFFEDataModule

from vanilla.base_neutralizer import BaseDecoder, BaseEncoder

from torch.nn import MSELoss, ModuleList

HPARAMS_DEFAULT = {
    'image_size': 256,
    'conv_dim': 8,
    'leaky_relu': .001,
    'lr': .001,
    'beta1': .45,
    'beta2': .1,
    'batch_size': 80,
    'interpolation_mode': 'bicubic'
}
import numpy as np


class BaseNeutralizer(pl.LightningModule):
    def __init__(self, hparams=HPARAMS_DEFAULT, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hparams = hparams

        self.encoders = ModuleList([BaseEncoder(hparams) for _ in Expression])
        self.decoder = BaseDecoder(hparams)

        self.loss = MSELoss(reduction='mean')

        self.exp = 0

    def forward(self, x):
        latent = self.encoder(x)
        neutralized = self.decoder(latent)

        return {
            'latent': latent,
            'neutralized': neutralized
        }

    def compute(self, batch):
        image = batch['image']
        desc = batch['desc']
        label = desc['exp']
        neutrals = desc['neutral']

        print(f'warning: incorrect batch given: {label}')

        latent = [self.encoders[l](image[idx]) for idx, l in enumerate(label)]

        neutralized = self.decoder(latent)

        loss = self.loss(neutrals, neutralized)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute(batch)

        self.log('train_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute(batch)

        self.log('test_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return optimizer


if __name__ == '__main__':
    pl.seed_everything(42)

    hparams = HPARAMS_DEFAULT

    jaffe_dm = JAFFEDataModule(batch_aligned=True, batch_size=hparams['batch_size'], img_size=data.config.IMG_SIZE_DEFAULT)
    jaffe_dm.setup(neutral=True, scenario='exp')

    model = BaseNeutralizer()

    # trainer = pl.Trainer(gpus=-1, max_epochs=6)
    trainer = pl.Trainer(max_epochs=2, row_log_interval=1)
    trainer.fit(model, jaffe_dm)

    trainer.test(model, test_dataloaders=jaffe_dm.val_dataloader())

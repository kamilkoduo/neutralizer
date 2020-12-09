import torch
import pytorch_lightning as pl

from modules import data
from modules.data.common import Expression
from modules.data.jaffe import JAFFEDataModule

from vanilla.base_neutralizer import BaseDecoder, BaseEncoder

from torch.nn import MSELoss, ModuleList

HPARAMS_DEFAULT = {
    'image_size': 64,
    'conv_dim': 8,
    'leaky_relu': .001,
    'lr': .001,
    'batch_size': 30,
    'interpolation_mode': 'bicubic'
}


def fix_dim(tensor: torch.Tensor):
    if len(tensor.size()) == 3:
        return torch.unsqueeze(tensor, 1)
    return tensor


class BaseNeutralizer(pl.LightningModule):
    def __init__(self, hparams=HPARAMS_DEFAULT, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hparams = hparams

        self.encoders = ModuleList([BaseEncoder(hparams) for _ in Expression])
        self.decoder = BaseDecoder(hparams)

        self.loss = MSELoss(reduction='mean')

        self.exp = 0


    def forward(self, sample):
        image, desc, neutrals = sample

        image = fix_dim(image)

        latents = [self.encoders[exp.value](image) for exp in Expression]

        neutralized = [self.decoder(l) for l in latents]

        losses = [torch.stack([self.loss(fix_dim(n), nd) for n in neutrals]).mean() for nd in neutralized]

        ind = torch.argmin(torch.stack(losses))

        return latents[int(ind)], neutralized[int(ind)], ind

    def compute_loss(self, batch):
        image, desc, neutrals = batch
        label = desc['exp']

        # todo: check
        # print(f'warning: incorrect batch given: {label}')

        latent = self.encoders[label[0]](image)
        neutralized = self.decoder(latent)

        loss = torch.stack([self.loss(n, neutralized) for n in neutrals]).mean()

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        self.log('val/loss', loss)

        return loss

    # def test_step(self, batch, batch_idx):
    # loss = self.compute_loss(batch)
    #
    # self.log('test/loss', loss)
    #
    #
    #
    # print(self.forward())

    # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return optimizer


if __name__ == '__main__':
    pl.seed_everything(42)

    hparams = HPARAMS_DEFAULT

    jaffe_dm = JAFFEDataModule(neutral=True, batch_aligned=True, batch_size=hparams['batch_size'],
                               img_size=data.config.IMG_SIZE_DEFAULT)
    jaffe_dm.setup(scenario='exp')

    model = BaseNeutralizer()

    # trainer = pl.Trainer(gpus=-1, max_epochs=6)
    trainer = pl.Trainer(max_epochs=1000)
    trainer.fit(model, jaffe_dm)

    # trainer.test(model, test_dataloaders=jaffe_dm.test_dataloader())

    correct = 0
    all = 0
    for sample in jaffe_dm.test_dataloader():
        l, n, exp = model.forward(sample)
        if exp == sample[1]['exp']:
            correct += 1
        all += 1

    accuracy = correct / all
    print(f'accuracy kostil : {accuracy}')

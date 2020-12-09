import torch
import pytorch_lightning as pl
from torch import nn

from modules import data
from modules.data.common import Expression
from modules.data.jaffe import JAFFEDataModule

from torch.nn import MSELoss, ModuleList

from vanilla.layers import InterpolateUp

HPARAMS_DEFAULT = {
    'image_size': 256,
    'leaky_relu': .001,
    'lr': .001,
    'batch_size': 30,
    'interpolation_mode': 'bicubic'
}


def fix_dim(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.size()) == 3:
        return torch.unsqueeze(tensor, 0)
    return tensor


class BaseEncoder(nn.Module):

    def __init__(self, hparams):
        super(BaseEncoder, self).__init__()

        # we do not tune image size, just use 256 as JAFFE original

        # hyper parameters
        self.leaky_relu = hparams['leaky_relu']

        self.sigmoid = nn.LeakyReLU(self.leaky_relu, inplace=True)
        self.conv = lambda c_in, c_out: nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.batch_norm = lambda x: nn.BatchNorm2d(x)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # layers of the component
        self.layers = nn.Sequential(
            self.conv(1, 16),
            self.batch_norm(16),
            self.sigmoid,
            self.max_pool,

            self.conv(16, 16),
            self.batch_norm(16),
            self.sigmoid,
            self.max_pool,

            self.conv(16, 64),
            self.batch_norm(64),
            self.sigmoid,
            self.max_pool,

            self.conv(64, 64),
            self.batch_norm(64),
            self.sigmoid,
            self.max_pool,

            self.conv(64, 128),
            self.batch_norm(128),
            self.sigmoid,
            self.max_pool,

            self.conv(128, 128),
            self.batch_norm(128),
            self.sigmoid,
            self.max_pool,
        )

    def forward(self, x):
        return self.layers(x)


class BaseDecoder(nn.Module):
    def __init__(self, hparams: dict):
        super(BaseDecoder, self).__init__()

        # we do not tune image size, just use 256 as JAFFE original

        # hyper parameters
        self.leaky_relu = hparams['leaky_relu']
        self.interpolation_mode = hparams['interpolation_mode']

        self.sigmoid = nn.LeakyReLU(self.leaky_relu, inplace=True)
        self.deconv = lambda c_in, c_out: nn.ConvTranspose2d(c_in, c_out, kernel_size=3, padding=1)
        self.batch_norm = lambda x: nn.BatchNorm2d(x)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.X2interpolate = InterpolateUp(2, mode=self.interpolation_mode)

        # layers of the component
        self.layers = nn.Sequential(
            self.deconv(128, 128),
            self.batch_norm(128),
            self.sigmoid,
            self.X2interpolate,

            self.deconv(128, 64),
            self.batch_norm(64),
            self.sigmoid,
            self.X2interpolate,

            self.deconv(64, 64),
            self.batch_norm(64),
            self.sigmoid,
            self.X2interpolate,

            self.deconv(64, 16),
            self.batch_norm(16),
            self.sigmoid,
            self.X2interpolate,

            self.deconv(16, 16),
            self.batch_norm(16),
            self.sigmoid,
            self.X2interpolate,

            self.deconv(16, 1),
            self.batch_norm(1),
            self.sigmoid,
            self.X2interpolate,
        )

    def forward(self, x):
        return self.layers(x)


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

        latents = [self.encoders[exp.value].__call__(image) for exp in Expression]

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
                               img_size=hparams['image_size'])
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

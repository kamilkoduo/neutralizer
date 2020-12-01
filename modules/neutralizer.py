import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM

from modules import data
from vanilla.deep_emotion import DeepEmotion as DeepEmotionV
from modules.data__ import JAFFEDataModule

from vanilla.neutralizer import GrayVGGEncoder, GrayVGGDecoder


class Neutralizer(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = GrayVGGEncoder()
        self.decoder = GrayVGGDecoder()

    def forward(self, x):
        latent = self.encoder(x)
        neutralized = self.decoder(latent)

        return {
            'latent': latent,
            'neutralized': neutralized
        }

    def training_step(self, batch, batch_idx):
        image, desc = batch
        label = desc['exp']
        neutral = desc['neutral']


class DeepEmotion(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vanilla = DeepEmotionV()

    def forward(self, x):
        return self.vanilla.forward(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        image, desc = batch
        label = desc['exp']

        predictions = self(image)

        loss = nn.functional.cross_entropy(predictions, label)
        print(loss)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)

        return result

    # def test_step(self, batch, batch_idx):
    #     image, desc = batch
    #     label = desc['exp']
    #
    #     predictions = self(image)
    #
    #     loss = nn.functional.cross_entropy(predictions, label)
    #
    #     label_hat = predictions.argmax(dim=1).flatten()
    #
    #     accuracy = FM.accuracy(label, label_hat, num_classes=len(data.Expression))
    #
    #     result = pl.EvalResult(checkpoint_on=loss)
    #
    #     result.batch_acc = accuracy
    #     result.batch_len = label.shape[0]
    #     result.batch_loss = loss
    #
    #     return result
    #
    # def test_epoch_end(self, outputs):
    #     all_accs = outputs.batch_acc
    #     all_loss = outputs.batch_loss
    #     all_lens = outputs.batch_len
    #     all_lens = torch.tensor(all_lens, dtype=torch.float, device=self.device)
    #
    #     epoch_acc = torch.dot(all_accs, all_lens) / all_lens.sum()
    #     epoch_loss = torch.dot(all_loss, all_lens) / all_lens.sum()
    #
    #     result = pl.EvalResult(checkpoint_on=epoch_loss)
    #
    #     result.log('test_acc', epoch_acc, on_step=False, on_epoch=True)
    #     result.log('test_loss', epoch_loss, on_step=False, on_epoch=True)
    #
    #     return result
    #
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    pl.seed_everything(42)

    jaffe_dm = JAFFEDataModule(img_size=data.IMG_SIZE_DEEP_EMOTION)
    jaffe_dm.setup()

    model = DeepEmotion()

    # trainer = pl.Trainer(gpus=-1, max_epochs=6)
    trainer = pl.Trainer(max_epochs=2, row_log_interval=1)
    trainer.fit(model, jaffe_dm)
    # trainer.test(deep_emotion, jaffe.test_dataloader())
    # trainer.test(model_l, test_dataloaders=valid_dl)

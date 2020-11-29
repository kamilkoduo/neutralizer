import torch
import torchvision.models as models

from torch import nn

from modules import data
from modules.data import JAFFEDataModule


# todo how to improve this?

class GrayVGGEncoder(nn.Module):
    def __init__(self):
        super(GrayVGGEncoder, self).__init__()

        self.vgg16 = models.vgg16(pretrained=True)

        self.gray_first_layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)

        self.features = nn.Sequential(
            self.gray_first_layer,
            self.vgg16.features
        )

    def forward(self, x):
        x = self.features(x)
        x = self.vgg16.avgpool(x)

        return x


# decoder
class InterpolateUp(nn.Module):
    def __init__(self, factor, mode):
        super(InterpolateUp, self).__init__()
        self.interp = nn.functional.interpolate
        self.factor = factor
        self.mode = mode

    def forward(self, x):
        t = x.shape[-1]
        x = self.interp(x, size=t * self.factor, mode=self.mode)
        return x


class GrayVGGDecoder(nn.Module):
    def __init__(self, interpolation_mode='bilinear'):
        super(GrayVGGDecoder, self).__init__()

        self.vgg16 = models.vgg16(pretrained=True)

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=3, padding=0),
            # nn.BatchNorm2d()
            nn.ReLU(inplace=True),

            InterpolateUp(3, interpolation_mode),

            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=0),
            # nn.BatchNorm2d()
            nn.ReLU(inplace=True),

            InterpolateUp(3, interpolation_mode),

            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=0),
            # nn.BatchNorm2d()
            nn.ReLU(inplace=True),

            InterpolateUp(3, interpolation_mode),

            nn.AdaptiveMaxPool2d(data.IMG_SIZE_VGG16)
        )

    def forward(self, x):
        return self.layers(x)


# code to test the external model

if __name__ == '__main__':
    jaffe = JAFFEDataModule(img_size=data.IMG_SIZE_VGG16)
    jaffe.setup()
    image, desc = jaffe.dataset[0]

    print(image.size())
    image = image.unsqueeze(0)
    print(image.size())

    encoder = GrayVGGEncoder()
    latent = encoder(image)

    print(latent)
    print(latent.size())
    print(desc)

    decoder = GrayVGGDecoder()
    decoded = decoder(latent)

    print(decoded.size())

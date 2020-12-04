from torch import nn

from vanilla.layers import InterpolateUp


class BaseEncoder(nn.Module):

    def __init__(self, hparams):
        super(BaseEncoder, self).__init__()

        # we do not tune image size, just use 256 as JAFFE original

        # hyper parameters
        self.conv_dim = cd = hparams['conv_dim']
        self.leaky_relu = hparams['leaky_relu']

        self.sigmoid = nn.LeakyReLU(self.leaky_relu, inplace=True)
        self.conv = lambda c_in, c_out: nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.batch_norm = lambda x: nn.BatchNorm2d(x)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # layers of the component
        self.layers = nn.Sequential(
            # 1
            self.conv(1, cd),
            self.batch_norm(cd),
            self.sigmoid,
            self.max_pool,

            # 2
            self.conv(cd, cd * 2),
            self.batch_norm(cd * 2),
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
        self.conv_dim = cd = hparams['conv_dim']
        self.leaky_relu = hparams['leaky_relu']
        self.interpolation_mode = hparams['interpolation_mode']

        self.sigmoid = nn.LeakyReLU(self.leaky_relu, inplace=True)
        self.deconv = lambda c_in, c_out: nn.ConvTranspose2d(c_in, c_out, kernel_size=3, padding=1)
        self.batch_norm = lambda x: nn.BatchNorm2d(x)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.X2interpolate = InterpolateUp(2, mode=self.interpolation_mode)

        # layers of the component
        self.layers = nn.Sequential(
            # 2
            self.deconv(cd * 2, cd),
            self.batch_norm(cd),
            self.sigmoid,

            # self.max_pool,
            self.X2interpolate,

            # 1
            self.deconv(cd, 1),
            self.batch_norm(1),
            self.sigmoid,
            # self.max_pool,
            self.X2interpolate,
        )

    def forward(self, x):
        return self.layers(x)

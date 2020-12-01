import torch
import torchvision.models as models

# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# mobilenet = models.mobilenet_v2(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# mnasnet = models.mnasnet1_0(pretrained=True)
from torch import nn

from modules import data
from modules.data__ import JAFFEDataModule


# todo how to improve this?


class GrayVGG(nn.Module):
    def __init__(self):
        super(GrayVGG, self).__init__()

        self.vgg16 = models.vgg16(pretrained=True)

        self.gray_first_layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)

        self.features = nn.Sequential(
            self.gray_first_layer,
            self.vgg16.features
        )

        self.clf_last_layer = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, len(data.Expression)),
        )

        self.classifier = nn.Sequential(
            self.vgg16.classifier,
            self.clf_last_layer
        )


        # # vgg_first_layer = vgg16.features[0]
        # # w = [vgg_first_layer.state_dict()['weight'][:, i, :, :] for i in range(3)]
        # #
        # # new_w = torch.mean(torch.stack(w), dim=0)
        # #
        # # gray_first_layer = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # # gray_first_layer.weight = torch.nn.Parameter(new_w, requires_grad=True)
        # # gray_first_layer.bias = torch.nn.Parameter(vgg_first_layer.state_dict()['bias'], requires_grad=True)
        # #
        # clf_layer = nn.Linear(1000, len(data.Expression))

        # self.layers = nn.Sequential(gray_first_layer, vgg16.features[1:], clf_layer)

    def forward(self, x):
        x = self.features(x)

        x = self.vgg16.avgpool(x)
        x = torch.flatten(x, 1)

        out = self.classifier(x)

        return out


# code to test the external model

if __name__ == '__main__':
    jaffe = JAFFEDataModule(img_size=data.IMG_SIZE_VGG16)
    jaffe.setup()
    image, desc = jaffe.dataset[0]

    print(image.size())
    image = image.unsqueeze(0)
    print(image.size())

    model = GrayVGG()
    predictions = model(image)

    print(predictions)
    print(desc)

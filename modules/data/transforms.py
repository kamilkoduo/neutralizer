from torchvision.transforms import transforms
from torchvision.transforms import functional as F


class DeNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

        self._mean = [- self.mean[i] / self.std[i] for i, _ in enumerate(self.std)]
        self._std = [1 / self.std[i] for i, _ in enumerate(self.std)]

    def __call__(self, tensor):
        return F.normalize(tensor, self._mean, self._std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


default_denormalize = DeNormalize((0.5,), (0.5,))

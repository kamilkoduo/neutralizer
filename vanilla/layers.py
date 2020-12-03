import torch
import torchvision.models as models
from matplotlib.lines import Line2D

from torch import nn

from modules import data
from modules.data.jaffe import JAFFEDataModule

from torch.autograd import Function


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


class ReverseGradFunction(Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return - ctx.constant * grad_output, None



class ReverseGradModule(nn.Module):
    def __init__(self, constant=1):
        super(ReverseGradModule, self).__init__()
        self.constant = constant

    def forward(self, x):
        return ReverseGradFunction.apply(x, self.constant)


if __name__ == '__main__':
    model = ReverseGradModule(1)

    inp = torch.tensor([1.0,1.0], requires_grad=True)
    output = model(inp)
    output.sum().backward()

    print(inp.grad.data)

    print(inp)
    print(output)

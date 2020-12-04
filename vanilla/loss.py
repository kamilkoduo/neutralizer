import torch
import pytorch_msssim


from torch.nn import MSELoss

def mse_loss():
    """MSE loss"""

    def loss(reconstructed, original):
        return torch.mean((reconstructed - original) ** 2)

    return loss


def mean_loss(losses_list):
    losses = torch.stack(losses_list)
    return torch.mean(losses)


def ssim_loss():
    """ Structural Similarity Index Measure (SSIM) """

    def loss(reconstructed, original):
        return torch.ones(original.shape) - pytorch_msssim.ssim(reconstructed, original)

    return loss


def ms_ssim_loss():
    """ Multi-Scale SSIM (MS-SSIM) """

    def loss(reconstructed, original):
        return torch.ones(original.shape) - pytorch_msssim.ms_ssim(reconstructed, original)

    return loss()


def psnr_loss(max_value=255):
    """ Peak Signal Noise Ratio """

    def loss(reconstructed, original):
        mse = mse_loss()(reconstructed, original)
        if mse == 0:
            return 100
        m = torch.ones(mse.shape) * (max_value * max_value)
        return - 10 * torch.log10(m / mse)

    return loss


def mse_ssim_comb_loss(f_mse):
    """ MSE SSIM combined attentional loss """
    assert 0 < f_mse < 1

    def loss(reconstructed, original):
        f_ssim = 1 - f_mse
        # print(f_mse, f_ssim)
        return f_mse * mse_loss()(reconstructed, original) + f_ssim * ssim_loss()(reconstructed, original)

    return loss


def get_loss(loss_str: str):
    available = {
        'mse': mse_loss,
        'ssim': ssim_loss,
        'ms-ssim': ms_ssim_loss,
        'psnr': psnr_loss,
        'mse_ssim': mse_ssim_comb_loss,
    }
    loss = loss_str.split('__')
    try:
        loss_func = available[loss[0]]
        if len(loss) == 2:
            p = float(loss[1])
            return loss_func(p)
        return loss_func()

    except KeyError as e:
        raise ValueError('Undefined LOSS: {}'.format(e.args[0]))
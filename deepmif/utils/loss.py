import torch.nn as nn

from deepmif.model.mif_decoder import MIFDecoderOutput
from deepmif.utils import is_none


def eikonal_loss(gradient):
    gradient = gradient.reshape((-1, 3))
    return ((1.0 - gradient.norm(2, dim=-1)) ** 2).mean()


class MIFLoss(nn.Module):
    def __init__(self, loss_config):
        super().__init__()

        self.loss_config = loss_config

    def forward(self, pred: MIFDecoderOutput):
        loss = 0.0

        eikonal_loss = None if is_none(pred.gradient) else eikonal_loss(pred.gradient)
        ray_monotonicity = pred.ray_monotonicity
        mif_sign = pred.mif_sign
        mif_surface = pred.mif_surface

        if not is_none(eikonal_loss) and self.loss_config.weights.eikonal:
            loss += eikonal_loss * self.loss_config.weights.eikonal

        if not is_none(ray_monotonicity) and self.loss_config.weights.ray_monotonicity:
            loss += pred.ray_monotonicity * self.loss_config.weights.ray_monotonicity

        if not is_none(mif_sign) and self.loss_config.weights.mif_sign:
            loss += pred.mif_sign * self.loss_config.weights.mif_sign

        if not is_none(mif_surface) and self.loss_config.weights.mif_surface:
            loss += pred.mif_surface * self.loss_config.weights.mif_surface

        return {
            "eikonal_loss": eikonal_loss,
            "ray_monotonicity": ray_monotonicity,
            "mif_sign": mif_sign,
            "mif_surface": mif_surface,
            "loss": loss,
        }

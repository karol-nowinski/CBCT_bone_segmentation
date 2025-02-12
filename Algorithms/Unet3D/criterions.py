import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, epsilon=1e-6):
        intersection = (input * target).sum(dim=(2, 3, 4))
        union = input.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = 2 * intersection / (union + epsilon)
        return 1 - dice.mean()

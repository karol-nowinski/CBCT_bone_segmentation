import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, epsilon=1e-6):
        intersection = (input * target).sum(dim=(2, 3, 4))
        union = input.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = 2 * intersection / (union + epsilon)
        return 1 - dice.mean()




class MultiClassDiceLoss(nn.Module):
    def __init__(self,num_classes,epsilon = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self,inputs,targets):

        #print(f"🔍 inputs shape: {inputs.shape}")
        #print(f"🔍 targets shape: {targets.shape}")

        targets = targets.squeeze(1)

        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)
        #print(f"🔍 targets_one_hot 1 shape: {targets_one_hot.shape}")
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        #print(f"🔍 targets_one_hot 2 shape: {targets_one_hot.shape}")

        inputs_soft = F.softmax(inputs, dim=1)
        inputs_flat = inputs_soft.view(inputs.size(0), self.num_classes, -1)
        targets_flat = targets_one_hot.view(inputs.size(0), self.num_classes, -1)

        intersection = (inputs_flat * targets_flat).sum(-1)
        denominator = inputs_flat.sum(-1) + targets_flat.sum(-1)

        dice_score = (2 * intersection + self.epsilon) / (denominator + self.epsilon)

        loss = 1 - dice_score.mean()
        return loss
        # inputs = F.softmax(inputs, dim=1)  # [B, C, D, H, W]
        # targets_one_hot = F.one_hot(targets, self.num_classes)  # [B, D, H, W, C]
        # targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]

        # dims = (0, 2, 3, 4)
        # intersection = torch.sum(inputs * targets_one_hot, dims)
        # cardinality = torch.sum(inputs + targets_one_hot, dims)
        # dice_loss = 1 - (2. * intersection + self.epsilon) / (cardinality + self.epsilon)
        # return dice_loss.mean()


# Połączenie MultiClassDiceLoss z CrossEntropyLoss
class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5, num_classes=33):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss(num_classes=num_classes)
        self.w_ce = weight_ce
        self.w_dice = weight_dice

    def forward(self, outputs, targets):
        ce_loss = self.ce(outputs, targets)
        dice_loss = self.dice(outputs, targets)
        return self.w_ce * ce_loss + self.w_dice * dice_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    '''
    Implementacja funkcji straty Dice Loss
    '''

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, epsilon=1e-6):
        intersection = (input * target).sum(dim=(2, 3, 4))
        union = input.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = 2 * intersection / (union + epsilon)
        return 1 - dice.mean()




class MultiClassDiceLoss(nn.Module):
    '''
    Funkcja straty oparta na wieloklasowym współczynniku Dice
    '''
    def __init__(self,num_classes,epsilon = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self,inputs,targets):

        targets = targets.squeeze(1)

        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

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


class CombinedLoss(nn.Module):
    '''
    Funkcja straty będąca połączenie MultiClassDiceLoss z CrossEntropyLoss.
    '''

    def __init__(self, num_classes, dice_weight=1.0, ce_weight=0.5, ignore_index=0, class_weights=None, epsilon=1e-6):
        """
        num_classes: liczba klas w segmentacji (łącznie z tłem)
        ignore_index: indeks klasy, którą pominąć (np. 0 dla tła), lub None żeby nic nie pomijać
        class_weights: tensor wag dla klas (shape: [num_classes])
        """
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        # Ustawienie CE z opcjonalnymi wagami i ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index= -100
        )

    def forward(self, inputs, targets):
        
        # Dice
        targets = targets.squeeze(1).long()  # usunięcie zbędnego wymiaru: (B, 1, D, H, W) → (B, D, H, W)

        # Cross entropy (CE)
        ce_l = self.ce_loss(inputs,targets)

        # Dice
        # One-hot encode ground truth
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)  # (B, D, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)

        # Apply softmax to logits
        inputs_soft = F.softmax(inputs, dim=1)  # (B, C, D, H, W)

        # spłaszczenie dla wyliczenia Dice
        inputs_flat = inputs_soft.view(inputs.size(0), self.num_classes, -1)
        targets_flat = targets_one_hot.view(inputs.size(0), self.num_classes, -1)

        # Wyliczenie Dice score
        intersection = (inputs_flat * targets_flat).sum(-1)
        denominator = inputs_flat.sum(-1) + targets_flat.sum(-1)
        dice_score = (2. * intersection + self.epsilon) / (denominator + self.epsilon)

        # Pomijamy klasę, jeśli ignore_index ustawiony
        if self.ignore_index is not None:
            mask = torch.ones(self.num_classes, device=inputs.device, dtype=torch.bool)
            mask[self.ignore_index] = False
            dice_score = dice_score[:, mask]

            if self.class_weights is not None:
                weights = self.class_weights[mask]
            else:
                weights = torch.ones(dice_score.shape[1], device=inputs.device)
        else:
            if self.class_weights is not None:
                weights = self.class_weights
            else:
                weights = torch.ones(self.num_classes, device=inputs.device)

        # średnia ważona Dice loss
        dice_loss = 1 - (weights * dice_score).sum() / weights.sum()

        return self.dice_weight * dice_loss + self.ce_weight * ce_l
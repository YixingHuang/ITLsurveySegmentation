import torch.nn as nn
import torch as torch


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        weight = target + 0.1
        intersection = torch.sum(pred * target * weight, dim=(1,2,3))
        union = torch.sum(pred * weight, dim=(1,2,3)) + torch.sum(target * weight, dim=(1,2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        loss = torch.sum(loss)
        return loss


class DiceAccuracy(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return dsc
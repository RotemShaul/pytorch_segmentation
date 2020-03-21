import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight 
from utils.lovasz_losses import lovasz_softmax

def make_one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)
    #print("classes {}".format(classes))
    #print("counts {}".format(counts))
    #print("cls_w {}".format(cls_w))
    #weights = np.ones(2)
    #weights[classes] = cls_w
    #print("weights {}".format(weights))
    return torch.from_numpy(cls_w).float().cuda()

class CrossEntropyLoss2dWeight(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2dWeight, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, target):
        weights = get_weights(target)
        CE = nn.CrossEntropyLoss(weight=weights, ignore_index=self.ignore_index, reduction=self.reduction)
        loss = CE(output, target)
        return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss

class CEOhem(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CEOhem, self).__init__()
        reduction = 'none'
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.device = 0
        self.ratio = 3
        
    def forward(self, output, target):
        x_ = output.clone()  #.flatten().to(self.device)
        y_ = target.clone() #flatten().to(self.device)

        #print("x, y {} {}".format(x_.size(), y_.size()))

        x_pos = x_.to(self.device)
        x_pos_0 = x_pos[:, 0, :, :].to(self.device)
        x_pos_1 = x_pos[:, 1, :, :].to(self.device)

        #print("xpos0, xpos1, y {} {} {}".format(x_pos_0.size(), x_pos_1.size(), y_.size()))

        x_pos_0 = x_pos_0[y_ == 1]
        x_pos_1 = x_pos_1[y_ == 1]
        #print(x_pos_0.size())
        #print(x_pos_1.size())
        x_pos_final = torch.stack((x_pos_0, x_pos_1)).to(self.device)

        #print("x, y {} {}".format(x_pos_final.size(), y_.size()))
        y_pos = torch.ones(x_pos_final.size(0)).to(self.device)
        
        x_neg = x_.to(self.device)
        x_neg_0 = x_neg[:, 0, :, :].to(self.device)
        x_neg_1 = x_neg[:, 1, :, :].to(self.device)
        
        #print("xneg0, xneg1, y {} {} {}".format(x_neg_0.size(), x_neg_1.size(), y_.size()))
        x_neg_0 = x_neg_0[y_ == 0]
        x_neg_1 = x_neg_1[y_ == 0]
        #print(x_neg_0.size())
        #print(x_neg_1.size())
        x_neg_final = torch.stack((x_neg_0, x_neg_1)).to(self.device)

        #print("x, y {} {}".format(x_neg_final.size(), y_.size()))
        y_neg = torch.ones(x_neg_final.size(0)).to(self.device)
        
        pos_losses = self.CE(x_pos_final, y_pos.long()).mean()  # we need the gradients

        print("numel {}".format(x_pos_final.numel()))
        with torch.no_grad():
            neg_losses = self.CE(x_neg_final, y_neg.long())

        _, idxs = neg_losses.topk(min(x_pos_final.numel() * self.ratio, neg_losses.numel()))
        neg_losses_topk = self.CE(x_neg_final[idxs], y_neg[idxs].long()).mean()

        # return {
        #    'loss': (3 * neg_losses_topk + pos_losses) / 4,
        #    'pos_loss': pos_losses.item(),
        #    'neg_loss': neg_losses.mean().item(),
        #    'neg_topk_loss': neg_losses_topk.item(),
        #    'wo_ohem_loss': criterion(seg_inputs, seg_targets).mean().item()
        # }

        # loss = 3 * neg_losses_topk + pos_losses) / 4
        loss = (neg_losses_topk + (3 * pos_losses)) / 4
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss

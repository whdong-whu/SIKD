import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""

    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss



class Com_class_mean(nn.Module):
    def __init__(self, classes):
        super(Com_class_mean, self).__init__()
        self.classes = classes

    def forward(self, img, gt):
        img_class_mean = img.clone()
        # plt.imshow(img.detach().cpu()[0][0])
        # plt.show()
        for i in range(self.classes):
            mask_feat = (gt == i).float()
            img_class_mean = (1 - mask_feat) * img_class_mean + torch.sum(img_class_mean * mask_feat, dim=[2, 3],
                                                                          keepdim=True) / (
                                         torch.sum(mask_feat, dim=[2, 3], keepdim=True) + 1e-8) * mask_feat
        # plt.imshow(img_class_mean.detach().cpu()[0][0])
        # plt.show()
        return img_class_mean

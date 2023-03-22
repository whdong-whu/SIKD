import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from utils.criterion import HintLoss, Com_class_mean

criterion_mse = nn.MSELoss(reduce=False)
def model_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ["loss", "tb_dict", "disp_dict"])

    def model_fn(model, data, criterion, perfermance=False, vis=False, device="cuda", epoch=0, num_class=9, imgmean=False, kd=False):
        imgs, gts = data[:2]
        imgs = imgs.to(device)
        gts = gts.to(device)

        if gts.dim()==3:
            gts=torch.unsqueeze(gts,1)
        if imgmean:
            com_class_mean = Com_class_mean(classes=num_class)
            imgs = com_class_mean(imgs, gts.to(device))
        if kd:
            criterion_kd = HintLoss()
            com_class_mean = Com_class_mean(classes=num_class)
            img_mean = com_class_mean(imgs, gts.to(device))
            gts = torch.squeeze(gts, 1)
            net_out = model(image=imgs,img_mean=img_mean, target=gts.long(), criterion=criterion, criterion_kd=criterion_kd, loss_kd_weight=2.0)
            loss = net_out[1]
        else:
            net_out = model(imgs)
            # norm ce loss
            gts = torch.squeeze(gts, 1)
            loss = criterion(net_out[0], gts.long())

        tb_dict = {}
        disp_dict = {}
        tb_dict.update({"loss": loss.item()})
        disp_dict.update({"loss": loss.item()})

        if perfermance:
            gts_ = gts.unsqueeze(1)
            net_out = F.softmax(net_out[0], dim=1)
            _, preds = torch.max(net_out, 1)
            preds = preds.unsqueeze(1)
            cal_subject_level_dice(preds, gts_, tb_dict, num_class)

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_fn


def cal_subject_level_dice(prediction, target, tb_dict, class_num=2):
    '''
    step1: calculate the dice of each category
    step2: remove the dice of the empty category and background, and then calculate the mean of the remaining dices.
    :param prediction: the automated segmentation result, a numpy array with shape of (h, w, d)
    :param target: the ground truth mask, a numpy array with shape of (h, w, d)
    :param class_num: total number of categories
    :return:
    '''
    prediction = prediction.cpu()
    target = target.cpu()
    eps = 1e-8
    dscs = np.zeros((class_num - 1), dtype=np.float32)
    for i in range(1, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0).astype(np.float32)
        prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

        tp = np.sum(prediction_per_class * target_per_class)
        fp = np.sum(prediction_per_class) - tp
        fn = np.sum(target_per_class) - tp
        dsc = 2 * tp / (2 * tp + fp + fn + eps)
        dscs[i - 1] = dsc
    # print(subject_level_dice)
    tb_dict.update({"mean_dice": np.mean(dscs)})


import torch
import torch.nn as nn
import torch.nn.functional as F


class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        return self.forward_train(**kwargs)

class KD_model(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher):
        super(KD_model, self).__init__(student, teacher)


    def forward_train(self, image, img_mean, target, criterion, criterion_kd, loss_kd_weight, criterion_kd_pred=None, **kwargs):
        logits_s, fea_s= self.student(image)

        with torch.no_grad():
            logits_t, fea_t = self.teacher(img_mean)

        loss_ce = criterion(logits_s, target)
        loss_kd = criterion_kd(fea_s, fea_t)
        loss = loss_ce + loss_kd_weight * loss_kd
        if criterion_kd_pred:
            loss += criterion_kd_pred(logits_s, logits_t)
        return logits_s, loss

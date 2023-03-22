from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C


# ============== general training config =====================
__C.TRAIN = edict()

__C.TRAIN.NET = "unet.U_Net"
# __C.TRAIN.NET = "unet.U_Net_teacher" # network without concatenation

__C.TRAIN.LR = 0.001
__C.TRAIN.LR_CLIP = 0.00001
__C.TRAIN.DECAY_STEP_LIST = [60, 100, 150, 180]
__C.TRAIN.LR_DECAY = 0.5

__C.TRAIN.GRAD_NORM_CLIP = 1.0

__C.TRAIN.OPTIMIZER = 'adam'
__C.TRAIN.WEIGHT_DECAY = 0  # "L2 regularization coeff [default: 0.0]"
__C.TRAIN.MOMENTUM = 0.9

# =============== model config ========================
__C.MODEL = edict()

__C.MODEL.SELFEATURE = False #True
__C.MODEL.SHIFT_N = 1
__C.MODEL.AUXSEG = False #True

# ================= dataset config ==========================
__C.DATASET = edict()
__C.DATASET.IMG_CH = 1

__C.DATASET.MEAN = 66.7945
__C.DATASET.STD = 61.2465

__C.DATASET.NUM_CLASS = 4
__C.DATASET.TRAIN_LIST = "/home/whdong/data/medical/MICCAI2021_multi_domain_robustness_datasets/processed_slice/ACDC_xu/train"
__C.DATASET.TEST_LIST = "/home/whdong/data/medical/MICCAI2021_multi_domain_robustness_datasets/processed_slice/ACDC_xu/val"
# __C.DATASET.TRAIN_LIST = "train_path"
# __C.DATASET.TEST_LIST = "test_path"

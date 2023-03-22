import sys
sys.path.append('..')
import torch
import argparse
import numpy as np
from tqdm import tqdm
from medpy import metric
import os
from torch.nn import functional as F
from torch.utils.data import Dataset
import nibabel as nib
from glob import glob
from multiprocessing.dummy import Pool
import shutil
from libs.network import U_Net


parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--data_path', type=str, help='data path',)
parser.add_argument('--checkpoint path', type=str, help='checkpoint path')
parser.add_argument('--temp_path', type=str, help='temp path for saving prediction')
parser.add_argument('--gpu', type=str, default='0', help='gpu id')

class ACDC(Dataset):
    def __init__(self,
                 data_path):
        self.data_path = data_path
        self.data_list = []
        case_list = glob(os.path.join(self.data_path, '*/*'))
        for case in case_list:
            if 'img' in case:
                self.data_list.append(case)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        vol_path = self.data_list[idx]
        seg_path = vol_path.replace('img', 'seg')

        vol = nib.load(vol_path)
        vol = vol.get_fdata()
        seg = nib.load(seg_path)
        seg = seg.get_fdata()
        vol = vol.transpose((2, 1, 0))
        seg = seg.transpose((2, 1, 0))
        vol_norm = vol[None]
        seg = seg[None]

        return {'image': torch.from_numpy(vol_norm.astype(np.float32)),
                'label': torch.from_numpy(seg.astype(np.float32)), 'case_name': vol_path.split('/')[-2]+vol_path.split('/')[-1].split('.')[0]}


def load_data(path):
    data = np.load(path)
    pred, seg = data['pred'], data['mask']
    return pred, seg


def cal_dice_hd(args):
    path = args[0]
    num_class = args[1]
    pred, label = load_data(path)
    metric_list = []

    for i in range(1, num_class):
        metric_list.append(calculate_metric_percase(pred == i, label == i))
    dice_mean, hd_mean = np.mean(metric_list, axis=0)[0], np.mean(metric_list, axis=0)[1]
    print("case {} mean_dice: {}, mean_hd95: {}".format(path.split('/')[-1], dice_mean, hd_mean))

    return metric_list


def pred_acdc(image, label, net, patch_size=[256,256], test_save_path=None, case=None):
    image, label = image.squeeze(0), label.squeeze(0)
    batch = 96
    x, y = image.shape[1], image.shape[2]
    prediction = torch.ones((1, x, y)).cuda()
    for ind in range(0, image.shape[0], batch):
        if ind + batch > image.shape[0]:
            slice = image[ind:image.shape[0], :, :]
        else:
            slice = image[ind:ind + batch, :, :]
        if x != patch_size[0] or y != patch_size[1]:
            input = F.interpolate(input=slice.unsqueeze(1),
                                  size=(patch_size[0], patch_size[1]), mode='bilinear', align_corners=True).cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs[0], dim=1), dim=1)
                if x != patch_size[0] or y != patch_size[1]:
                    pred = F.interpolate(input=out.unsqueeze(1).float(), size=(x, y),
                                         mode='nearest').squeeze(1)
                else:
                    pred = out
                prediction = torch.cat((prediction, pred.long()), dim=0)
    prediction = prediction[1:]
    prediction = prediction.cpu().numpy()
    if test_save_path is not None:
        np.savez_compressed(f'{test_save_path}/{case}.npz', pred=prediction, mask=label)


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0



def dice(model, test_save_path=None):
    print("{} test case".format(len(data_load)))
    model.cuda()
    for i_batch, sampled_batch in tqdm(enumerate(data_load)):
        image, label, case_name = sampled_batch.values()
        if len(case_name) == 1:
            case_name = case_name[0]
        print(case_name)
        pred_acdc(image, label, model, patch_size=[256, 256], test_save_path=test_save_path, case=case_name)


def test_from_dir_multi_process(in_path, num_class=4, processes=10):
    paths = glob(in_path)
    pool = Pool(processes)
    metric = pool.map(cal_dice_hd, [(p, num_class) for p in paths])
    metric_list = np.sum(np.array(metric), axis=0)

    metric_list = metric_list / len(paths)
    for i in range(1, num_class):
        print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    print(in_path)
    return "Testing Finished!"

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    temp_path = args.temp_path
    ckpt_path = args.ckpt_path
    model = U_Net(img_ch=1, num_class=4).cuda()
    if os.path.isfile(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        a, b = model.load_state_dict(checkpoint['model_state'])
        print('unexpected keys:', a)
        print('missing keys:', b)

        print("=> loaded checkpoint '{}'".format(ckpt_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(ckpt_path))
    model.eval()
    pred_path = os.path.join(temp_path, ckpt_path.split('/')[-3])
    if os.path.exists(pred_path):
        shutil.rmtree(pred_path)
        os.makedirs(pred_path)
    else:
        os.makedirs(pred_path)
    num_class = 4
    data_load = ACDC(data_path=args.data_path)
    print('--------saving pred---------')
    dice(model, test_save_path=pred_path)
    print('--------calculating metric----------')
    test_from_dir_multi_process(in_path=os.path.join(pred_path, '*'), num_class=num_class, processes=20)
    print('--------done & remove pred--------')
    print('ckpt_path:', ckpt_path)
    shutil.rmtree(pred_path)

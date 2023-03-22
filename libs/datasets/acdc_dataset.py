import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import os
import json
import numpy as np
from PIL import Image
import h5py

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../'))

class AcdcDataset(Dataset):
    def __init__(self, data_path,  joint_augment=None, augment=None, target_augment=None, split='train'):
        self.joint_augment = joint_augment
        self.augment = augment
        self.target_augment = target_augment
        self.data_path=data_path
        self.data_list = np.load(os.path.join(data_path, '%s_slices.npy' % split))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,index):
        f_name = self.data_list[index]
        case = f_name.split('_')[0]
        npz = np.load(os.path.join(self.data_path, f_name), allow_pickle=True)
        img = npz.get('image')[:,:,None].astype(np.float32)
        gt = npz.get('mask')[:,:,None].astype(np.float32)

        if self.joint_augment is not None:
            img, gt = self.joint_augment(img, gt)
        if self.augment is not None:
            img = self.augment(img)
        if self.target_augment is not None:
            gt = self.target_augment(gt)

        return img, gt





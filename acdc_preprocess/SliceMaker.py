import os
import argparse
import numpy as np
import nibabel as nib
import json

parser = argparse.ArgumentParser(description='Slice Maker')
parser.add_argument('--in_path', type=str, help='input path',)
parser.add_argument('--out_path', type=str, help='output path')
parser.add_argument('--data_json', type=str, help='data json file')
parser.add_argument('--process_num', type=int, default=20)
parser.add_argument('--mode', type=str, default='test')

args = parser.parse_args()

def main():
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    list_path = os.path.join(args.out_path, args.mode)
    if not os.path.exists(list_path):
        os.makedirs(list_path)
    case_list = json.load(open(args.data_json, 'r'))
    seg_path_list = []
    for test_case in case_list:
        case_name = os.listdir(os.path.join(args.in_path, 'patient' + test_case))
        for name in case_name:
            if 'gt' in name:
                img_name = os.path.join(os.path.join(args.in_path, 'patient' + test_case), name.replace('_gt', ''))
                seg_name = os.path.join(os.path.join(args.in_path, 'patient' + test_case), name)
                seg_path_list.append(seg_name)


    result = make_slice(seg_path_list)
    np.save(os.path.join(list_path, '%s_slices.npy' % args.mode), result)


def make_slice(path_lit):
    """
    Cut 3D kits data into 2D slices
    :param path: /*/*.nii.gz
    :return: Slices and Infos
    """
    result = []
    for path in path_lit:
        case, vol, seg = read_data(path)

        for i in range(vol.shape[2]):
            ct_slice = vol[:,:,i]
            mask_slice = seg[:,:,i]
            np.savez_compressed(f'{args.out_path}/{args.mode}/{case}_{i}.npz', image=ct_slice, mask=mask_slice)
            if np.any(mask_slice > 0):
                result.append(f'{case}_{i}.npz')

        print(f'complete making slices of {case}')
    return result

def normalize_minmax_data(image_data):
    """
    # 3D MRI scan is normalized to range between 0 and 1 using min-max normalization.
    Here, the minimum and maximum values are used as 2nd and 98th percentiles respectively from the 3D MRI scan.
    We expect the outliers to be away from the range of [0,1].
    input params :
        image_data : 3D MRI scan to be normalized using min-max normalization
    returns:
        final_image_data : Normalized 3D MRI scan obtained via min-max normalization.
    """
    min_val_2p = np.percentile(image_data, 2)
    max_val_98p = np.percentile(image_data, 98)
    # min-max norm on total 3D volume
    image_data[image_data < min_val_2p] = min_val_2p
    image_data[image_data > max_val_98p] = max_val_98p

    final_image_data = (image_data - min_val_2p) / (1e-10 + max_val_98p - min_val_2p)

    return final_image_data


def read_data(path):
    dir, fname = os.path.split(path)
    case_img_name = fname.replace('_gt', '')
    case = case_img_name.split('.')[0]
    img_path = os.path.join(dir, case_img_name)
    seg = nib.load(path).get_fdata().astype(np.float32)
    vol = nib.load(img_path).get_fdata().astype(np.float32)
    vol = normalize_minmax_data(vol)

    return case, vol, seg



if __name__ == '__main__':
    main()

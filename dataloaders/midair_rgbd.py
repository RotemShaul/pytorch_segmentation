from base import BaseDataSetFuse, BaseDataLoaderFuse
from utils import palette
from glob import glob
import numpy as np
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MidAirRGBDDataset(BaseDataSetFuse):
    def __init__(self, mode='fine', **kwargs):
        self.num_classes = 2
        self.mode = mode
        self.palette = palette.CityScpates_palette
        super(MidAirRGBDDataset, self).__init__(**kwargs)

    def _set_files(self):
        assert (self.mode == 'fine' and self.split in ['train', 'val']) or \
        (self.mode == 'coarse' and self.split in ['train', 'train_extra', 'val'])

        file_list_path = ''
        if self.split in ['train']:
            file_list_path = os.path.join(self.root, 'training.list')
        elif self.split in ['val']:
            file_list_path = os.path.join(self.root, 'validation.list')
        else:
            print("Not train or val, exiting")
            exit(-1)

        f = open(file_list_path, 'r')
        file_list = f.readlines()
        f.close()
        self.files = file_list

    def _load_data(self, index):
        current_file = self.files[index]
        filename = self.root + 'color_left/trajectory_5000/' + current_file[0: len(current_file) - 1]
        rgb_image = Image.open(filename)
        filename = self.root + 'stereo_disparity/trajectory_5000/' + current_file[
                                                                            0: len(current_file) - 5] + 'PNG'
        disp_img = Image.open(filename)
        filename = self.root + 'segmentation/trajectory_5000/' + current_file[0: len(current_file) - 5] + 'PNG'
        seg_label = Image.open(filename)

        rgb_image = np.asarray(rgb_image, np.float32)
        seg_label = np.asarray(seg_label, np.int64)
        ######
        # seg_label[seg_label == 0] = 14 #Give the background class label
        seg_label[seg_label == 2] = -1  # Tree
        seg_label[seg_label >= 0] = 0
        seg_label[seg_label == -1] = 1 #Tree
        ######
        disp_img = np.asarray(disp_img, np.uint16)
        disp_img.dtype = np.float16
        disp_img = disp_img.astype(np.float32)

        image_id = os.path.splitext(os.path.basename(filename))[0]
        return rgb_image, seg_label, image_id, disp_img



class MidAirRGBD(BaseDataLoaderFuse):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, mode='fine', val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.28689529, 0.32513294, 0.28389176]
        self.STD = [0.17613647, 0.18099176, 0.17772235]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = MidAirRGBDDataset(mode=mode, **kwargs)
        super(MidAirRGBD, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)



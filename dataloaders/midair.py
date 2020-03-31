from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ignore_label = 0

ID_TO_TRAINID = {-1: ignore_label,
                 0: ignore_label,
                 1: ignore_label,
                 2: 1,
                 3: ignore_label,
                 4: ignore_label,
                 5: ignore_label,
                 6: ignore_label,
                 7: 0,
                 8: 0,
                 9: ignore_label,
                 10: ignore_label,
                 11: 0,
                 12: 0,
                 13: 0,
                 14: ignore_label,
                 15: ignore_label,
                 16: ignore_label,
                 17: 0,
                 18: ignore_label,
                 19: 0,
                 20: 0,
                 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0,
                 28: 0,
                 29: ignore_label, 30: ignore_label,
                 31: 0, 32: 0, 33: 0}


class MidAirDataset(BaseDataSet):
    def __init__(self, mode='fine', **kwargs):
        self.num_classes = 2
        self.mode = mode
        self.palette = palette.CityScpates_palette
        self.id_to_trainId = ID_TO_TRAINID
        super(MidAirDataset, self).__init__(**kwargs)

    def _set_files(self):
        assert (self.mode == 'fine' and self.split in ['train', 'val']) or \
        (self.mode == 'coarse' and self.split in ['train', 'train_extra', 'val'])

       #SUFIX = '_gtFine_labelIds.png'
       #if self.mode == 'coarse':
       #    img_dir_name = 'leftImg8bit_trainextra' if self.split == 'train_extra' else 'leftImg8bit_trainvaltest'
       #    label_path = os.path.join(self.root, 'gtCoarse', 'gtCoarse', self.split)
       #else:
       #    img_dir_name = 'leftImg8bit_trainvaltest'
       #    label_path = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split)
       #image_path = os.path.join(self.root, img_dir_name, 'leftImg8bit', self.split)
       #assert os.listdir(image_path) == os.listdir(label_path)

       #image_paths, label_paths = [], []
       #for city in os.listdir(image_path):
       #    image_paths.extend(sorted(glob(os.path.join(image_path, city, '*.png'))))
       #    label_paths.extend(sorted(glob(os.path.join(label_path, city, f'*{SUFIX}'))))
       #self.files = list(zip(image_paths, label_paths))

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
        filename = self.parent_path + 'color_left/trajectory_5000/' + current_file[0: len(current_file) - 1]
        rgb_image = Image.open(filename)
        filename = self.parent_path + 'stereo_disparity/trajectory_5000/' + current_file[
                                                                            0: len(current_file) - 5] + 'PNG'
        disp_img = Image.open(filename)
        filename = self.parent_path + 'segmentation/trajectory_5000/' + current_file[0: len(current_file) - 5] + 'PNG'
        seg_label = Image.open(filename)

        rgb_img = np.asarray(rgb_image, np.float32)
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
        return rgb_image, seg_label, image_id

        ####
        #image_path, label_path = self.files[index]
        #image_id = os.path.splitext(os.path.basename(image_path))[0]
        #image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        #label = np.asarray(Image.open(label_path), dtype=np.int32)
        #for k, v in self.id_to_trainId.items():
        #    label[label == k] = #
        #return image, label, image_id



class MidAir(BaseDataLoader):
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

        self.dataset = MidAirDataset(mode=mode, **kwargs)
        super(MidAir, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)



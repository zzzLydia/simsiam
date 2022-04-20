from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image# using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders
    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 is_train,
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.is_train=is_train
        self.interp = Image.ANTIALIAS


        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
# transform
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        
    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        # color = color.crop((0, 160, 1280, 960-160))
        # color = color.resize((512, 256),Image.ANTIALIAS)

        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color
      
    def get_image_path(self, folder, frame_index):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        #print(image_path)
        return image_path

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required
        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
    #introduce color_aug to input dict
#         for k in list(inputs):
#             frame = inputs[k]
#             if "color" in k or "color_n" in k:
#                 n= k
#                 for i in range(self.num_scales):
#                     inputs[n] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k or "color_n" in k:
                n = k
                inputs[n] = self.to_tensor(f)
                inputs[n + "_aug"] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        
#change variable of filename(1 to 2), return 2 pic once
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:
            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
        """
        inputs = {}
#transform
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
#split each line in txt:  eg；day_train_all/0000000315

        line = self.filenames[index].split('/') #line=['day_train_all','0000000315']
        folder = line[0] # folder='day_train_all'

        # if len(line) == 3:
        frame_index = int(line[1]) #frame_index=0000000315
        # else:
        #     frame_index = 0

        #is_train = folder.split('_')[1] # istrain=train


            
        if folder[0] == 'd': # folder=day_train_all
            folder2 = folder + '_fake_night' # folder=day_train_all_fake_night
            flag = 0
        else:
            folder2 = folder + '_fake_day'
            tmp = folder
            folder = folder2
            folder2 = tmp
            flag = 1

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        inputs["color"] = self.get_color(folder, frame_index, do_flip)
        inputs["color_n"] = self.get_color(folder2, frame_index, do_flip)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
            print(color_aug)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
           


 
        return inputs
                
                
                
                
                
            

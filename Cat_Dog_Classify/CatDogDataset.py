import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DogCatDataSet(Dataset):
    def __init__(self, root_dir, image_dir, transform):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.transform = transform
        self.image_list.sort()
        self.labelId = {'cat': 0, 'dog': 1}

    def __getitem__(self, idx):
        global label
        img_name = self.image_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        img = Image.open(img_item_path)
        # img = Image.fromarray(img)
        # img = np.array(img)

        # print(img_name)
        for idx in range(len(img_name)):
            if img_name[idx] == '_':
                label = img_name[0:idx]
                label = self.labelId[label]
                break

        # label = np.array(label)
        label = torch.from_numpy(label)
        # print("target type is {}".format(type(label)))
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_list)

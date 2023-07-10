from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # device = torch.device('cuda:0')
        # path_label = np.array(self.data.loc[index])
        img_gray = imread(self.data.loc[index, 'filename'])
        img_2rgb = np.transpose(gray2rgb(img_gray), axes=(2, 0, 1))
        # img_tensor = torch.from_numpy(img_2rgb).to(device)
        img_tensor = torch.from_numpy(img_2rgb)
        label = torch.from_numpy(np.array(self.data.loc[index, 'crack': 'inactive'], dtype=float))
        if self.mode == 'val':
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                     tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(mean=train_mean, std=train_std)])

        else:
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                     tv.transforms.RandomApply([tv.transforms.RandomRotation((0, 180)),
                                                                                tv.transforms.RandomResizedCrop(
                                                                                    (300, 300))], p=0.5),
                                                     tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(mean=train_mean, std=train_std)])
        return self._transform(img_tensor), label

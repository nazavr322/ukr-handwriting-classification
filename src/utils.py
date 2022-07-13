import os
from typing import Union, Optional

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage

from PIL import Image
from sklearn.model_selection import train_test_split


class HandwritingDataset(Dataset):
    def __init__(self, csv_path: str, transforms=None):
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx) -> tuple[Union[Image.Image, torch.Tensor],
                                        int, str]:
        label, lbl_code, is_uppercase, filename = self.df.iloc[idx]
        full_path = os.path.join('../data/raw', filename)
        img = Image.open(full_path)
        if self.transforms:
            img = self.transforms(img)
        return img, lbl_code, label


def split_train_test(data_path: str,
                     test_size: Union[int, float],
                     random_state: Optional[int] = None) -> None:
    data = pd.read_csv(data_path)
    x = data.drop('label', axis='columns')
    y = data.label
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    x_train.insert(0, 'label', y_train)
    x_train.reset_index(inplace=True, drop=True)
    x_train.to_csv('../data/processed/train_data_v4.csv', index=False)
    x_test.insert(0, 'label', y_test)
    x_test.reset_index(inplace=True, drop=True)
    x_test.to_csv('../data/processed/test_data_v4.csv', index=False)


def add_mnist_to_dset(data, num_samples: int) -> None:
    targets = np.array(data.targets)
    indices = (np.where(targets == cls)[0][:num_samples] for cls in range(10)) 
    filename_template = 'glyphs/{cls}-{count}.png'
    with open('../data/raw/digits.csv', 'w', encoding='utf-8') as f:
        f.write('label,is_uppercase,filename')
        for cls_indices in indices:
            for i, cls_idx in enumerate(cls_indices):
                img, cls = data[cls_idx]
                filename = filename_template.format(cls=cls, count=i)
                img.save(os.path.join('../data/raw', filename))
                f.write(f'\n{cls},{False},{filename}')

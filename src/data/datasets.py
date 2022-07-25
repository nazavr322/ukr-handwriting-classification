import os
from typing import Union, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from src import ROOT_DIR


class HandwritingDataset(Dataset):
    """ Custom dataset to load handwriting samples """
    
    def __init__(self, csv_path: str, transforms=None,
                 src_dir: Optional[str] = None):
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms
        self.src_dir = src_dir if src_dir else 'data/processed'
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(
        self, idx
    ) -> tuple[Union[Image.Image, torch.Tensor], int,  bool]:
        row = self.df.iloc[idx]
        full_path = os.path.join(ROOT_DIR, self.src_dir, row.filename)
        img = Image.open(full_path)
        if self.transforms:
            img = self.transforms(img)
        return img, row.lbl_code, row.is_uppercase


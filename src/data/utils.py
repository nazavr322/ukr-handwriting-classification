import os
from os.path import join
from typing import Union, Optional

import numpy as np
import pandas as pd
import cv2 as cv
import torchvision.transforms as T
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split

from ..features.features import encode_labels
from src import ROOT_DIR


def prepare_mnist(num_samples: int, out_name: str) -> None:
    """
    Prepares `num_samples` samples of each class to be joined with original
    data.
    
    Each sample is saved as an image, and corresponding information is added\
 to the `out_name`.csv file
    """
    if not out_name.endswith('.csv'):
        out_name += '.csv'
    data = MNIST(
        join(ROOT_DIR, 'data/raw'),
        download=True,
        train=True,
        transform=T.Compose([T.Grayscale(3)])
    )
    targets = np.array(data.targets)
    indices = (np.where(targets == cls)[0][:num_samples] for cls in range(10)) 
    with open(
        join(ROOT_DIR, 'data/interim', out_name), 'w', encoding='utf-8'
    ) as f:
        f.write('label,is_uppercase,filename')
        for cls_indices in indices:
            for i, cls_idx in enumerate(cls_indices):
                img, cls = data[cls_idx]
                filename = f'glyphs/{cls}-{i}.png'
                img.save(join(ROOT_DIR, 'data/processed', filename))
                f.write(f'\n{cls},{False},{filename}')


def prepare_glyphs(glyphs_path: str) -> None:
    """ 
    Prepares original glyphs to be compatible with MNIST format.

    Each image is resized to the shape 28x28 and inverted.
    """
    for file in os.scandir(glyphs_path):
        img = cv.imread(file.path)
        img = cv.resize(img, (28, 28))
        img = cv.bitwise_not(img)
        cv.imwrite(join(ROOT_DIR, 'data/processed/glyphs', file.name), img)


def make_dataset(raw_path: str, mnist_path: str, out_path: str) -> None:
    """ Generates a complete dataset """
    raw_df = pd.read_csv(raw_path)
    cols_to_drop = ['transliter_kmu2010', 'name', 'type', 'is_alternate',
                    'top', 'bottom', 'left', 'right', 'height', 'width']
    raw_df.drop(cols_to_drop, axis='columns', inplace=True)
    raw_df.replace({False: 0, True: 1}, inplace=True)
    mnist_df = pd.read_csv(mnist_path)
    completed_df = pd.concat((mnist_df, raw_df), ignore_index=True)
    completed_df = encode_labels(completed_df)
    completed_df.to_csv(out_path, index=False)
    raw_df = encode_labels(raw_df)
    raw_df.to_csv(
        join(ROOT_DIR, 'data/interim/data_cleaned.csv'), index=False
    )


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
    x_train.to_csv(
        join(ROOT_DIR, 'data/processed/train_data.csv'), index=False
    )
    x_test.insert(0, 'label', y_test)
    x_test.reset_index(inplace=True, drop=True)
    x_test.to_csv(join(ROOT_DIR, 'data/processed/test_data.csv'), index=False)


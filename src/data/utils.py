import os

import numpy as np
from torchvision.datasets import MNIST

def parse_mnist(data: MNIST, num_samples: int, out_name: str) -> None:
    """
    Prepares `num_samples` samples of each class to be joined with original
    data.
    
    Each sample is saved as an image, and corresponding information is added\
 to the `out_name`.csv file
    """
    if not out_name.endswith('.csv'):
        out_name += '.csv'
    targets = np.array(data.targets)
    indices = (np.where(targets == cls)[0][:num_samples] for cls in range(10)) 
    with open(f'../data/raw/{out_name}', 'w', encoding='utf-8') as f:
        f.write('label,is_uppercase,filename')
        for cls_indices in indices:
            for i, cls_idx in enumerate(cls_indices):
                img, cls = data[cls_idx]
                filename = f'glyphs/{cls}-{i}.png'
                img.save(os.path.join('../data/raw', filename))
                f.write(f'\n{cls},{False},{filename}')


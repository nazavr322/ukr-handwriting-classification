from argparse import ArgumentParser

import numpy as np
import torchvision.transforms as T
from torchvision.datasets import MNIST


def create_parser() -> ArgumentParser:
    """ Initializes parser """
    parser = ArgumentParser()
    parser.add_argument('raw_dir', help='root dir where MNIST/raw exists')
    parser.add_argument('out_path', help='path to .csv file')
    parser.add_argument('num_samples', type=int,
                        help='number of samples per class')
    return parser


def prepare_mnist(raw_dir: str, out_path: str, num_samples: int, ) -> None:
    """
    Prepares `num_samples` samples of each class to be joined with original
    data.
    
    Each sample is saved as an image, and corresponding information is added\
 to the `out_name`.csv file
    """
    data = MNIST(raw_dir, download=True, train=True, transform=T.Grayscale(3))
    targets = np.array(data.targets)
    indices = (np.where(targets == cls)[0][:num_samples] for cls in range(10)) 
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('label,is_uppercase,filename')
        for cls_indices in indices:
            for i, cls_idx in enumerate(cls_indices):
                img, cls = data[cls_idx]
                filename = f'glyphs/{cls}-{i}.png'
                img.save('data/processed', filename)
                f.write(f'\n{cls},{False},{filename}')


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    prepare_mnist(args.raw_dir, args.out_path, args.num_samples)   


import os
from argparse import ArgumentParser

import cv2 as cv


def create_parser() -> ArgumentParser:
    """ Initializes parser """
    parser = ArgumentParser()
    parser.add_argument('raw_img_dir',
                        help='path to directory where raw images are stored')
    return parser


def prepare_glyphs(raw_img_dir: str) -> None:
    """ 
    Prepares original glyphs to be compatible with MNIST format.

    Each image is resized to the shape 28x28 and inverted.
    """
    for file in os.scandir(raw_img_dir):
        img = cv.imread(file.path)
        img = cv.resize(img, (28, 28))
        img = cv.bitwise_not(img)
        cv.imwrite(os.path.join('data/processed/glyphs', file.name), img)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    prepare_glyphs(args.raw_img_dir)

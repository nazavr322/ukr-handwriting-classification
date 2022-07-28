import os
from argparse import ArgumentParser

import cv2 as cv


def create_parser() -> ArgumentParser:
    """ Initializes parser """
    parser = ArgumentParser()
    parser.add_argument('raw_img_dir',
                        help='path to directory where raw images are stored')
    parser.add_argument('out_img_dir',
                        help='path to directory for processed images')
    return parser


def prepare_glyphs(raw_img_dir: str, out_img_dir: str) -> None:
    """ 
    Prepares original glyphs to be compatible with MNIST format.

    Each image is resized to the shape 28x28 and inverted.
    """
    # create glyphs directory if it doesn't exist
    if not os.path.isdir(out_img_dir):
        os.mkdir(out_img_dir)

    for file in os.scandir(raw_img_dir):
        img = cv.imread(file.path)
        img = cv.resize(img, (28, 28))
        img = cv.bitwise_not(img)
        cv.imwrite(os.path.join(out_img_dir, file.name), img)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    prepare_glyphs(args.raw_img_dir, args.out_img_dir)

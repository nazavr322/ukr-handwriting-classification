import os
from argparse import ArgumentParser

import cv2 as cv
import pandas as pd

from src import ROOT_DIR


def create_parser() -> ArgumentParser:
    """ Initializes parser """
    parser = ArgumentParser()
    parser.add_argument('raw_df', help='path to .csv file with raw data')
    parser.add_argument('raw_img_dir',
                        help='path to directory where raw images are stored')
    parser.add_argument('out_img_dir',
                        help='path to directory for processed images')
    return parser


def prepare_glyphs(
    raw_df: pd.DataFrame, raw_img_dir: str, out_img_dir: str
) -> None:
    """
    Prepares original glyphs to be compatible with MNIST format.
    Each token is cropped, then padded a little bit, resized to the 28x28
    shape and inverted.
    """
    # create glyphs directory if it doesn't exist
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)

    for row in raw_df.itertuples(index=False):
        _, filename = row.filename.split('/')
        image = cv.imread(os.path.join(raw_img_dir, filename))
        image = image[row.left:row.right, row.top:row.bottom]
        image = cv.copyMakeBorder(
            image, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=[255, 255, 255]
        )
        image = cv.resize(image, (28, 28))
        image = cv.bitwise_not(image)
        cv.imwrite(os.path.join(out_img_dir, filename), image)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    raw_img_dir_path = os.path.join(ROOT_DIR, args.raw_img_dir)
    out_img_dir_path = os.path.join(ROOT_DIR, args.out_img_dir)
    raw_df = pd.read_csv(os.path.join(ROOT_DIR, args.raw_df))
    prepare_glyphs(raw_df, raw_img_dir_path, out_img_dir_path)

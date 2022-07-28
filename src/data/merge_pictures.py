import os
import shutil
from argparse import ArgumentParser


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('glyphs_path', help='path to folder where processed glyphs pictures are stored')
    parser.add_argument('mnist_path', help='path to folder where processed mnist pictures are stored')
    return parser


def merge_pictures(glyphs_path: str, mnist_path: str) -> None:
    """ 
    Merges pictures from 2 directories into final directory named
    `data/processed/glyphs`.
    2 directories are then deleted.
    """
    target_dir = 'data/processed/glyphs/'
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    for directory in (glyphs_path, mnist_path):
        for root, _, files in os.walk(directory):
            for filename in files:
                shutil.copy2(os.path.join(root, filename), target_dir)
        shutil.rmtree(directory) 


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    merge_pictures(args.glyphs_path, args.mnist_path)

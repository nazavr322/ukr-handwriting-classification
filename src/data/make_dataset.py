from argparse import ArgumentParser

import pandas as pd

from ..features.features import encode_labels


def create_parser() -> ArgumentParser:
    """ Initializes parser """
    parser = ArgumentParser()
    parser.add_argument('glyphs_path',
                        help='path to the cleaned .csv file with glyphs')
    parser.add_argument('mnist_path',
                        help='path to the raw .csv file with mnist')
    parser.add_argument('out_path', help='path to store final .csv file')
    return parser


def make_dataset(glyphs_path: str, mnist_path: str, out_path: str) -> None:
    """ Generates a complete dataset """
    glyphs_df = pd.read_csv(glyphs_path)
    mnist_df = pd.read_csv(mnist_path)
    completed_df = pd.concat((mnist_df, glyphs_df), ignore_index=True)
    completed_df = encode_labels(completed_df)
    completed_df.to_csv(out_path, index=False)
    

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    make_dataset(args.glyphs_path, args.mnist_path, args.out_path)

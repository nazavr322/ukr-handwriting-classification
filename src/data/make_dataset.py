from argparse import ArgumentParser

import pandas as pd

from ..features.features import encode_labels


def create_parser() -> ArgumentParser:
    """ Initializes parser """
    parser = ArgumentParser()
    parser.add_argument('raw_path', help='path to raw .csv file with glyphs')
    parser.add_argument('mnist_path',
                         help='path to raw .csv file with mnist')
    parser.add_argument('out_path', help='path to final .csv file')
    return parser


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
    raw_df.to_csv('data/interim/data_cleaned.csv', index=False)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    make_dataset(args.raw_path, args.mnist_path, args.out_path)

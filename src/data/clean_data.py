from argparse import ArgumentParser

import pandas as pd


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('raw_path', help='path to the .csv file with raw data')
    parser.add_argument('out_path',
                        help='path to the .csv file to store cleaned data')
    return parser


def clean_data(raw_path: str, out_path: str) -> None:
    raw_df = pd.read_csv(raw_path)
    cols_to_drop = ['transliter_kmu2010', 'name', 'type', 'is_alternate',
                    'top', 'bottom', 'left', 'right', 'height', 'width']
    raw_df.drop(cols_to_drop, axis='columns', inplace=True)
    raw_df.replace({False: 0, True: 1}, inplace=True)
    raw_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    clean_data(args.raw_path, args.out_path)

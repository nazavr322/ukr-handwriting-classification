import os
from typing import Union, Optional
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split


def create_parser() -> ArgumentParser:
    """ Initializes parser """
    parser = ArgumentParser()
    parser.add_argument('data_path', help='path to .csv file')
    parser.add_argument('test_size', nargs='?', type=int, default=300,
                        help='amount of samples in test subset')
    parser.add_argument('--random_state', type=int, default=1,
                        help='random state for shuffling')
    return parser


def split_train_test(data_path: str,
                     test_size: Union[int, float],
                     random_state: Optional[int] = None) -> None:
    """ Splits data into train and test """
    data = pd.read_csv(data_path)
    x = data.drop('label', axis='columns')
    y = data.label

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    target_dir = os.path.split(data_path)[0]

    x_train.insert(0, 'label', y_train)
    x_train.reset_index(inplace=True, drop=True)
    x_train.to_csv(os.path.join(target_dir, 'train_data.csv'), index=False)
    x_test.insert(0, 'label', y_test)
    x_test.reset_index(inplace=True, drop=True)
    x_test.to_csv(os.path.join(target_dir, 'test_data.csv'), index=False)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    split_train_test(args.data_path, args.test_size, args.random_state)


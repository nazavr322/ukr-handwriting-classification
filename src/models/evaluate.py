import os
from argparse import ArgumentParser

import mlflow
import numpy as np
import torchvision.transforms as T
from torch import load, cuda, device
from torch.utils.data import DataLoader

from src import ROOT_DIR
from .models import HandwritingClassifier
from .functional import evaluate, get_confusion_matrix
from ..data.datasets import HandwritingDataset


MEAN = HandwritingClassifier._mean
STD = HandwritingClassifier._std
# initialize device
DEVICE = device('cuda') if cuda.is_available() else device('cpu')


def create_parser() -> ArgumentParser:
    """Initializes parser"""
    parser = ArgumentParser()
    parser.add_argument('test_path', help='.csv file with test data')
    parser.add_argument(
        'model_weights_path',
        help='.pt file with the weights of a Multi-Output CNN',
    )
    parser.add_argument(
        'out_fig_path',
        help='path to a directory where figure artifacts will be stored',
    )
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()  # parse cmd arguments
    
    # initialize model
    model = HandwritingClassifier()
    model.load_state_dict(
        load(os.path.join(ROOT_DIR, args.model_weights_path)),
        strict=False,
    )
    model.to(DEVICE)

    # initialize test dataset
    tfs = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    dataset = HandwritingDataset(os.path.join(ROOT_DIR, args.test_path), tfs)

    # initialize dataloader
    test_loader = DataLoader(dataset)

    # set mlflow experiment
    mlflow.set_experiment('Multi-output CNN')

    # start evaluation run
    with mlflow.start_run(run_name='Evaluation') as run:
        width = os.get_terminal_size()[0]  # get terminal width
        print(
            f'{"="*width}\n{"Evaluation started".center(width)}\n{"="*width}\n'
        )
        
        # compute accuracies
        lbl_acc, is_upp_acc, preds = evaluate(model, test_loader, DEVICE)
        acc_msg = 'Accuracy of a {} classification on a test dataset = {:.2%}'
        print(acc_msg.format('label', lbl_acc))
        print(acc_msg.format('case', is_upp_acc))

        # log evaluation metrics
        mlflow.log_metric('Label accuracy', lbl_acc)
        mlflow.log_metric('Is upp accuracy', is_upp_acc)

        # create array of true labels
        gt = np.array([(x.item(), y.item()) for _, x, y in test_loader])
        
        # create and log confusion matrices
        labels = list('0123456789абвгґдеєжзиіїйклмнопрстуфхцчшщьюя')
        lbl_cm = get_confusion_matrix(
            gt[:, 0],
            preds[:, 0],
            labels,
            'Confusion matrix for label classification',
            figsize=(14, 14),
            fontsize=22,
            dpi=300,
        )
        full_path = os.path.join(ROOT_DIR, args.out_fig_path, 'lbl_cm.png')
        print('\nConfusion matrix is saved at', full_path)
        mlflow.log_figure(lbl_cm.figure_, full_path)

        is_upp_cm = get_confusion_matrix(
            gt[:, 1],
            preds[:, 1],
            ('lowercase', 'uppercase'),
            'Confusion matrix for case determination',
            dpi=300,
        )
        full_path = os.path.join(ROOT_DIR, args.out_fig_path, 'is_upp_cm.png')
        print('Confusion matrix is saved at', full_path)
        mlflow.log_figure(is_upp_cm.figure_, full_path)

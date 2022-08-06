import os
import json
from argparse import ArgumentParser

import mlflow
import torch.optim as optim
from torch import load, save, cuda, device
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torchvision import transforms as T

from .models import HandwritingClassifier
from .functional import initialize_loaders, train_model
from ..data.datasets import HandwritingDataset
from src import ROOT_DIR


MEAN = HandwritingClassifier._mean
STD = HandwritingClassifier._std
# initialize device
DEVICE = device('cuda') if cuda.is_available() else device('cpu')


def create_parser() -> ArgumentParser:
    """Initializes parser"""
    parser = ArgumentParser()
    parser.add_argument('train_path', help='.csv file with the training data')
    parser.add_argument(
        'model_weights_path',
        help='.pt file with the weights of a model pretrained on MNIST',
    )
    parser.add_argument(
        'params_path', help='.json file with hyperparameters values'
    )
    parser.add_argument(
        'out_weights_path', help='where to store trained model weights'
    )
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()  # parse cmd arguments

    width = os.get_terminal_size()[0]  # get terminal width
    title_template = '=' * width + '\n' + '{}' + '=' * width + '\n'
    print(title_template.format('Training started'.center(width)))

    # initialize model
    model = HandwritingClassifier()
    model.load_state_dict(
        load(os.path.join(ROOT_DIR, args.model_weights_path)),
        strict=False,
    )
    model.to(DEVICE)

    # initialize train dataset
    train_tfs = T.Compose(
        [
            T.RandomRotation(30),
            T.RandomAffine(0, (0.1, 0.1)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    train_dset = HandwritingDataset(
        os.path.join(ROOT_DIR, args.train_path), train_tfs
    )

    # initialize test dataset
    test_tfs = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    test_dset = HandwritingDataset(
        os.path.join(ROOT_DIR, args.test_path), test_tfs
    )

    # read hyperparameters from .json file
    with open(os.path.join(ROOT_DIR, args.params_path), 'r') as f:
        params = json.load(f)

    # set mlflow experiment
    mlflow.set_experiment('Multi-output CNN')

    # start training run
    with mlflow.start_run(run_name='Training') as run:
        # log hyperparameters
        mlflow.log_params(params)

        # initialize data loaders
        BATCH_SIZE = params['batch_size']
        train_loader, val_loader = initialize_loaders(train_dset, BATCH_SIZE)

        # initialize other hyperparameters
        NUM_EPOCHS = params['num_epochs']
        LR = params['learning_rate']
        REG = params['weight_decay']
        GAMMA = params['factor']
        PAT = params['patience']

        # initialize loss functions
        criterion1 = CrossEntropyLoss().to(DEVICE)
        criterion2 = BCEWithLogitsLoss().to(DEVICE)
        losses = (criterion1, criterion2)

        # initialize optimizer and lr-scheduler
        optimizer = optim.SGD(
            model.parameters(), lr=LR, momentum=0.9, weight_decay=REG
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=GAMMA, patience=PAT
        )

        # train model
        train_results = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            losses,
            NUM_EPOCHS,
            DEVICE,
            scheduler,
        )

        # log training and validation metrics
        metrics = (
            'Training loss',
            'Validation loss',
            'Validation label accuracy',
            'Validation is_upp accuracy',
        )
        for metric, history in zip(metrics, train_results):
            for epoch, value in enumerate(history):
                mlflow.log_metric(metric, value, epoch)

        # save trained model
        out_path = os.path.join(ROOT_DIR, args.out_weights_path)
        save(model.state_dict(), out_path)
        print(f'\nYour model is saved at {out_path}\n')

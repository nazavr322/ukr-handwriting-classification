import os
import json
from argparse import ArgumentParser

import mlflow
import numpy as np
import torch.optim as optim
from torch import save, load, cuda, device
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src import ROOT_DIR
from .models import HandwritingClassifier
from .functional import (
    initialize_loaders,
    train_model,
    evaluate,
    save_confusion_matrix,
)
from ..data.datasets import HandwritingDataset


MEAN = HandwritingClassifier._mean
STD = HandwritingClassifier._std
# initialize device
DEVICE = device('cuda') if cuda.is_available() else device('cpu')


def create_parser() -> ArgumentParser:
    """Initializes parser"""
    parser = ArgumentParser()
    parser.add_argument('train_path', help='.csv file with the training data')
    parser.add_argument('test_path', help='.csv file with test data')
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
    parser.add_argument(
        'out_fig_path',
        help='path to directory where output images will be stored',
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

    # start mlflow run
    with mlflow.start_run() as run:
        # log hyperparameters
        mlflow.log_params(params)
        
        # initialize data loaders
        BATCH_SIZE = params['batch_size']
        train_loader, val_loader = initialize_loaders(train_dset, BATCH_SIZE)
        test_loader = DataLoader(test_dset)

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
        train_results = train_model(model, train_loader, val_loader,
                                    optimizer, losses, NUM_EPOCHS, DEVICE,
                                    scheduler)
        
        # log training and validation metrics
        metrics = ('Training loss', 'Validation loss',
                   'Validation label accuracy', 'Validation is_upp accuracy')
        for metric, history in zip(metrics, train_results):
            for value in history:
                mlflow.log_metric(metric, value)


        # save trained model
        out_path = os.path.join(ROOT_DIR, args.out_weights_path)
        save(model.state_dict(), out_path)
        print(f'\nYour model is saved at {out_path}\n')

        # start model evaluation
        print(title_template.format('Evaluation started'.center(width)))
        lbl_acc, is_upp_acc, preds = evaluate(model, test_loader, DEVICE)
        acc_msg = 'Accuracy of a {} classification on a test dataset = {:.2%}'
        print(acc_msg.format('label', lbl_acc))
        print(acc_msg.format('case', is_upp_acc))

        # log metrics on test dataset
        mlflow.log_metric('Label accuracy', lbl_acc)
        mlflow.log_metric('Is upp acuuracy', is_upp_acc)

    # create array of true labels
    ground_truth = np.array([(x.item(), y.item()) for _, x, y in test_loader])

    # save confusion matrices
    labels = list('0123456789абвгґдеєжзиіїйклмнопрстуфхцчшщьюя')
    full_path = os.path.join(ROOT_DIR, args.out_fig_path, 'lbl_cm.png')
    save_confusion_matrix(
        full_path,
        ground_truth[:, 0],
        preds[:, 0],
        labels,
        'Confusion matrix for label classification',
        figsize=(14, 14),
        fontsize=22,
        dpi=300,
    )
    print('\nConfusion matrix is saved at', full_path)
    full_path = os.path.join(ROOT_DIR, args.out_fig_path, 'is_upp_cm.png')
    save_confusion_matrix(
        full_path,
        ground_truth[:, 1],
        preds[:, 1],
        ('lowercase', 'uppercase'),
        'Confusion matrix for case determination',
        dpi=300,
    )
    print('Confusion matrix is saved at', full_path)

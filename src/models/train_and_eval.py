import os
import json
import warnings
from argparse import ArgumentParser

import mlflow
import numpy as np
import torchvision.transforms as T
import torch.optim as optim
from torch import load, cuda, device
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader

from src import ROOT_DIR
from .models import HandwritingClassifier
from .functional import (
    initialize_loaders,
    train_model,
    evaluate_model,
    get_confusion_matrix,
    predict,
)
from ..data.datasets import HandwritingDataset


def create_parser() -> ArgumentParser:
    """Initializes parser"""
    parser = ArgumentParser()
    parser.add_argument('train_path', help='.csv file with the training data')
    parser.add_argument('test_path', help='.csv file with the test data')
    parser.add_argument(
        'weights_path',
        help='.pt file with the weights of a model pretrained on MNIST',
    )
    parser.add_argument(
        'params_path', help='.json file with hyperparameters values'
    )
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()  # parse cmd arguments

    # initialize device
    DEVICE = device('cuda') if cuda.is_available() else device('cpu')
    # initalize precomputed mean and standart deviation
    MEAN = HandwritingClassifier._mean
    STD = HandwritingClassifier._std

    TERM_WIDTH = os.get_terminal_size()[0]  # get terminal width
    # prepare template for displayed messages
    title_template = '=' * TERM_WIDTH + '\n{}\n' + '=' * TERM_WIDTH + '\n'

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

    # read hyperparameters from .json file
    with open(os.path.join(ROOT_DIR, args.params_path), 'r') as f:
        params = json.load(f)

    # initialize model
    model = HandwritingClassifier()
    model.load_state_dict(
        load(os.path.join(ROOT_DIR, args.weights_path)),
        strict=False,
    )
    model.to(DEVICE)

    # set mlflow tracking uri
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
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

        # start training
        print(title_template.format('Training started'.center(TERM_WIDTH)))
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

    # initialize test dataset
    test_tfs = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    test_dset = HandwritingDataset(
        os.path.join(ROOT_DIR, args.test_path), test_tfs
    )
    # initialize dataloader
    test_loader = DataLoader(test_dset)

    # set accuracy threshold for label classification
    ACC_THRESH = 0.91

    # start evaluation run
    with mlflow.start_run(run_name='Evaluation') as run:
        print(
            '\n'
            + title_template.format('Evaluation started'.center(TERM_WIDTH))
        )
        # compute accuracies
        lbl_acc, is_upp_acc, preds = evaluate_model(model, test_loader, DEVICE)
        acc_msg = 'Accuracy of a {} classification on a test dataset = {:.2%}'
        print(acc_msg.format('label', lbl_acc))
        print(acc_msg.format('case', is_upp_acc) + '\n')

        # log evaluation metrics
        mlflow.log_metric('Label accuracy', lbl_acc)
        mlflow.log_metric('Is upp accuracy', is_upp_acc)

        # create array of true labels
        gt = np.array(
            [(lbl.item(), is_upp.item()) for _, lbl, is_upp in test_loader]
        )

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
        mlflow.log_figure(lbl_cm.figure_, 'figures/lbl_cm.png')
        print('Confusion matrix successfully logged!')

        is_upp_cm = get_confusion_matrix(
            gt[:, 1],
            preds[:, 1],
            ('lowercase', 'uppercase'),
            'Confusion matrix for case classification',
            dpi=300,
        )
        mlflow.log_figure(is_upp_cm.figure_, 'figures/is_upp_cm.png')
        print('Confusion matrix successfully logged!')

        if lbl_acc >= ACC_THRESH:
            # get sample of model input and unprocessed output
            inp_tensor = train_dset[0][0].unsqueeze(0).to(DEVICE)
            outs = [
                p.cpu().detach().numpy() for p in predict(model, inp_tensor)
            ]
            # create model signature
            np_inp_tensor = inp_tensor.cpu().detach().numpy()
            signature = mlflow.models.infer_signature(
                {'image': np_inp_tensor},
                {'label_probs': outs[0], 'is_upp_prob': outs[1]},
            )

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # log model
                mlflow.pytorch.log_model(
                    model,
                    'multi_output_cnn',
                    signature=signature,
                    input_example=np_inp_tensor,
                )
            print('Your model successfully logged')
        else:
            print(
                'Your model is not logged because accuracy of label '
                f'classification ({lbl_acc}) is lower than accuracy '
                f'threshold ({ACC_THRESH})'
            )

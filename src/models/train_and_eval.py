import os
import json
from argparse import ArgumentParser
from itertools import chain

import mlflow
import numpy as np
import torchvision.transforms as T
import torch.optim as optim
from torch import load, cuda, device
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from src import ROOT_DIR
from .models import HandwritingClassifier
from .functional import *
from ..data.datasets import HandwritingDataset


def create_parser() -> ArgumentParser:
    """Initializes parser"""
    parser = ArgumentParser()
    parser.add_argument('train_path', help='.csv file with the training data')
    parser.add_argument('test_path', help='.csv file with the test data')
    parser.add_argument(
        'params_path', help='.json file with hyperparameters values'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--mnist_weights',
        help='.pt file with the weights of a model pretrained on MNIST',
    )
    group.add_argument(
        '--head_weights',
        help='.pth file with the weights of a MNIST model with 2 heads',
    )
    parser.add_argument(
        '--experiment-name',
        default='Multi-Output CNN',
        help='MLFlow expertiment name',
    )
    return parser


if __name__ == '__main__':
    # initialize constants

    # initialize device
    DEVICE = device('cuda') if cuda.is_available() else device('cpu')

    # initialize precomputed mean and standard deviation
    MEAN = HandwritingClassifier._mean
    STD = HandwritingClassifier._std

    # initialize train/val metric names
    METRICS = (
        'Training loss',
        'Validation loss',
        'Validation label accuracy',
        'Validation is_upp accuracy',
    )

    # message to print when displaying accuracies
    ACC_MSG = 'Accuracy of a {} classification on a test dataset = {:.2%}'

    # message to show when model is not logged
    MODEL_NOT_LOGGED_MSG = (
        'Your model is not logged because accuracy of label '
        'classification ({}) is lower than accuracy '
        'threshold ({})'
    )

    # parse cmd arguments
    parser = create_parser()
    args = parser.parse_args()

    load_dotenv()  # load environmental variables

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
    test_dset = HandwritingDataset(
        os.path.join(ROOT_DIR, args.test_path),
        T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]),
    )

    # read hyperparameters from .json file
    with open(os.path.join(ROOT_DIR, args.params_path), 'r') as f:
        pretrain_params, finetune_params = json.load(f).values()

    # set mlflow experiment
    mlflow.set_experiment(args.experiment_name)

    if args.head_weights is None:
        # initialize model
        model = HandwritingClassifier()
        model.load_state_dict(
            load(os.path.join(ROOT_DIR, args.mnist_weights)), strict=False
        )

        # freeze pretrained model
        freeze_model(model)
        model.to(DEVICE)

        # start pre-training run
        with mlflow.start_run(run_name='Pre-training') as run:
            # log hyperparameters
            mlflow.log_params(pretrain_params)

            # initialize data loaders
            BATCH_SIZE = pretrain_params['batch_size']
            train_loader, val_loader = initialize_loaders(
                train_dset, BATCH_SIZE
            )

            # initialize other hyperparameters
            NUM_EPOCHS = pretrain_params['num_epochs']
            LR = pretrain_params['learning_rate']
            REG = pretrain_params['weight_decay']
            GAMMA = pretrain_params['factor']
            PAT = pretrain_params['patience']

            # initialize loss functions
            criterion1 = CrossEntropyLoss().to(DEVICE)
            criterion2 = BCEWithLogitsLoss().to(DEVICE)

            # initialize optimizer and lr-scheduler
            heads = chain(
                model.token_classifier.parameters(),
                model.is_upp_classifier.parameters(),
            )
            optimizer = optim.Adam(heads, lr=LR, weight_decay=REG)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=GAMMA, patience=PAT
            )

            # start pre-training
            print_title('Pre-Training started', decorator='+')
            train_results = train_model(
                model,
                train_loader,
                val_loader,
                optimizer,
                (criterion1, criterion2),
                NUM_EPOCHS,
                DEVICE,
                scheduler,
            )

            # log training and validation metrics
            for metric, history in zip(METRICS, train_results):
                for epoch, value in enumerate(history):
                    mlflow.log_metric(metric, value, epoch)

            # initialize test dataloader
            test_loader = DataLoader(test_dset)

            # compute accuracies
            lbl_acc, is_upp_acc, preds = evaluate_model(
                model, test_loader, DEVICE
            )
            print(ACC_MSG.format('label', lbl_acc))
            print(ACC_MSG.format('case', is_upp_acc) + '\n')

            # log evaluation metrics
            mlflow.log_metric('Label accuracy', lbl_acc)
            mlflow.log_metric('Is upp accuracy', is_upp_acc)

            # set accuracy threshold for label classification
            ACC_THRESH = 0.81

            # log model
            if lbl_acc >= ACC_THRESH:
                log_model(model, 'pretrained_heads')
            else:
                print(MODEL_NOT_LOGGED_MSG.format(lbl_acc, ACC_THRESH))

            # unfreeze model for further fine-tuning
            unfreeze_model(model)
    else:
        # load and unfreeze model
        model = load(os.path.join(ROOT_DIR, args.head_weights))
        unfreeze_model(model)

    # start training run
    with mlflow.start_run(run_name='Training') as run:
        # log hyperparameters
        mlflow.log_params(finetune_params)

        # initialize data loaders
        BATCH_SIZE = finetune_params['batch_size']
        train_loader, val_loader = initialize_loaders(train_dset, BATCH_SIZE)

        # initialize other hyperparameters
        NUM_EPOCHS = finetune_params['num_epochs']
        LR = finetune_params['learning_rate']
        REG = finetune_params['weight_decay']
        GAMMA = finetune_params['factor']
        PAT = finetune_params['patience']

        # initialize loss functions
        criterion1 = CrossEntropyLoss().to(DEVICE)
        criterion2 = BCEWithLogitsLoss().to(DEVICE)

        # initialize optimizer and lr-scheduler
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=REG)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=GAMMA, patience=PAT
        )

        # start training
        print_title('Training started')
        train_results = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            (criterion1, criterion2),
            NUM_EPOCHS,
            DEVICE,
            scheduler,
        )

        # log training and validation metrics
        for metric, history in zip(METRICS, train_results):
            for epoch, value in enumerate(history):
                mlflow.log_metric(metric, value, epoch)

    # initialize test dataloader
    test_loader = DataLoader(test_dset)

    # set accuracy threshold for label classification
    ACC_THRESH = 0.93

    # start evaluation run
    with mlflow.start_run(run_name='Evaluation') as run:
        print_title('Evaluation started', '*')
        # compute accuracies
        lbl_acc, is_upp_acc, preds = evaluate_model(model, test_loader, DEVICE)
        print(ACC_MSG.format('label', lbl_acc))
        print(ACC_MSG.format('case', is_upp_acc) + '\n')

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
            # get the sample of model input and unprocessed output
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

            # log model
            log_model(model, 'multi_output_cnn', signature, np_inp_tensor)
        else:
            print(MODEL_NOT_LOGGED_MSG.format(lbl_acc, ACC_THRESH))

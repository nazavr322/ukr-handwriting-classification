import os
import json
from argparse import ArgumentParser

import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms as T

from src import ROOT_DIR
from .models import HandwritingClassifier
from ..data.datasets import HandwritingDataset


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


def initialize_dataset(data_path: str) -> HandwritingDataset:
    """Initializes dataset"""
    MEAN = HandwritingClassifier._mean
    STD = HandwritingClassifier._std
    transforms = T.Compose(
        [
            T.RandomRotation(30),
            T.RandomAffine(0, (0.1, 0.1)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ]
    )
    return HandwritingDataset(os.path.join(ROOT_DIR, data_path), transforms)


def initialize_loaders(
    dataset, batch_size: int = 64, val_size: int = 100
) -> tuple[DataLoader, DataLoader]:
    """Initializes train and validation dataloaders"""
    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[val_size:], indices[:val_size]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, sampler=val_sampler)
    return train_loader, val_loader


def compute_accuracy(prediction, ground_truth) -> float:
    """Computes accuracy between prediction and ground_truth"""
    correct_count = torch.sum(prediction == ground_truth).item()
    return correct_count / len(ground_truth)


def validate_model(model, losses, loader, device) -> tuple[float, ...]:
    """Validates model based on 2 metrics"""
    model.eval()
    loss_acum = 0
    lbl_acc = 0
    is_upp_acc = 0

    for i, (x, *y) in enumerate(loader):
        x_gpu = x.to(device)
        y[1] = y[1].unsqueeze(1).float()
        y_gpu = tuple(target.to(device) for target in y)

        prediction = model(x_gpu)
        loss_value = sum(
            loss(out, targ)
            for loss, out, targ in zip(losses, prediction, y_gpu)
        )

        loss_acum += loss_value.item()
        labels = torch.argmax(prediction[0], 1)
        lbl_acc += compute_accuracy(labels, y_gpu[0])
        is_upp = 0 if prediction[1].item() < 0.5 else 1
        is_upp_acc += compute_accuracy(is_upp, y_gpu[1])
    return loss_acum / i, lbl_acc / i, is_upp_acc / i


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    losses,
    num_epochs: int,
    device,
    scheduler=None,
) -> tuple[list[float], ...]:
    """Performs a full training process"""
    t_loss_history = []
    v_loss_history = []
    lbl_acc_history = []
    is_upp_acc_history = []

    for epoch in range(num_epochs):
        model.train()

        loss_acum = 0
        for i, (x, *y) in enumerate(train_loader):
            x_gpu = x.to(device)
            y[1] = y[1].unsqueeze(1).float()
            y_gpu = tuple(target.to(device) for target in y)

            prediction = model(x_gpu)
            loss_value = sum(
                loss(out, targ)
                for loss, out, targ in zip(losses, prediction, y_gpu)
            )

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_acum += loss_value.item()
        epoch_loss = loss_acum / i
        val_loss, lbl_acc, is_upp_acc = validate_model(
            model, losses, val_loader, device
        )

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        t_loss_history.append(epoch_loss)
        v_loss_history.append(val_loss)
        lbl_acc_history.append(lbl_acc)
        is_upp_acc_history.append(is_upp_acc)

        print(
            f'{epoch + 1}. Loss = {epoch_loss:.6f}; Val loss = {val_loss:.6f}'
        )
        print(f'Label accuracy = {lbl_acc}; Is_upper accuracy = {is_upp_acc}')
    return t_loss_history, v_loss_history, lbl_acc_history, is_upp_acc_history


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()   # read cmd arguments

    # initialize device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # initialize model
    model = HandwritingClassifier()
    model.load_state_dict(
        torch.load(os.path.join(ROOT_DIR, args.model_weights_path)),
        strict=False
    )
    model.to(device)

    # initialize dataset
    dataset = initialize_dataset(args.train_path)

    # read hyperparameters from .json file
    with open(os.path.join(ROOT_DIR, args.params_path), 'r') as f:
        params = json.load(f)

    # initialize data loaders
    BATCH_SIZE = params['batch_size']
    train_loader, val_loader = initialize_loaders(dataset, BATCH_SIZE)

    # initialize hyperparameters
    NUM_EPOCHS = params['num_epochs']
    LR = params['learning_rate']
    REG = params['weight_decay']
    GAMMA = params['factor']
    PAT = params['patience']

    # initialize loss functions
    criterion1 = CrossEntropyLoss().to(device)
    criterion2 = BCEWithLogitsLoss().to(device)
    losses = (criterion1, criterion2)

    # initialize optimizer and lr-scheduler
    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=0.9, weight_decay=REG
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=GAMMA, patience=PAT
    )
    
    # get terminal width
    width = os.get_terminal_size()[0]
    print(
        f'{"=" * width}\n{"Training started".center(width)}\n{"=" * width}\n'
    )

    # train model
    train_model(model, train_loader, val_loader, optimizer, losses,
                NUM_EPOCHS, device, scheduler)

    # save trained model
    torch.save(
        model.state_dict(), os.path.join(ROOT_DIR, args.out_weights_path)
    )

    print('\nYour model is saved!')

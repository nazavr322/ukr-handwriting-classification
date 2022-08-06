import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import ConfusionMatrixDisplay


def initialize_loaders(
    dataset, batch_size: int = 64, val_size: int = 100
) -> tuple[DataLoader, ...]:
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
            if isinstance(scheduler, ReduceLROnPlateau):
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


def evaluate(model, loader, device) -> tuple[float, float, np.ndarray]:
    """
    Evaluates model by computing accuracies for 2 targets. Also returns numpy
    array of model predictions.
    """
    model.eval()
    preds = []

    lbl_acc = 0
    is_upp_acc = 0
    for i, (x, *y) in enumerate(loader):
        x_gpu = x.to(device)
        y_gpu = tuple(target.to(device) for target in y)

        prediction = model(x_gpu)

        label = torch.argmax(prediction[0], 1)
        is_upp = 0 if prediction[1] < 0.5 else 1
        preds.append((label.item(), is_upp))

        lbl_acc += compute_accuracy(label, y_gpu[0])
        is_upp_acc += compute_accuracy(is_upp, y_gpu[1])
    return lbl_acc / i, is_upp_acc / i, np.array(preds)


def get_confusion_matrix(
    gt, preds, disp_labels, title: str, **kwargs
) -> ConfusionMatrixDisplay:
    """
    Returns confusion matrix between `gt` and `preds` as a 
    `ConfusionMatrixDisplay` object.
    """
    cm = ConfusionMatrixDisplay.from_predictions(
        gt, preds, display_labels=disp_labels
    )

    cm.ax_.set_title(title, fontsize=kwargs.get('fontsize', 'large'))
    figsize = kwargs.get('figsize')
    if figsize:
        cm.figure_.set_size_inches(figsize)

    cm.figure_.set_dpi(kwargs.get('dpi', 100))
    return cm

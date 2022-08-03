import torch
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .train import compute_accuracy


def evaluate(model, loader, device) -> tuple[float, float, np.array]:
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


def get_confusion_matrix(ground_truth, predictions) -> None:

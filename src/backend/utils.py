import mlflow.pytorch
import numpy as np
import cv2 as cv
from fastapi import UploadFile
from torch import Tensor, from_numpy, device

from ..models.models import HandwritingClassifier


class InferenceModel:
    """
    Wrapper around PyTorch model to load it from MLFlow model registry and
    make predictions conviniently.
    """
    def __init__(self):
        model_uri = 'models:/Multi-Output CNN/Production'
        kwargs = {'map_location': device('cpu')}
        self.model = mlflow.pytorch.load_model(model_uri, kwargs=kwargs)
    
    def predict(self, images: Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Returns raw model predictions"""
        self.model.eval()
        lbl_logits, is_upp_logits = self.model(images)
        return lbl_logits.numpy(), is_upp_logits.numpy()


def read_img(img_file: UploadFile) -> Tensor:
    """
    Reads image from buffer and returns it as `torch.Tensor` that you can pass
    directly to a model.
    """
    # read image from buffer
    image = np.frombuffer(img_file.file.read(), np.uint8)
    image = cv.imdecode(image, cv.IMREAD_UNCHANGED)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (28, 28))
    image = cv.bitwise_not(image)  # invert
    image = np.float32(image / 255)  # clip to [0, 1] range
    # normalize
    mean, std = HandwritingClassifier._mean[0], HandwritingClassifier._std[0]
    image = (image - mean) / std
    return from_numpy(image.transpose(2, 0, 1)[np.newaxis]) 


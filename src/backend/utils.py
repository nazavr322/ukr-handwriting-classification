import mlflow.pytorch
import numpy as np
import cv2 as cv
from fastapi import UploadFile
from torch import Tensor, from_numpy, device

from ..models.models import HandwritingClassifier

    
MEAN, STD = HandwritingClassifier._mean[0], HandwritingClassifier._std[0]


class InferenceModel:
    """
    Wrapper around PyTorch model to load it from MLFlow model registry and
    make predictions conviniently.
    """
    def __init__(self):
        model_uri = 'models:/Multi-Output CNN/Production'
        kwargs = {'map_location': device('cpu')}
        self.model = mlflow.pytorch.load_model(model_uri, **kwargs)
    
    def predict(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Returns raw model predictions"""
        self.model.eval()
        return self.model(images)


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
    image = (image - MEAN) / STD
    return from_numpy(image.transpose(2, 0, 1)[np.newaxis]) 


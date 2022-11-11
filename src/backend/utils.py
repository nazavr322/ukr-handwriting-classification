import uuid
from typing import Optional

import mlflow.pytorch
import numpy as np
import boto3
import cv2 as cv
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


def read_img(byte_img: bytes) -> Tensor:
    """
    Reads image from buffer and returns it as `torch.Tensor` that you can pass
    directly to a model.
    """
    # read image from buffer
    image = np.frombuffer(byte_img, np.uint8)
    image = cv.imdecode(image, cv.IMREAD_UNCHANGED)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (28, 28))
    image = cv.bitwise_not(image)  # invert
    image = np.float32(image / 255)  # clip to [0, 1] range
    # normalize
    image = (image - MEAN) / STD
    return from_numpy(image.transpose(2, 0, 1)[np.newaxis])


def send_to_bucket(
    byte_img: bytes,
    bucket_name: str,
    client_args: dict,
    filename: Optional[str] = None,
) -> None:
    """
    Uploads image as bytes to S3 bucket named `bucket_name`. If
    filename is not specified a random UUID4 is generated.
    """
    file_name = filename if filename is not None else f'{uuid.uuid4()}.png'
    s3 = boto3.resource('s3', **client_args)
    bucket = s3.Bucket(bucket_name)
    bucket.put_object(Body=byte_img, Key=file_name, ContentType='image/png')

import os

from boto3.session import Config
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel, conlist

from .utils import read_img, InferenceModel, send_to_bucket


class Prediction(BaseModel):
    """Response model for prediction"""

    label_logits: conlist(float, min_items=43, max_items=43)
    is_uppercase_logits: float


app = FastAPI()

BUCKET_NAME = os.environ['USER_INPUT_BUCKET_NAME']
AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
S3_ENDPOINT_URL = os.environ['MLFLOW_S3_ENDPOINT_URL']
CLIENT_ARGS = {
    'endpoint_url': S3_ENDPOINT_URL,
    'aws_access_key_id': AWS_ACCESS_KEY,
    'aws_secret_access_key': AWS_SECRET_KEY,
    'aws_session_token': None,
    'config': Config(signature_version='s3v4'),
    'verify': False,
}


@app.on_event('startup')
def load_model():
    global model
    model = InferenceModel()


@app.get('/')
def root():
    return {}


@app.post('/predict/', response_model=Prediction)
def predict(img_file: UploadFile):
    byte_img = img_file.file.read()
    send_to_bucket(byte_img, BUCKET_NAME, CLIENT_ARGS)
    image = read_img(byte_img)
    lbl_logits, is_upp_logits = model.predict(image)
    return {
        'label_logits': lbl_logits.tolist()[0],
        'is_uppercase_logits': is_upp_logits.item(),
    }

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel, conlist

from .utils import read_img, InferenceModel


class Prediction(BaseModel):
    """Response model for prediction"""

    label_logits: conlist(float, min_items=43, max_items=43)
    is_uppercase_logits: float


app = FastAPI()


@app.on_event('startup')
def load_model():
    global model
    model = InferenceModel()


@app.get('/')
def root():
    return {}


@app.post('/predict/', response_model=Prediction)
def predict(img_file: UploadFile):
    image = read_img(img_file)
    lbl_logits, is_upp_logits = model.predict(image)
    return {
        'label_logits': lbl_logits.tolist()[0],
        'is_uppercase_logits': is_upp_logits.item(),
    }

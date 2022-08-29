import numpy as np
import cv2 as cv
from fastapi import FastAPI, UploadFile


app = FastAPI()

@app.get('/')
def root():
    return {}

@app.post('/file/')
def save_file(img_file: UploadFile):
    img = np.frombuffer(img_file.file.read(), np.uint8)
    img = cv.imdecode(img, cv.IMREAD_UNCHANGED)
    status = cv.imwrite(img_file.filename, img)
    msg = 'Your image succesfully saved' if status else 'Error occured'
    return {'status': msg}

FROM python:3.9-slim 

RUN apt-get update \
    && apt-get install -y libgl1 libglib2.0-0 \
    && pip install --upgrade pip --no-cache-dir \
    && pip install fastapi==0.81.0 python-multipart==0.0.5 mlflow==1.28.0 numpy==1.23.2 opencv-python==4.6.0.66 torch==1.12.1 boto3 uvicorn[standart]==0.18.3 --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir \
    && apt-get clean

COPY ./backend/main.py ./backend/utils.py ./backend/__init__.py /src/backend/
COPY ./models/models.py ./models/__init__.py /src/models/
COPY ./models/__init__.py /src/

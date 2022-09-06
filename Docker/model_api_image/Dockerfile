FROM python:3.9-slim 

RUN pip install --upgrade pip --no-cache-dir \
    && pip install fastapi==0.81.0 python-multipart==0.0.5 mlflow==1.28.0 numpy==1.23.2 opencv-python==4.6.0.66 --no-cache-dir \
    && pip install torch==1.12.1 --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu 
COPY ./backend/main.py ./backend/utils.py ./backend/__init__.py /app/backend/
COPY ./models/models.py ./models/__init__.py /app/models/
COPY ./models/__init__.py /app/

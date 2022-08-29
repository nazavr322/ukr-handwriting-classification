FROM python:3.9-slim 

RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && pip install mlflow boto3 psycopg2

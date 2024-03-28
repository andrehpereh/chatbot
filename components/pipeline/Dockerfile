FROM python:3.9-slim

WORKDIR /pipeline
RUN ls
COPY requirements.txt .
RUN pip install -U -r requirements.txt
COPY . /pipeline 
WORKDIR /pipeline
RUN ls
RUN pwd
RUN pip list
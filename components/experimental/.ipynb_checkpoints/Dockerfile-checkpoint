FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -U -r requirements.txt
COPY experimental.py /app
WORKDIR /app
RUN pip list
RUN ls

ENTRYPOINT ["python"]  
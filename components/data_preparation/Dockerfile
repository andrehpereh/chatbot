FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -U -r requirements.txt
COPY data_ingestion.py /app
WORKDIR /app
RUN ls
RUN pwd
RUN pip list
ENTRYPOINT ["python"]  
# CMD ["data_ingestion.py"]
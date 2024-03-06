FROM python:3.9-slim

WORKDIR /root

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 5000

RUN pwd
RUN ls

CMD ["python", "app/app.py"]  
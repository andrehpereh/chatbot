FROM python:3.9-slim

WORKDIR /root
RUN pwd
RUN ls

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN ls

COPY . /app

WORKDIR /app

EXPOSE 5000

RUN pwd
RUN ls

CMD ["python", "app/app.py"]  
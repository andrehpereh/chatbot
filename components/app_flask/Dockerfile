FROM python:3.9-slim

WORKDIR /root
COPY requirements.txt .
RUN pip install -U -r requirements.txt
ARG PROJECT_ID
ENV PROJECT_ID=$PROJECT_ID
RUN echo $PROJECT_ID
COPY . /app 
WORKDIR /app
EXPOSE 5000


CMD ["python", "-m", "app.app"]  
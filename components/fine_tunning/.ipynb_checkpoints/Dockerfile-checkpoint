FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu121.py310

WORKDIR /root
RUN ls
COPY requirements.txt .
RUN pip install -U -r requirements.txt
RUN ls
COPY trainer /trainer
RUN ls
WORKDIR /trainer
RUN ls
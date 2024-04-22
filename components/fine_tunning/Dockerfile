FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu121.py310

WORKDIR /trainer
COPY requirements.txt .
RUN pip install -U -r requirements.txt
ARG KAGGLE_USERNAME
ENV KAGGLE_USERNAME=$KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV KAGGLE_KEY=$KAGGLE_KEY
COPY . /trainer
WORKDIR /trainer
RUN ls

ENTRYPOINT ["python"]
# CMD ["-c", "import os; my_variable = os.environ['KAGGLE_KEY']; print(my_variable)"]
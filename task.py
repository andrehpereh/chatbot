
import os
import argparse

from trainer import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    PROJECT_ID = os.getenv("PROJECT_ID")
    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "andrehpereh1")  
    KAGGLE_KEY = os.getenv("KAGGLE_KEY", "5859e39806d9456749dcbac685f04bc9") 
    KERAS_BACKEND = os.getenv("KERAS_BACKEND", "tensorflow")
    REGION = os.getenv("REGION", "us-central1")  
    BUCKET_NAME = os.getenv("BUCKET_NAME", f"{PROJECT_ID}-chatbot-v1") # Using os.getenv for consistency 
    BUCKET_URI = f"gs://{BUCKET_NAME}"
    SERVICE_ACCOUNT_NAME = "gemma-vertexai-chatbot"
    SERVICE_ACCOUNT_DISPLAY_NAME = "Gemma Vertex AI endpoint"
    SERVICE_ACCOUNT = f"{SERVICE_ACCOUNT_NAME}@{PROJECT_ID}.iam.gserviceaccount.com"

    # Argument Parser Setup
    parser = argparse.ArgumentParser()

    # ... (Your parser.add_argument calls)

    # Using the variables with their defaults
    parser.add_argument('--kaggle-username', dest='kaggle_username', default=KAGGLE_USERNAME, type=str, help='Kaggle Username')
    parser.add_argument('--kaggle-key', dest='kaggle_key', default=KAGGLE_KEY, type=str, help='Kaggle API Key')
    parser.add_argument('--keras-backend', dest='keras_backend', default=KERAS_BACKEND, type=str, help='Keras backend to use')
    parser.add_argument('--region', dest='region', default=REGION, type=str, help='Which region')
    parser.add_argument('--bucket-name', dest='bucket_name', default=BUCKET_NAME, type=str, help='Name of the GCS bucket')
    parser.add_argument('--service-account', dest='service_account', default=SERVICE_ACCOUNT, type=str, help='Service Account for the project')
    parser.add_argument('--timestamp', dest='timestamp', default=TIMESTAMP, type=str, help='Timestamp to mark training runs')
    
    args = parser.parse_args()
    hparams = args.__dict__

    model.train_evaluate(hparams)

    
   
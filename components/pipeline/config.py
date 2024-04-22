# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The pipeline configurations.
"""

import os
import datetime

class Config:
    """Sets configuration vars."""
    # Lab user environment resource settings
    PROJECT_ID = os.getenv("PROJECT_ID", "able-analyst-416817")  # Replace with the logic to get your project ID 
    # Other Variables
    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "andrehpereh1")
    USER_NAME = os.getenv("USER_NAME", "andrehpereh")
    KAGGLE_KEY = os.getenv("KAGGLE_KEY", "5859e39806d9456749dcbac685f04bc9")
    KERAS_BACKEND = os.getenv("KERAS_BACKEND", "tensorflow")
    REGION = os.getenv("REGION", "us-central1")
    BUCKET_NAME = os.getenv("BUCKET_NAME", f"{PROJECT_ID}-chatbot-v1")
    # BUCKET_URI = os.getenv("BUCKET_URI", f"gs://{BUCKET_NAME}")
    SERVICE_ACCOUNT_NAME = os.getenv("SERVICE_ACCOUNT_NAME", "gemma-vertexai-chatbot")
    SERVICE_ACCOUNT_DISPLAY_NAME = "Gemma Vertex AI endpoint"  # Not directly converted 
    SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT", f"{SERVICE_ACCOUNT_NAME}@{PROJECT_ID}.iam.gserviceaccount.com")
    TIMESTAMP = os.getenv("TIMESTAMP", datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    MODEL_NAME = os.getenv("MODEL_NAME", "gemma_2b_en")
    RANK_LORA = os.getenv("RANK_LORA", 6)  # Default value of 6
    SEQUENCE_LENGTH = os.getenv("SEQUENCE_LENGTH", 256)
    EPOCHS = os.getenv("EPOCHS", 1)
    BATCH_SIZE = os.getenv("BATCH_SIZE", 1)
    TRAIN_DATA_DIR = os.getenv("TRAIN_DATA_DIR", "input_data/andrehpereh")
    SERVING_IMAGE = os.getenv("SERVING_IMAGE", "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240220_0936_RC01")
    FINE_TUNE_FLAG = os.getenv("FINE_TUNE_FLAG", False)




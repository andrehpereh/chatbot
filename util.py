from config import Config
import os
import re
import numpy as np
from google.cloud import storage

def get_model_paths_and_config(model_name):
    """
    Constructs paths, determines machine configuration, and gets the VLLM model name for a given model.

    Args:
        model_name (str): The base name of the model (e.g., "gemma_2b_en").

    Returns:
        dict: A dictionary containing the following keys:
            - 'model_size': The size of the model ("2b" or "7b").
            - 'finetuned_model_dir': Path to the finetuned model directory.
            - 'finetuned_weights_path': Path to the finetuned model weights.
            - 'finetuned_vocab_path': Path to the finetuned model vocabulary.
            - 'huggingface_model_dir': Path to the Hugging Face model directory.
            - 'deployed_model_blob': Blob name of the deployed model in Cloud Storage.
            - 'deployed_model_uri': URI of the deployed model in Cloud Storage.
            - 'machine_type': The appropriate machine type.
            - 'accelerator_type': The accelerator type.
            - 'accelerator_count': The number of accelerators.
            - 'model_name_vllm': The VLLM-specific model name.
    """

    allowed_models = [
        "gemma_2b_en",
        "gemma_instruct_2b_en",
        "gemma_7b_en",
        "gemma_instruct_7b_en",
    ]

    if model_name not in allowed_models:
        raise ValueError(f"Invalid model_name. Supported models are: {allowed_models}")

    # Construct paths
    model_size = model_name.split("_")[-2]
    assert model_size in ("2b", "7b")

    finetuned_model_dir = f"./{model_name}"
    finetuned_weights_path = f"{finetuned_model_dir}/model.weights.h5"
    finetuned_vocab_path = f"{finetuned_model_dir}/vocabulary.spm"
    huggingface_model_dir = f"./{model_name}_huggingface"
    timestamp = Config.TIMESTAMP
    deployed_model_blob = f"{model_name}/{timestamp}" 
    deployed_model_uri = f"{Config.BUCKET_URI}/{deployed_model_blob}"  # Assuming BUCKET_URI is globally defined

    # Determine machine configuration
    machine_config = {
        "2b": {
            "machine_type": "g2-standard-8",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1
        },
        "7b": {
            "machine_type": "g2-standard-12",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1
        }
    }[model_size]  # Efficient lookup

    return {
        "model_size": model_size,
        "finetuned_model_dir": finetuned_model_dir,
        "finetuned_weights_path": finetuned_weights_path,
        "finetuned_vocab_path": finetuned_vocab_path,
        "huggingface_model_dir": huggingface_model_dir,
        "deployed_model_blob": deployed_model_blob,
        "deployed_model_uri": deployed_model_uri,
        "model_name_vllm": f"{model_name}-vllm", 
        **machine_config  # Add machine config directly
    }

def upload2bs(local_directory, bucket_name, destination_subfolder=""):
    """Uploads a local directory and its contents to a Google Cloud Storage bucket.

    Args:
        local_directory (str): Path to the local directory.
        bucket_name (str): Name of the target Google Cloud Storage bucket.
        destination_subfolder (str, optional): Prefix to append to the path within the bucket. 
                                        Defaults to "".
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, _, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            # Construct the path within the bucket
            blob_path = os.path.join(destination_subfolder, os.path.relpath(local_path, local_directory))
            blob = bucket.blob(blob_path)
            #blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_path}")
    destination_path = os.path.dirname(f"gs://{bucket_name}/{blob_path}")
    return destination_path


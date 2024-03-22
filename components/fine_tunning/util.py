import os
import re
from google.cloud import storage


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
        print("Those are the files in the directoty", files)
        for file in files:
            local_path = os.path.join(root, file)
            # Construct the path within the bucket
            blob_path = os.path.join(destination_subfolder, os.path.relpath(local_path, local_directory))
            blob = bucket.blob(blob_path)
            print("This is the blob", blob.name)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_path}")
    destination_path = os.path.dirname(f"gs://{bucket_name}/{blob_path}")
    return destination_path

def download_all_from_blob(bucket_name, blob_prefix, local_destination=""):
    """Downloads all files from a Google Cloud Storage blob (with an optional prefix) to a local directory.

    Args:
        bucket_name (str): Name of the Google Cloud Storage bucket.
        blob_prefix (str): Prefix specifying the subfolder within the bucket to download from.
        local_destination (str, optional): Local directory to download files into. Defaults
                                           to the current working directory.
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=blob_prefix)  # List blobs with the prefix
    # print("This are the blobs", blobs)
    for blob in blobs:
        # Construct local download path (ensuring directories exist)
        print("This is the blob to be downloaded", blob.name)
        print("This is the file name", os.path.basename(blob.name))
        destination_filepath = os.path.join(local_destination, os.path.basename(blob.name))
        os.makedirs(os.path.dirname(destination_filepath), exist_ok=True)

        # Download the file 
        blob.download_to_filename(destination_filepath)
        print(f"Downloaded gs://{bucket_name}/{blob.name} to {destination_filepath}")



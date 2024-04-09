import re
import os
import sys
import json
import argparse
from typing import List
from google.cloud import storage
from io import BytesIO

def process_whatsapp_chat(bucket_name: str, directory: str, pair_count: int=6) -> List[str]:
    print("Bucket Name", bucket_name)
    print("Directory", directory)
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    print(bucket_name)
    print(directory)

    current_sender = None
    current_file = None
    consecutive_messages = []
    qa_pairs_all = []  # List to store question-answer pairs
    print(bucket.list_blobs(prefix=directory))

    for blob in bucket.list_blobs(prefix=directory):
        # Extract filename from blob
        print(blob.name)
        filename = blob.name.split('/')[-1]

        if filename.endswith('.txt') and 'WhatsApp' in filename:
            print(filename)
            file_content = blob.download_as_string().decode('utf-8')
            lines = file_content.split('\n')

            for line in lines:
                if line.startswith(('- ', '\n')):
                    continue

                # Extract sender and message (Regex handles potential variations)
                match = re.search(r'^.*-\s(?P<sender>.*?):\s(?P<message>.*)$', line)
                if match:
                    sender = match.group('sender').strip()
                    if sender != 'Andres Perez':
                        sender = 'Sender'
                    message = match.group('message').strip()
                    message = message.replace("<Media omitted>", "")
                    message = message.replace("Missed video call", "")
                    message = message.replace("null", "")
                    message = re.sub(r'http\S+', '', message).strip()

                    # Concatenate consecutive messages by the same sender
                    if sender == current_sender:
                        consecutive_messages.append(message)
                    else:
                        if consecutive_messages:
                            qa_pairs_all.append(' '.join(consecutive_messages))
                        current_sender = sender
                        consecutive_messages = [f"{sender}:\n{message}"]

            # Add the last set of messages
            if consecutive_messages:
                qa_pairs_all.append(', '.join(consecutive_messages))

    result = []
    current_group = ""
    for i, element in enumerate(qa_pairs_all):
        current_group += element 
        current_group += "\n\n"  # Add double newline after every even element

        if (i + 1) % pair_count == 0:
            result.append(current_group)
            current_group = ""  # Reset for the next group

    # Handle a potential incomplete last group
    if current_group:
        result.append(current_group)
    return result


def process_transcripts(bucket_name, directory, gemini_pro_model, pair_count=6, data_augmentation_iter=4):
    """Processes all text files within a folder.

    Args:
        folder_path: Path to the input folder.
        gemini_pro_model: The model used for generating responses.
    """

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    model_response_all = ""
    script_dir = os.path.dirname(__file__)  # Get the directory of the current script
    filepath = os.path.join(script_dir, 'prompt.json')

    with open(filepath, "r") as file:
        data = json.load(file)
    pre_prompt = data["prompt"]

    for blob in bucket.list_blobs(prefix=directory):
        # Extract filename from blob
        print(blob.name)
        filename = blob.name.split('/')[-1]
        if filename.endswith('.txt') and 'transcript' in filename:
            # Read content of the file
            print(filename)
            contents = blob.download_as_string().decode('utf-8')
            prompt = pre_prompt.format(contents)
            for i in range(data_augmentation_iter):
                try:
                    model_response = gemini_pro_model.generate_content(prompt).text
                    model_response_all += model_response.replace("** ", "\n").replace("**", "")
                    print("-----------------------------------------------------------------------------------------------------------------------------------")
                except Exception as e:  # Catch any type of error
                    print(f"An error occurred: {e}. Skipping...")

    result = []
    current_pair = ""
    speaker_turn_count = 0
    for line in model_response_all.splitlines():
        if line.startswith("Speaker"):  # Detect speaker changes
            speaker_turn_count += 1
            line = line.replace("Speaker 1", "Sender")
            line = line.replace("Speaker 2", "Andres Perez")
            if speaker_turn_count > pair_count:
                result.append(current_pair)
                current_pair = ""
                speaker_turn_count = 1  # Reset count for a new pair

        current_pair += line + "\n"  

    # Add the last pair (if any)
    if current_pair:
        result.append(current_pair)
    return result

def data_preparation(bucket_name, directory, gemini_pro_model, pair_count=6, data_augmentation_iter=4):
    transcripts = process_transcripts(
        bucket_name=bucket_name, directory=directory, gemini_pro_model=gemini_pro_model,
        pair_count=pair_count, data_augmentation_iter=data_augmentation_iter
    )
    whatsapp = process_whatsapp_chat(bucket_name=bucket_name, directory=directory, pair_count=pair_count)
    input_data = transcripts + whatsapp
    print("Number of elements in the list", len(input_data))
    print(input_data[0:15])
    return input_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name', dest='bucket-name',
                        default='able-analyst-416817-chatbot-v1', type=str, help='GCS URI for saving model artifacts.')
    parser.add_argument('--directory', dest='directory', 
                        default='input_data/andrehpereh', type=str, help='TF-Hub URL.')
    args = parser.parse_args()
    hparams = args.__dict__
    process_whatsapp_chat(hparams['bucket-name'], hparams['directory'])
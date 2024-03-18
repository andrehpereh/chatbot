import re
import argparse
from typing import List
import sys
from google.cloud import storage
from io import BytesIO

def process_whatsapp_chat(bucket_name: str, directory: str) -> List[str]:
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
        print(filename)
        if filename.endswith('.txt'):
            print(blob.name)
            
            # Read content of the file
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

    if len(qa_pairs_all) % 2 != 0:
        qa_pairs_all = qa_pairs_all[:-1]
    
    res = []
    for i in range(0, len(qa_pairs_all), 2):
        res.append((qa_pairs_all[i], qa_pairs_all[i+1]))

    formatted_messages = [f"{message_pair[0]}\n\n{message_pair[1]}" for message_pair in res]

    return formatted_messages

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name', dest='bucket-name',
                        default='able-analyst-416817-chatbot-v1', type=str, help='GCS URI for saving model artifacts.')
    parser.add_argument('--directory', dest='directory', 
                        default='input_data/andrehpereh', type=str, help='TF-Hub URL.')
    args = parser.parse_args()
    hparams = args.__dict__
    process_whatsapp_chat(hparams['bucket-name'], hparams['directory'])
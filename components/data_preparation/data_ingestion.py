import os
import re
import numpy as np

def process_whatsapp_chat(directory):

    current_sender = None
    current_file = None
    consecutive_messages = []
    qa_pairs_all = []  # List to store question-answer pairs

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        consecutive_messages = []
        qa_pairs = []  # List to store question-answer pairs
        # Check if the current item is a file and ends with ".txt"
        if os.path.isfile(filepath) and filename.endswith('.txt'):
            print(filepath)
             
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()

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
                            qa_pairs.append(' '.join(consecutive_messages))
                        current_sender = sender
                        consecutive_messages = [f"{sender}:\n{message}"]

            # Add the last set of messages
            if consecutive_messages:
                qa_pairs.append(', '.join(consecutive_messages))
        
        qa_pairs_all.extend(qa_pairs)
    if len(qa_pairs_all) % 2 != 0:
        qa_pairs_all = qa_pairs_all[:-1]
    res = []
    for i in range(0, len(qa_pairs_all), 2):
        res.append((qa_pairs_all[i], qa_pairs_all[i+1]))
    formatted_messages = [f"{message_pair[0]}\n\n{message_pair[1]}" for message_pair in res]
    return formatted_messages 


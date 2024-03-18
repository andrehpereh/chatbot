import os
import argparse

import data_ingestion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name', dest='bucket-name',
                        default='able-analyst-416817-chatbot-v1', type=str, help='GCS URI for saving model artifacts.')
    parser.add_argument('--directory', dest='directory', 
                        default='input_data/andrehpereh', type=str, help='TF-Hub URL.')
    args = parser.parse_args()
    hparams = args.__dict__
    data_ingestion.process_whatsapp_chat(hparams['bucket-name'], hparams['directory'])
import argparse
import sys
import keras
import keras_nlp
import os
import json

def add_test(data: list[str], model_paths:dict, model_name: str='gemma_2b_en', rank_lora: int=6, sequence_length: int=256, epochs: int=2, batch_size: int=1):
    # res: str="gs://able-analyst-416817-chatbot-v1/gemma_2b_en_raw/gemma_2b_en"
    print(os.getenv("KAGGLE_USERNAME"))
    print(os.getenv("KAGGLE_KEY"))
    model = keras_nlp.models.GemmaCausalLM.from_preset(model_name)
    print("This is the data", data)
    print("This is the data", model_paths)
    os.makedirs(model_paths['finetuned_model_dir'], exist_ok=True)
    model.save_weights(model_paths['finetuned_weights_path'])
    model.preprocessor.tokenizer.save_assets(model_paths['finetuned_model_dir'])
    print("Saving model in ", model_paths['finetuned_model_dir'])
    finetuned_weights_path = model_paths['finetuned_weights_path']
    return finetuned_weights_path
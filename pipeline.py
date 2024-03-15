import keras
import keras_nlp
import os
from util import get_model_paths_and_config, process_whatsapp_chat, upload2bs
from config import Config
from trainer import finetune_gemma
from conversion_function import convert_checkpoints
from numba import cuda
from google_cloud_pipeline_components import aiplatform as gcc_aip
import kfp.dsl as dsl

model_paths_and_config = get_model_paths_and_config(Config.MODEL_NAME)

data = process_whatsapp_chat(Config.TRAIN_DATA_DIR)

finetuned_weights_path = finetune_gemma(data=data[:50], model_paths=model_paths_and_config, model_name=Config.MODEL_NAME, rank_lora=Config.SEQUENCE_LENGTH, sequence_length=Config.SEQUENCE_LENGTH, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE)

device = cuda.get_current_device()
cuda.select_device(device.id)
cuda.close()

output_dir = convert_checkpoints(
    weights_file=finetuned_weights_path,
    size=model_paths_and_config['model_size'],
    output_dir=model_paths_and_config['huggingface_model_dir'],
    vocab_path=model_paths_and_config['finetuned_vocab_path'],
)

destination_path = upload2bs(
    local_directory = output_dir, bucket_name = Config.BUCKET_NAME,
    destination_subfolder = model_paths_and_config['deployed_model_blob']
)



@dsl.component(
    base_image="python:3.9-slim"  # Choose appropriate Python image
)
def process_whatsapp_chat_op(chat_path: str) -> NamedTuple("Outputs", [("data_list", List[str]), ("data_dict", Dict[str, int])]):
    model_paths_and_config = get_model_paths_and_config(Config.MODEL_NAME)
    data = process_whatsapp_chat(Config.TRAIN_DATA_DIR)
    return data, model_paths_and_config


@dsl.component(
    base_image="tensorflow/tensorflow:latest-gpu"  # Or other GPU-enabled image 
)
def finetune_gemma_op(
    data: str,
    model_paths: dict,  
    model_name: str,
    rank_lora: int,
    sequence_length: int,
    epochs: int,
    batch_size: int
) -> str:
    job = gcc_aip.CustomContainerTrainingJobRunOp(
        ... # Configure your training job parameters
    )
    finetuned_weights_path = finetune_gemma(
        data=data[:50], model_paths=model_paths_and_config, model_name=Config.MODEL_NAME,
        rank_lora=Config.SEQUENCE_LENGTH,sequence_length=Config.SEQUENCE_LENGTH, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE
    )
    return finetuned_weights_path
    # ... return path to finetuned weights
    
@dsl.component(
    base_image="python:3.8"  
)
def convert_checkpoints_op(
    weights_file: str,
    size: str, 
    output_dir: str, 
    vocab_path: str    
) -> str:
    output_dir = convert_checkpoints(
        weights_file=finetuned_weights_path,
        size=model_paths_and_config['model_size'],
        output_dir=model_paths_and_config['huggingface_model_dir'],
        vocab_path=model_paths_and_config['finetuned_vocab_path'],
    )
    return output_dir

import kfp.dsl as dsl

@dsl.component(
    base_image="python:3.8"  # You might need additional cloud storage libraries
)
def upload2bs_op(
    local_directory: str,
    bucket_name: str, 
    destination_subfolder: str
) -> str:
    destination_path = upload2bs(
        local_directory = output_dir, bucket_name = Config.BUCKET_NAME,
        destination_subfolder = model_paths_and_config['deployed_model_blob']
    )
    return destination_path


import kfp.dsl as dsl

@dsl.pipeline(
    name="whatsapp-chat-finetuning",
    description="Pipeline to process and fine-tune a model on WhatsApp chat data"
)
def whatsapp_chat_pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
    staging_bucket: str = GCS_BUCKET,
    display_name: str = DISPLAY_NAME,
    container_uri: str = IMAGE_URI,
    model_serving_container_image_uri: str = SERVING_IMAGE_URI,
    base_output_dir: str = GCS_BASE_OUTPUT_DIR,
):
    chat_data = process_whatsapp_chat_op("./CHAT")
    model_paths_config = get_model_paths_and_config(Config.MODEL_NAME)

    finetune_task = finetune_gemma_op(
        data=chat_data.output,
        model_paths=model_paths_config,
        model_name=Config.MODEL_NAME,
        rank_lora=Config.SEQUENCE_LENGTH,
        sequence_length=Config.SEQUENCE_LENGTH,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    )

    conversion_task = convert_checkpoints_op(
        weights_file=finetune_task.output,
        size=model_paths_config['model_size'], 
        output_dir=model_paths_config['huggingface_model_dir'], 
        vocab_path=model_paths_config['finetuned_vocab_path']
    )

    upload_task = upload2bs_op(
        local_directory=conversion_task.output, 
        bucket_name=Config.BUCKET_NAME, 
        destination_subfolder=model_paths_config['deployed_model_blob']
    )

    model_upload_task = ModelUploadOp(destination_path=upload_task.output)
    endpoint_create_task = gcc_aip.EndpointCreateOp()
    model_deploy_task = gcc_aip.ModelDeployOp(
        model=model_upload_task.outputs["model"],
        endpoint=endpoint_create_task.outputs["endpoint"]
    ) 





























ModelUploadOp(destination_path)

endpoint_create_op = gcc_aip.EndpointCreateOp()

model_deploy_op = gcc_aip.ModelDeployOp(
    # Link to model training component through output model artifact.
    model=model_train_evaluate_op.outputs["model"],
    # Link to the created Endpoint.
    endpoint=endpoint_create_op.outputs["endpoint"]
)
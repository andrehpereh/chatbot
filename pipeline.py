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






















@dsl.component(base_image="your-conversion-image:latest") 
def keras_to_huggingface_op(
    keras_model_path: dsl.InputPath("Model"), 
    hf_model_uri: dsl.OutputPath("Model")
) -> str:
    keras_model_path = keras_model_path # model_paths['finetuned_weights_path']
    convert_checkpoints(
        preset=model_name,
        weights_file=keras_model_path,
        size=model_paths['model_size'],
        output_dir=model_paths['finetuned_vocab_path'],
        vocab_path=model_paths['huggingface_model_dir'],
    )
    hf_model_uri = model_paths['huggingface_model_dir']
    return hf_model_uri

@component(base_image="your-training-image:latest")  # Replace with your Docker image
def custom_training_op(
    # ... Any necessary parameters for your training job ...
) -> aiplatform.Model:
    # Import the component
    from google_cloud_pipeline_components import aiplatform as gcc_aip

    job = gcc_aip.CustomContainerTrainingJobRunOp(
        # ... Configuration for your Vertex AI container training job ...
    )

    # Assuming your training job produces a model directly as an artifact
    return job.outputs["model"] 

@dsl.pipeline(name="bert-sentiment-classification", pipeline_root=PIPELINE_ROOT)
def pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
    staging_bucket: str = GCS_BUCKET,
    display_name: str = DISPLAY_NAME,
    container_uri: str = IMAGE_URI,
    model_serving_container_image_uri: str = SERVING_IMAGE_URI,
    base_output_dir: str = GCS_BASE_OUTPUT_DIR,
):

    from google_cloud_pipeline_components import aiplatform as gcc_aip

    # Create the training job component
    model_train_evaluate_op = gcc_aip.CustomContainerTrainingJobRunOp(
        # Vertex AI Python SDK authentication parameters.     
        project=project,
        location=location,
        staging_bucket=staging_bucket,
        display_name=display_name,  # Added from pipeline definition
        container_uri=container_uri,
        model_serving_container_image_uri=model_serving_container_image_uri,

        # WorkerPool arguments.
        replica_count=1,
        machine_type="e2-standard-4",

        # Additional Arguments 
        base_output_dir=base_output_dir 
        # ... other arguments specific to your training code
    )

    training_op = gcc_aip.CustomContainerTrainingJobRunOp(...) 

    conversion_op = keras_to_huggingface_op(
        keras_model_path=training_op.outputs["model_path"]  
    )



    gcc_aip.ModelUploadOp(
        artifact_uri=conversion_op.outputs["hf_model_uri"],
        serving_container_image_uri=serving_container_uri,
    )


    # Create a Vertex Endpoint resource in parallel with model training.
    endpoint_create_op = gcc_aip.EndpointCreateOp(
        # Vertex AI Python SDK authentication parameters.
        project=project,
        location=location,
        display_name=display_name
    
    )   
    
    # Deploy your model to the created Endpoint resource for online predictions.
    model_deploy_op = gcc_aip.ModelDeployOp(
        # Link to model training component through output model artifact.
        model=model_train_evaluate_op.outputs["model"],
        # Link to the created Endpoint.
        endpoint=endpoint_create_op.outputs["endpoint"],
        # Define prediction request routing. {"0": 100} indicates 100% of traffic 
        # to the ID of the current model being deployed.
        traffic_split={"0": 100},
        # WorkerPool arguments.        
        dedicated_resources_machine_type="e2-standard-4",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=2
    )
res = !gcloud config get core/project
PROJECT_ID = res[0]


from kfp import dsl
import kfp as kfp
from kfp.dsl import OutputPath, Artifact, InputPath
from kfp import compiler
from config import Config
from util import get_model_paths_and_config
from google.cloud import aiplatform as vertexai

@dsl.component(
  base_image ='gcr.io/able-analyst-416817/gemma-chatbot-data-preparation:latest'
)
def process_whatsapp_chat_op(
  bucket_name: str,
  directory: str,
  dataset_path: OutputPath('Dataset')
):
    import data_ingestion
    import json
    formatted_messages = data_ingestion.process_whatsapp_chat(bucket_name, directory)
    with open(dataset_path, 'w') as f:
        json.dump(formatted_messages, f)


@dsl.component(
  base_image = 'gcr.io/able-analyst-416817/gemma-chatbot-fine-tunning:latest'
)
def fine_tunning(
  dataset_path: InputPath('Dataset'),
  model_paths: dict,
  #finetuned_weights_dir: OutputPath('Model'),
) -> str:
    # import test_container
    import trainer
    import json
    import util
    import os
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    os.makedirs(model_paths['finetuned_model_dir'], exist_ok=True)
    finetuned_weights_path = os.path.join(model_paths['finetuned_model_dir'], 'model.weights.h5') 
    
    model = trainer.finetune_gemma(dataset, model_paths, False)
    print("Its gonna save it here", finetuned_weights_path)
    #model.save_weights(finetuned_weights_path)
    #model.preprocessor.tokenizer.save_assets(model_paths['finetuned_model_dir'])
    bucket_name = 'able-analyst-416817-chatbot-v1' # move to parameter.
    util.upload2bs(
        local_directory = model_paths['finetuned_model_dir'], bucket_name = bucket_name,
        destination_subfolder = model_paths['fine_tuned_keras_blob']
    )
    model_gcs = "gs://{}/{}".format(bucket_name, model_paths['fine_tuned_keras_blob'])  
    print("This is the storage bucket", model_gcs)
    return model_gcs
    

@dsl.component(
  base_image = 'gcr.io/able-analyst-416817/gemma-chatbot-fine-tunning:latest'
)
def convert_checkpoints_op(
  keras_gcs_model: str,
  model_paths: dict
) -> str:
    import conversion_function
    import os
    import util
    bucket_name, blob_name = os.path.dirname(keras_gcs_model).lstrip("gs://").split("/", 1) 
    print("This is the keras passed", keras_gcs_model)
    util.download_all_from_blob(bucket_name, model_paths['fine_tuned_keras_blob'], local_destination=model_paths['finetuned_model_dir'])
    if os.path.exists("./model.weights.h5"):
        print("File exists!")
    else:
        print("File does not exist.")
    converted_fined_tuned_path = conversion_function.convert_checkpoints(
        weights_file=model_paths['finetuned_weights_path'],
        size=model_paths['model_size'],
        output_dir=model_paths['huggingface_model_dir'],
        vocab_path=model_paths['finetuned_vocab_path']
    )
    util.upload2bs(
        local_directory = converted_fined_tuned_path, bucket_name = bucket_name,
        destination_subfolder = model_paths['deployed_model_blob']
    )
    return model_paths['deployed_model_uri']



@kfp.dsl.pipeline(name="Model deployment.")
def model_deployment_pipeline(
    project: str = PROJECT_ID, bucket_name: str = "able-analyst-416817-chatbot-v1", directory: str = "input_data/andrehpereh", 
    model_paths: dict=get_model_paths_and_config(Config.MODEL_NAME)
):
    WORKING_DIR = 'gs://able-analyst-416817-chatbot-v1/gemma_2b_en/20240322091040/'
    VLLM_DOCKER_URI = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240220_0936_RC01"
    print(Config.MODEL_NAME)
    model_paths_and_config = get_model_paths_and_config(Config.MODEL_NAME)

    from google_cloud_pipeline_components.types import artifact_types
    from google_cloud_pipeline_components.v1.endpoint import (EndpointCreateOp,
                                                              ModelDeployOp)
    from google_cloud_pipeline_components.v1.model import ModelUploadOp
    from kfp.dsl import importer_node


    port = 7080
    accelerator_count=1
    max_model_len=256
    dtype="bfloat16"
    vllm_args = [
        "--host=0.0.0.0",
        f"--port={port}",
        f"--tensor-parallel-size={accelerator_count}",
        "--swap-space=16",
        "--gpu-memory-utilization=0.95",
        f"--max-model-len={max_model_len}",
        f"--dtype={dtype}",
        "--disable-log-stats",
    ]

    metadata = {
      "imageUri": VLLM_DOCKER_URI,
      "command": ["python", "-m", "vllm.entrypoints.api_server"],
      "args": vllm_args,
      "ports": [
        {
          "containerPort": port
        }
      ],
      "predictRoute": "/generate",
      "healthRoute": "/ping"
    }

    # from google_cloud_pipeline_components import ModelUploadOp
    model_paths = get_model_paths_and_config(Config.MODEL_NAME)
    whatup = process_whatsapp_chat_op(bucket_name = bucket_name, directory = directory)

    trainer = fine_tunning(dataset_path=whatup.outputs['dataset_path'], model_paths=model_paths)
    trainer.set_memory_limit("50G").set_cpu_limit('12.0m').set_accelerator_limit(1).add_node_selector_constraint("NVIDIA_L4")

    print("This is the dictionary", model_paths)
    converted = convert_checkpoints_op(
        keras_gcs_model=trainer.output, model_paths=model_paths
    ).set_memory_limit("50G").set_cpu_limit('8.0m').set_accelerator_limit(1).add_node_selector_constraint("NVIDIA_L4")

    import_unmanaged_model_task = importer_node.importer(
        artifact_uri=converted.output,
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            "containerSpec": metadata,
        },
    )

    model_upload_op = ModelUploadOp(
        project=project,
        display_name="Mini Andres, first version automated",
        unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],
    )
    model_upload_op.after(import_unmanaged_model_task)

    endpoint_create_op = EndpointCreateOp(
        project=project,
        display_name="pipelines-created-endpoint",
    )

    ModelDeployOp(
        endpoint=endpoint_create_op.outputs["endpoint"],
        model=model_upload_op.outputs["model"],
        deployed_model_display_name=model_paths_and_config['model_name_vllm'],
        dedicated_resources_machine_type=model_paths_and_config['machine_type'],
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
        dedicated_resources_accelerator_type=model_paths_and_config['accelerator_type'],
        dedicated_resources_accelerator_count=model_paths_and_config['accelerator_count']
    )

    
compiler.Compiler().compile(
    pipeline_func=model_deployment_pipeline, package_path="model_deployment_pipeline.json"
)
vertexai.init(project=PROJECT_ID, location="us-central1")
vertex_pipelines_job = vertexai.pipeline_jobs.PipelineJob(
    display_name="test-model_deployment_pipeline",
    template_path="model_deployment_pipeline.json"
)
vertex_pipelines_job.run()
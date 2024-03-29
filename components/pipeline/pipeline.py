from kfp import dsl
import kfp as kfp
from kfp.dsl import OutputPath, Artifact, InputPath
from config import Config
from util import get_model_paths_and_config
import os
TAG_NAME = os.environ.get('TAG_NAME', 'latest') 

@dsl.component(
  base_image =f"gcr.io/able-analyst-416817/gemma-chatbot-data-preparation:{TAG_NAME}"
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
  base_image = f"gcr.io/able-analyst-416817/gemma-chatbot-fine-tunning:{TAG_NAME}"
)
def fine_tunning(
  dataset_path: InputPath('Dataset'),
  model_paths: dict,
  fine_tune_flag: bool,
  epochs: int,
  model_name: str
) -> str:
    import trainer
    import json
    import util
    import os
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    os.makedirs(model_paths['finetuned_model_dir'], exist_ok=True)
    finetuned_weights_path = os.path.join(model_paths['finetuned_model_dir'], 'model.weights.h5') 
    
    model = trainer.finetune_gemma(dataset, model_paths, fine_tune_flag, epochs=epochs, model_name=model_name)
    print("Its gonna save it here", finetuned_weights_path)
    bucket_name = 'able-analyst-416817-chatbot-v1' # move to parameter.
    util.upload2bs(
        local_directory = model_paths['finetuned_model_dir'], bucket_name = bucket_name,
        destination_subfolder = model_paths['fine_tuned_keras_blob']
    )
    model_gcs = "gs://{}/{}".format(bucket_name, model_paths['fine_tuned_keras_blob'])  
    print("This is the storage bucket", model_gcs)
    return model_gcs
    

@dsl.component(
  base_image = f"gcr.io/able-analyst-416817/gemma-chatbot-fine-tunning:{TAG_NAME}"
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


@dsl.pipeline(name="Model deployment.")
def fine_tune_pipeline(
    project: str = os.environ.get('PROJECT_ID') ,
    bucket_name: str = "able-analyst-416817-chatbot-v1",
    directory: str = "input_data/andrehpereh",
    serving_image: str = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240220_0936_RC01",
    fine_tune_flag: bool = False,
    epochs: int = 3,
    model_name: str = 'gemma_2b_en'
):
    print("Aqui esta el mismo pedo", bucket_name)
    print("Aqui esta el mismo pedo", type(bucket_name))
    from config import Config
    from google_cloud_pipeline_components.types import artifact_types
    from google_cloud_pipeline_components.v1.endpoint import (EndpointCreateOp, ModelDeployOp)
    from google_cloud_pipeline_components.v1.model import ModelUploadOp
    from kfp.dsl import importer_node
    from util import get_model_paths_and_config
    print("This is the model name", Config.MODEL_NAME)
    model_paths = get_model_paths_and_config(Config.MODEL_NAME)

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
      "imageUri": serving_image,
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

    whatup = process_whatsapp_chat_op(bucket_name = bucket_name, directory = directory)

    trainer = fine_tunning(dataset_path=whatup.outputs['dataset_path'], model_paths=model_paths, fine_tune_flag=fine_tune_flag, epochs=epochs, model_name=model_name)
    trainer.set_memory_limit("70G").set_cpu_limit("13.0").set_accelerator_limit(1).add_node_selector_constraint(model_paths['accelerator_type'])

    print("This is the dictionary", model_paths)
    converted = convert_checkpoints_op(
        keras_gcs_model=trainer.output, model_paths=model_paths
    ).set_memory_limit("70G").set_cpu_limit("13.0").set_accelerator_limit(1).add_node_selector_constraint(model_paths['accelerator_type'])

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
        display_name="model_deployment_pipeline: End Point Created",
    )

    ModelDeployOp(
        endpoint=endpoint_create_op.outputs["endpoint"],
        model=model_upload_op.outputs["model"],
        deployed_model_display_name=model_paths['model_name_vllm'],
        dedicated_resources_machine_type=model_paths['machine_type'],
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
        dedicated_resources_accelerator_type=model_paths['accelerator_type'],
        dedicated_resources_accelerator_count=model_paths['accelerator_count']
    )


if __name__ == '__main__':
    from kfp import compiler
    from google.cloud import aiplatform as vertexai
    from config import Config
    # from util import get_model_paths_and_config
    print("This is the model name", Config.MODEL_NAME, "Ahuevito")
    # model_paths = get_model_paths_and_config(Config.MODEL_NAME)
    pipeline_name = f"fine_tune_pipeline{Config.USER_NAME}.json"
    compiler.Compiler().compile(
        pipeline_func=fine_tune_pipeline, package_path=pipeline_name
    )
    vertexai.init(project=Config.PROJECT_ID, location=Config.REGION)
    vertex_pipelines_job = vertexai.pipeline_jobs.PipelineJob(
        display_name="test-fine_tune_pipeline",
        template_path=pipeline_name,
        parameter_values={
            "project": Config.PROJECT_ID,
            "bucket_name": Config.BUCKET_NAME,
            "directory": Config.TRAIN_DATA_DIR,
            "serving_image": Config.SERVING_IMAGE,
            "fine_tune_flag": Config.FINE_TUNE_FLAG,
            "epochs": Config.EPOCHS,
            "model_name": Config.MODEL_NAME
        }
    )
    vertex_pipelines_job.run()
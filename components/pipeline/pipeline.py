from kfp import dsl
import kfp as kfp
from kfp.dsl import OutputPath, Artifact, InputPath, PipelineTaskFinalStatus, ExitHandler
from kfp import compiler
from config import Config
from util import get_model_paths_and_config
from google.cloud import aiplatform as vertexai
import os


@dsl.component(base_image='python:3.9')
def send_pipeline_completion_email_op(
    status: PipelineTaskFinalStatus,
    smtp_server: str = 'smtp.gmail.com',
    smtp_port: int = 587,
    sender_email: str = 'andrehpereh96@gmail.com',
    recipient_emails: str = "andrehpereh@gmail.com",
    email_password: str = "ssuy rubm kzge juid"
):
    import smtplib
    from email.mime.text import MIMEText
    recipient_emails = [recipient_emails]
    """
    Monitors for a success flag file and sends an email upon detection.

    Args:
        smtp_server (str): SMTP server address. Defaults to 'smtp.gmail.com'.
        smtp_port (int): SMTP server port. Defaults to 587.
        sender_email (str): Email address of the sender. Defaults to 'your_email@gmail.com'.
        recipient_emails (list): List of recipient email addresses. Defaults to ['recipient@example.com'].
        email_password (str): Password for the sender's email account.
        success_flag_path (str): Path to the success flag file. Defaults to '/tmp/pipeline_success_flag.txt'.
    """

    msg = MIMEText(
        f"Kubeflow Pipeline Completion Status; {status.state} and Job resource name:{status.pipeline_job_resource_name},\
        \nPipeline task name: {status.pipeline_task_name} Errormessage: , {status.error_message}"
    )
    msg['Subject'] = 'Kubeflow Pipeline Completion'
    msg['From'] = sender_email
    msg['To'] = ', '.join(recipient_emails)

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()  # Enable TLS encryption
        print("This is the email", sender_email)
        print("This is the password", email_password)
        server.login(sender_email, email_password)
        server.sendmail(sender_email, recipient_emails, msg.as_string())

    print('Email sent!')




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
def fine_tune_pipeline(
    project: str = os.environ.get('PROJECT_ID') ,
    bucket_name: str = "able-analyst-416817-chatbot-v1",
    directory: str = "input_data/andrehpereh",
    serving_image: str = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240220_0936_RC01",
    fine_tune_flag: bool = False,
    epochs: int = 3,
    model_name: str = 'gemma_2b_en'
):

    from google_cloud_pipeline_components.types import artifact_types
    from google_cloud_pipeline_components.v1.endpoint import (EndpointCreateOp, ModelDeployOp)
    from google_cloud_pipeline_components.v1.model import ModelUploadOp
    from google_cloud_pipeline_components.v1.vertex_notification_email import VertexNotificationEmailOp
    from kfp.dsl import importer_node
    from util import get_model_paths_and_config
    from config import Config

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
    send_email = send_pipeline_completion_email_op(recipient_emails = f'{Config.USER_NAME}@gmail.com')
    with ExitHandler(send_email):
        whatup = process_whatsapp_chat_op(bucket_name = bucket_name, directory = directory)

        trainer = fine_tunning(dataset_path=whatup.outputs['dataset_path'], model_paths=model_paths, fine_tune_flag=fine_tune_flag, epochs=epochs, model_name=model_name)
        trainer.set_memory_limit("50G").set_cpu_limit("12.0").set_accelerator_limit(1).add_node_selector_constraint(model_paths['accelerator_type'])

        print("This is the dictionary", model_paths)
        converted = convert_checkpoints_op(
            keras_gcs_model=trainer.output, model_paths=model_paths
        ).set_memory_limit("50G").set_cpu_limit("12.0").set_accelerator_limit(1).add_node_selector_constraint(model_paths['accelerator_type'])

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

    from config import Config
    from util import get_model_paths_and_config
    
    model_paths = get_model_paths_and_config(Config.MODEL_NAME)
    compiler.Compiler().compile(
        pipeline_func=fine_tune_pipeline, package_path="fine_tune_pipeline.json"
    )
    vertexai.init(project=Config.PROJECT_ID, location=Config.REGION)
    vertex_pipelines_job = vertexai.pipeline_jobs.PipelineJob(
        display_name="test-fine_tune_pipeline",
        template_path="fine_tune_pipeline.json",
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
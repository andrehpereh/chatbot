from kfp import dsl
import kfp as kfp
from kfp.dsl import OutputPath, Artifact, InputPath, PipelineTaskFinalStatus, ExitHandler
from kfp import compiler
from config import Config
from google.cloud import aiplatform as vertexai
import os

TAG_NAME = os.environ.get('TAG_NAME', 'masterv6') 

@dsl.component(base_image='python:3.9', packages_to_install=['google-cloud-bigquery'])
def send_pipeline_completion_email_op(
    project: str,
    status: PipelineTaskFinalStatus,
    smtp_server: str = 'smtp.gmail.com',
    smtp_port: int = 587,
    sender_email: str = 'andrehpereh96@gmail.com',
    recipient_emails: str = "andrehpereh@gmail.com",
    email_password: str = "ssuy rubm kzge juid"
):
    import smtplib
    from email.mime.text import MIMEText
    from google.cloud import bigquery
    recipient_emails = [recipient_emails]
    
    DATASET_ID = 'chatbot' # This should be moved to a config file
    USER_TRAINING_STATUS = 'user_training_status' # This should be moved to a config file
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
    if status.state == "SUCCEEDED":
        client = bigquery.Client(project)
        print("This is the client", client)
        table_ref = client.dataset(DATASET_ID).table(USER_TRAINING_STATUS)
        table = client.get_table(table_ref)
        row_to_insert = {
            'email': recipient_emails,
            'training_status': 1
        }
        client.insert_rows(table, [row_to_insert]) 
        errors = client.insert_rows(table, [row_to_insert])
        if errors:  # Check if there were errors
            print("The model has been trained, but error updating training_status for {}: {}".format(email_password, errors))
        else:
            print("User training has been updated")

    print('Email sent!')




@dsl.component(
    base_image =f"gcr.io/{Config.PROJECT_ID}/gemma-chatbot-data-preparation:{TAG_NAME}"
)
def data_preparation_op(
    bucket_name: str,
    directory: str,
    dataset_path: OutputPath('Dataset'),
    pair_count: int=4,
    data_augmentation_iter: int=4
):
    import data_ingestion
    import json
    from vertexai.preview.generative_models import GenerativeModel
    gemini_pro_model = GenerativeModel("gemini-1.0-pro")
    formatted_messages = data_ingestion.data_preparation(
        bucket_name=bucket_name, directory=directory, gemini_pro_model=gemini_pro_model,
        pair_count=pair_count, data_augmentation_iter=data_augmentation_iter
    )
    with open(dataset_path, 'w') as f:
        json.dump(formatted_messages, f)


@dsl.component(
    base_image = f"gcr.io/{Config.PROJECT_ID}/gemma-chatbot-fine-tunning:{TAG_NAME}"
)
def fine_tunning(
  dataset_path: InputPath('Dataset'),
  model_paths: dict,
  fine_tune_flag: bool,
  epochs: int,
  model_name: str,
  bucket_name: str
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
    util.upload2bs(
        local_directory = model_paths['finetuned_model_dir'], bucket_name = bucket_name,
        destination_subfolder = model_paths['fine_tuned_keras_blob']
    )
    model_gcs = "gs://{}/{}".format(bucket_name, model_paths['fine_tuned_keras_blob'])  
    print("This is the storage bucket", model_gcs)
    return model_gcs
    

@dsl.component(
    base_image = f"gcr.io/{Config.PROJECT_ID}/gemma-chatbot-fine-tunning:{TAG_NAME}"
)
def convert_checkpoints_op(
  keras_gcs_model: str,
  model_paths: dict,
  bucket_name: str
) -> str:
    import conversion_function
    import os
    import util
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



@dsl.component(base_image='python:3.9', packages_to_install=['google-cloud-bigquery'])
def update_user_endpoint(
    endpoint_resource: str,
    email: str,
    project: str
):

    import os
    from google.cloud import bigquery
    DATASET_ID = 'chatbot' # This should be moved to a config file
    USER_TRAINING_STATUS = 'user_training_status' # This should be moved to a config file
    #This part can be wrapped in a function
    import json
    data = json.loads(endpoint_resource)
    resource_uri = data['resources'][0]['resourceUri']

    print("This is the passed end pooint", endpoint_resource)
    print(dir(endpoint_resource))
    print(type(endpoint_resource))
    print("This is the project", project)
    
    client = bigquery.Client(project)
    print("This is the client", client)
    table_ref = client.dataset(DATASET_ID).table(USER_TRAINING_STATUS)
    table = client.get_table(table_ref)
    row_to_insert = {
        'email': email,
        'end_point': resource_uri,
        'training_status': True
    }
    client.insert_rows(table, [row_to_insert]) 
    errors = client.insert_rows(table, [row_to_insert])
    if errors:  # Check if there were errors
        print("The model has been trained, but error updating resource_uri for {}: {}".format(email, errors))
    else:
        print("User training has been updated")
    print("End point has been stored.")


@dsl.pipeline(name="Model deployment.")
def fine_tune_pipeline(
    project: str = os.environ.get('PROJECT_ID') ,
    bucket_name: str = "able-analyst-416817-chatbot-v1",
    directory: str = "input_data/andrehpereh",
    serving_image: str = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240220_0936_RC01",
    fine_tune_flag: bool = False,
    epochs: int = 3,
    model_name: str = 'gemma_2b_en',
    pair_count: int = 6,
    data_augmentation_iter: int = 4
):

    from google_cloud_pipeline_components.types import artifact_types
    from google_cloud_pipeline_components.v1.endpoint import (EndpointCreateOp, ModelDeployOp)
    from google_cloud_pipeline_components.v1.model import ModelUploadOp
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
    # This should come from a dataset instead of hardcoding it.
    email = f'{Config.USER_NAME}@gmail.com'
    send_email = send_pipeline_completion_email_op(recipient_emails = email, project=project)
    with ExitHandler(send_email):
        whatup = data_preparation_op(
            bucket_name = bucket_name, directory = directory,
            pair_count=pair_count, data_augmentation_iter=data_augmentation_iter
        )

        trainer = fine_tunning(
            dataset_path=whatup.outputs['dataset_path'], model_paths=model_paths, fine_tune_flag=fine_tune_flag,
            epochs=epochs, model_name=model_name, bucket_name = bucket_name
        )
        trainer.set_memory_limit("50G").set_cpu_limit("12.0").set_accelerator_limit(1).add_node_selector_constraint(model_paths['accelerator_type'])

        print("This is the dictionary", model_paths)
        converted = convert_checkpoints_op(
            keras_gcs_model=trainer.output, model_paths=model_paths, bucket_name = bucket_name
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
            display_name=f"Mini {Config.USER_NAME} model uploaded.",
            unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],
        )
        model_upload_op.after(import_unmanaged_model_task)

        endpoint_create_op = EndpointCreateOp(
            project=project,
            display_name=f"End point created for {Config.USER_NAME}",
        )

        model_end_point = ModelDeployOp(
            endpoint=endpoint_create_op.outputs["endpoint"],
            model=model_upload_op.outputs["model"],
            deployed_model_display_name=f"Model {model_paths['model_name_vllm']}, for user:{Config.USER_NAME}",
            dedicated_resources_machine_type=model_paths['machine_type'],
            dedicated_resources_min_replica_count=1,
            dedicated_resources_max_replica_count=1,
            dedicated_resources_accelerator_type=model_paths['accelerator_type'],
            dedicated_resources_accelerator_count=model_paths['accelerator_count']
        )
        print("This is the project", project)
        update_user_endpoint(endpoint_resource=model_end_point.outputs["gcp_resources"], email=email, project=project)

if __name__ == '__main__':
    
    os.environ['TRAIN_DATA_DIR'] = 'andrehpereh/input_data' 
    os.environ['BUCKET_NAME'] = 'personalize-chatbots-v1'
    from kfp import compiler
    from google.cloud import aiplatform as vertexai
    from config import Config
    print("This is the model name", Config.MODEL_NAME, "Ahuevito")
    print("This is the directory", Config.TRAIN_DATA_DIR, "Ahuevito")
    print("This is the BUCKET_NAME", Config.BUCKET_NAME, "Ahuevito")
    print("This is the FINE_TUNE_FLAG", Config.FINE_TUNE_FLAG, "Ahuevito")
    print("This is the EPOCHS", Config.EPOCHS, "Ahuevito")
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
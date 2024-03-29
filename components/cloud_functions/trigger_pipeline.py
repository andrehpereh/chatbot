import os
from kfp import dsl
PROD_TAG = os.environ.get('PROD_TAG', 'master_v1') 

@dsl.container_component
def triger_pipeline_component():
    return dsl.ContainerSpec(
      image=f"gcr.io/able-analyst-416817/gemma-chatbot-pipeline-app:{PROD_TAG}",
      args=['python', 'pipeline.py']
    )

@dsl.pipeline(name="Trigger Pipeline")
def pipeline_trigger_pipeline_chatbot(
):
    task = triger_pipeline_component()    
    task.set_env_variable('MODEL_NAME', os.environ.get('MODEL_NAME'))
    task.set_env_variable('TRAIN_DATA_DIR', os.environ.get('TRAIN_DATA_DIR'))
    task.set_env_variable('BUCKET_NAME', os.environ.get('BUCKET_NAME'))
    task.set_env_variable('FINE_TUNE_FLAG', os.environ.get('FINE_TUNE_FLAG'))
    task.set_env_variable('USER_NAME', os.environ.get('USER_NAME'))
    task.set_env_variable('PROJECT_ID', os.environ.get('PROJECT_ID'))
    task.set_env_variable('EPOCHS', os.environ.get('EPOCHS'))



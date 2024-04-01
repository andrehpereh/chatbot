import base64
import functions_framework
import logging
import os
import json
import logging
from kfp import compiler
from google.cloud import aiplatform as vertexai

@functions_framework.cloud_event
def trigger_pipeline_cloud_function(cloud_event):
    try:
        parameters = base64.b64decode(cloud_event.data["message"]["data"])
        parameters = json.loads(parameters.decode('utf-8'))
        print("Parameters fine tunning personalized bot:", parameters)
        print("This is the production pipeline branch; should not be hardcoded.")
        if len(parameters['tag_version']) > = 0:
            os.environ['TAG_NAME'] = parameters['tag_version']
        os.environ['USER_NAME'] = parameters['user_name']
        os.environ['MODEL_NAME'] = parameters['model_name']
        os.environ['MY_API_KEY'] = parameters['project_id']
        os.environ['BUCKET_NAME'] = parameters['bucket_name']
        os.environ['FINE_TUNE_FLAG'] = 'True'
        os.environ['EPOCHS'] = parameters['epochs']
        os.environ['PROJECT_ID'] = parameters['project_id']
        os.environ['TRAIN_DATA_DIR'] = parameters['blob_folder']
    
        pipeline_name = f"trigger_fine_tune_pipeline_{parameters['user_name']}.json"

        print("This is path name", pipeline_name)
        from trigger_pipeline import pipeline_trigger_pipeline_chatbot

        compiler.Compiler().compile(
            pipeline_func=pipeline_trigger_pipeline_chatbot, package_path=pipeline_name
        )
        vertexai.init(project=parameters['project_id'])
        vertex_pipelines_job = vertexai.pipeline_jobs.PipelineJob(
            display_name="cloud_function_trigger_fine_tunning_pipeline",
            template_path=pipeline_name
        )
        vertex_pipelines_job.run()
    except Exception as e: 
        logging.error(f"Pipeline trigger failed: {e}")

from flask import Flask, render_template, request, jsonify, redirect, url_for
from google.cloud import bigquery
import bcrypt
import os

app = Flask(__name__)

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START aiplatform_predict_custom_trained_model_sample]
from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import re

# Connect to BigQuery
os.environ['PROJECT_ID'] = 'able-analyst-416817'
DATASET_ID = 'chatbot'
TABLE_ID = 'users'


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
    user_input: str = input
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    prompt_input = f"Sender:\n{user_input}\n\nAndres Perez:\n",
    instances={'prompt': prompt_input[0] , 'max_tokens': 256, 'temperature': 1.4, 'top_p': 0.8, 'top_k': 4}
    
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )

    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    pattern = r"Perez:\nOutput:\n(.*)"
    match = re.search(pattern, predictions[0])

    if match:
        return match.group(1)
    else:
        return "Error: Prediction not found in the response."


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Data validation
        error_message = None
        if not email or not password or not confirm_password:
            error_message = 'Please fill in all fields.'
        elif password != confirm_password:
            error_message = 'Passwords do not match.'
        # Add more validation rules as needed (e.g., email format, password strength)

        if error_message:
            return error_message, 400

        # Hash the password for security
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        client = bigquery.Client(os.environ.get('PROJECT_ID'))
        table_ref = client.dataset(DATASET_ID).table(TABLE_ID)
        table = client.get_table(table_ref)

        # Insert data with error handling
        row_to_insert = [(email, hashed_password.decode('utf-8'))]
        errors = client.insert_rows(table, row_to_insert)
        if errors:  # Check if there were errors
            return 'Error submitting data: {}'.format(errors), 500
        else:
            return redirect(url_for('chat_page'))

@app.route('/login', methods=['POST'])
def login():
    print("Si entro en este pedo")
    if request.method == 'POST':
        email = request.form['email']
        print("Si entro en este email", email)
        password = request.form['password']
        print("Si entro en este password", password)

        # Fetch user data from BigQuery
        client = bigquery.Client(os.environ.get('PROJECT_ID'))
        query = f"SELECT password_hash FROM `{os.environ.get('PROJECT_ID')}.{DATASET_ID}.{TABLE_ID}` WHERE username = '{email}'"
        print("This is el query...", query)
        results = client.query(query).result()
        print("This is the results", results, type(results))
        stored_password_hash = None
        for row in results:
            stored_password_hash = row.password_hash  # Assuming 'password' is the column name

        # Verify password
        if stored_password_hash and bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
            # Successful login
            return redirect(url_for('chat_page'))
        else:
            # Invalid credentials
            return 'Invalid email or password', 401

@app.route('/chat_page')
def chat_page():
    return render_template('chat.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json['message']
    # Placeholder chatbot code
    #response = predict_custom_trained_model_sample(
     #   project="24796876098",
      #  endpoint_id="7974742443096539136",
       # location="us-central1",
        #user_input= user_message,
    #)
    print(user_message)
    print("Esta es la respuesta")
    response = "Hello"
    print(response)
    return jsonify({'message': response})

if __name__ == '__main__':
    print("Pues si empezo a correr esta madre")
    app.run(host='0.0.0.0', port=5000, debug=True) 
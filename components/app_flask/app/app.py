from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from google.cloud import bigquery, storage, pubsub_v1
from werkzeug.datastructures import FileStorage
import bcrypt, os, base64, json
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Constants for BigQuery, Storage, and Pub/Sub
DATASET_ID = 'chatbot'
USERS_TABLE = 'users'
USER_TRAINING_STATUS = 'user_training_status'
BUCKET_NAME = f"{os.environ.get('PROJECT_ID')}-personalized-chatbots-v1"
PUBSUB_TOPIC = 'your-pipeline-trigger-topic'

# Import libraries for AI Platform predictions
from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import re

# Log project ID and bucket name
logger.debug(f"Project ID from environment: {os.environ.get('PROJECT_ID')}")
logger.debug(f"Using bucket: {BUCKET_NAME}")


def predict_custom_trained_model_sample(project: str, endpoint_id: str, location: str = "us-central1", api_endpoint: str = "us-central1-aiplatform.googleapis.com", user_input: str = input):
    """Predicts text using a custom-trained Vertex AI model.

    Args:
        project: The Google Cloud project ID.
        endpoint_id: The ID of the Vertex AI endpoint.
        location: The region where the endpoint is located.
        api_endpoint: The API endpoint of Vertex AI.
        user_input: The user's input text.

    Returns:
        The predicted text from the model.
    """

    logger.debug("Function predict_custom_trained_model_sample started")

    try:
        # Prepare prompt input for prediction
        prompt_input = f"Sender:\n{user_input}\n\nAndres Perez:\n"
        conversation_track = session.get('conversation_track')[-3:]
        logger.debug(f"Last 3 conversation track items: {conversation_track}")
        conversation_track_str = "\n\n".join(conversation_track + [prompt_input])
        logger.debug(f"Joined input for prediction: {conversation_track_str}")
        instances = {'prompt': conversation_track_str, 'max_tokens': 1024, 'temperature': 1, 'top_p': 0.7, 'top_k': 6}

        # Initialize AI Platform prediction client
        client_options = {"api_endpoint": api_endpoint}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        # Format instances for prediction request
        instances = instances if isinstance(instances, list) else [instances]
        instances = [json_format.ParseDict(instance_dict, Value()) for instance_dict in instances]

        # Set prediction parameters (empty in this case)
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())

        # Get endpoint path
        endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)

        # Send prediction request and get response
        response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
        predictions = response.predictions

        # Extract predicted text using regex
        pattern = r"Perez:\nOutput:\n(.*)"
        match = re.search(pattern, predictions[0])

        if match:
            logger.info(f"Successful prediction: {match.group(1)}")
            conversation_track.append(prompt_input + str(match.group(1)))
            logger.debug(f"Updated conversation track: {conversation_track}")
            session['conversation_track'] = conversation_track
            return match.group(1)
        else:
            logger.error("Prediction not found in the response.")
            return "Error: Prediction not found in the response."

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise  # Re-raise for error handling


def extract_info_from_endpoint(url):
    """Extracts location, endpoint, and project information from a given Vertex AI endpoint URL.

    Args:
        url: The Vertex AI endpoint URL string.

    Returns:
        A dictionary containing the extracted values:
            locations: The region.
            endpoints: The endpoint ID.
            projects: The project ID.
    """

    pattern = r"\/projects\/([^\/]+)\/locations\/([^\/]+)\/endpoints\/([^\/]+)\/operations\/([^\/]+)"
    logger.debug(f"Using regex pattern: {pattern}")
    match = re.search(pattern, url)
    logger.debug(f"Regex match result: {match}")
    if match:
        return {"projects": match.group(1), "locations": match.group(2), "endpoints": match.group(3)}
    else:
        return None


@app.route('/signup', methods=['POST'])
def signup():
    """Handles user signup requests."""

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Validate user input
        error_message = None
        if not email or not password or not confirm_password:
            error_message = 'Please fill in all fields.'
        elif password != confirm_password:
            error_message = 'Passwords do not match.'

        if error_message:
            return error_message, 400

        # Hash password for security
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Insert user into BigQuery
        client = bigquery.Client(os.environ.get('PROJECT_ID'))
        table_ref = client.dataset(DATASET_ID).table(USERS_TABLE)
        table = client.get_table(table_ref)
        row_to_insert = {'email': email, 'password_hash': hashed_password.decode('utf-8')}
        client.insert_rows(table, [row_to_insert])
        errors = client.insert_rows(table, [row_to_insert])
        if errors:
            logger.error(f"Error submitting data to BigQuery: {errors}")
            return 'Error submitting data: {}'.format(errors), 500
        else:
            logger.info(f"User signup successful: {email}")
            session['email'] = email
            return redirect(url_for('upload'))


@app.route('/login', methods=['POST'])
def login():
    """Handles user login requests."""

    logger.debug("Login endpoint called")
    if request.method == 'POST':
        email = request.form['email']
        logger.debug(f"Login attempt for email: {email}")
        password = request.form['password']

        # Fetch user data from BigQuery
        client = bigquery.Client(os.environ.get('PROJECT_ID'))

        # Check password
        query = f"SELECT password_hash FROM `{os.environ.get('PROJECT_ID')}.{DATASET_ID}.{USERS_TABLE}` WHERE email = '{email}'"
        logger.debug(f"Executing BigQuery password query: {query}")
        results = client.query(query).result()
        logger.debug(f"BigQuery password query results: {results}")
        stored_password_hash = None
        for row in results:
            logger.debug(f"Retrieved password hash from BigQuery: {row}")
            stored_password_hash = row.password_hash

        # Check training status
        query = f"""
            SELECT training_status, end_point
            FROM `{os.environ.get('PROJECT_ID')}.{DATASET_ID}.{USER_TRAINING_STATUS}`
            WHERE email = '{email}'
            ORDER BY created_at DESC
            LIMIT 1
        """
        logger.debug(f"Executing BigQuery training status query: {query}")
        results = client.query(query).result()
        logger.debug(f"BigQuery training status query results: {results}")
        user_training_status = None
        endpoint_uri = None
        for row in results:
            logger.debug(f"Retrieved training status from BigQuery: {row}")
            user_training_status = row.training_status
            endpoint_uri = row.end_point

        # Verify password and redirect based on training status
        if stored_password_hash and bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
            session['email'] = email
            if user_training_status:
                endpoint_details = extract_info_from_endpoint(endpoint_uri)
                logger.debug(f"Setting session variables for endpoint: {endpoint_details}")
                session['endpoint'] = endpoint_details["endpoints"]
                session['location'] = endpoint_details["locations"]
                session['project'] = endpoint_details["projects"]
                logger.info(f"User logged in with existing training: {email}")
                return redirect(url_for('chat_page'))
            else:
                logger.info(f"User logged in without training: {email}")
                return redirect(url_for('upload'))
        else:
            logger.warning(f"Invalid login attempt for email: {email}")
            return 'Invalid email or password', 401


@app.route('/upload')
def upload():
    """Renders the file upload page for users to provide training data."""

    email = session.get('email')
    logger.debug(f"Upload page accessed by email: {email}")
    if not email:
        logger.info("Redirecting to home page due to missing email in session")
        return redirect(url_for('home'))
    logger.debug("Rendering upload template")
    return render_template('upload.html')


@app.route('/handle_upload', methods=['POST'])
def handle_upload():
    """Handles file upload and triggers the training pipeline."""

    files_metadata = []
    email = session.get('email')
    logger.debug(f"Handling file upload for email: {email}")
    code_version = request.form.get('code_version')
    logger.debug(f"Code version for training: {code_version}")
    model_name = request.form.get('model_name')
    epochs = request.form.get('epochs')
    logger.debug(f"Selected model: {model_name}, Epochs: {epochs}")
    user_name = re.match(r'^([^@]+)', str(email)).group(1)

    # Initialize Pub/Sub publisher and prepare storage paths
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(os.environ.get('PROJECT_ID'), PUBSUB_TOPIC)
    logger.debug(f"Publishing message to Pub/Sub topic: {topic_path}")
    blob_folder = os.path.join(user_name, 'input_data')

    if not email:
        logger.info("Redirecting to home page due to missing email in session")
        return redirect(url_for('home'))

    # Upload files to Cloud Storage
    for file in request.files.getlist('files'):
        logger.debug(f"Processing uploaded file: {file.filename}")
        client = storage.Client(os.environ.get('PROJECT_ID'))
        bucket = client.get_bucket(BUCKET_NAME)
        blob_string = os.path.join(blob_folder, file.filename)
        blob = bucket.blob(blob_string)
        blob.upload_from_string(FileStorage(file).stream.read())
        logger.info(f"Uploaded file to Cloud Storage: {blob.name}")
        files_metadata.append({"file_path": f"gs://{BUCKET_NAME}/{blob_string}", "filename": file.filename})

    # Prepare and publish Pub/Sub message
    message_data = {
        "user_name": user_name,
        "files": files_metadata,
        "blob_folder": blob_folder,
        "model_name": model_name,
        "epochs": epochs,
        "bucket_name": BUCKET_NAME,
        "tag_version": code_version,
        "project_id": os.environ.get('PROJECT_ID')

    }
    message_data_json = json.dumps(message_data)
    message_data_bytes = message_data_json.encode('utf-8')
    logger.debug(f"Publishing message data to Pub/Sub: {message_data_bytes}")
    publisher.publish(topic_path, message_data_bytes)

    # Update training status in BigQuery
    client = bigquery.Client(os.environ.get('PROJECT_ID'))
    table_ref = client.dataset(DATASET_ID).table(USER_TRAINING_STATUS)
    table = client.get_table(table_ref)
    row_to_insert = {'email': email, 'training_status': False}
    logger.debug(f"Inserting training status into BigQuery: {row_to_insert}")
    client.insert_rows(table, [row_to_insert])
    errors = client.insert_rows(table, [row_to_insert])
    if errors:
        logger.error(f"Error submitting data to BigQuery: {errors}")
        return 'Error submitting data: {}'.format(errors), 500
    else:
        logger.info("File upload and training initiation successful!")
        return "Upload Successful!", 200


@app.route('/home')
def home():
    """Renders the home page of the application."""
    logger.debug("Rendering home page")
    return render_template('index.html')


@app.route('/chat_page')
def chat_page():
    """Renders the chat page where users interact with the chatbot."""

    email = session.get('email')
    if not email:
        logger.info("Redirecting to home page due to missing email in session")
        return redirect(url_for('home'))
    logger.debug("Initializing conversation track and rendering chat page")
    session['conversation_track'] = []
    return render_template('chat.html')


@app.route('/send_message', methods=['POST'])
def send_message():
    """Handles sending user messages to the chatbot and returning responses."""

    user_message = request.json['message']
    logger.debug(f"Received user message: {user_message}")
    email = session.get('email')
    if not email:
        logger.info("User not logged in. Redirecting to home")
        return redirect(url_for('home'))

    endpoint = session.get('endpoint')
    project = session.get('project')
    location = session.get('location')
    logger.debug(f"Session data: endpoint={endpoint}, project={project}, location={location}")
    try:
        response = predict_custom_trained_model_sample(
            project=project,
            endpoint_id=endpoint,
            location=location,
            user_input=user_message,
        )
    except Exception as e:
        logger.exception(f"Error in chatbot prediction: {e}")
        response = "An error occurred. Please try again later."
    logger.debug(f"Returning chatbot response: {response}")
    return jsonify({'message': response})


if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=5000, debug=True)
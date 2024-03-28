from flask import Flask, request
import base64  # For decoding Pub/Sub data
import json

app = Flask(__name__)

@app.route('/', methods=['POST'])
def process_message():
    if request.headers.get('Content-Type') == 'application/json':
        json_parameters = request.get_json()
        parameters = json.loads(json_parameters)
    else:  # Likely base64 encoded from Pub/Sub
        data = request.data
        print(data)
        json_parameters = base64.b64decode(data).decode('utf-8')
        print(json_parameters)
        parameters = json.loads(json_parameters)
    print(parameters)

    return 'Message processed', 200 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 

import os
import sys
import unittest
from unittest.mock import patch
os.chdir(os.path.dirname(__file__))
sys.path.append('.')
import app  # Assuming your Flask app code is in app.py
print(os.getcwd())

class TestChatbotApp(unittest.TestCase):

    def setUp(self):
        self.app = app.app.test_client()
        self.app.testing = True

    def test_chat_page_logged_out(self):
        response = self.app.get('/chat_page')
        self.assertEqual(response.status_code, 302)  # Redirect to login
        print(response.location)
        self.assertEqual(response.location, '/home') 

    @patch('app.predict_custom_trained_model_sample')
    def test_send_message_logged_in(self, mock_predict):
        mock_predict.return_value = "Mock response from model"
        with self.app.session_transaction() as sess:
            sess['email'] = 'test@example.com'
            sess['endpoint'] = 'mock_endpoint'
            sess['project'] = 'mock_project'
            sess['location'] = 'mock_location'
            sess['conversation_track'] = []
        response = self.app.post('/send_message', json={'message': 'Hello'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'message': 'Mock response from model'})

    def test_send_message_logged_out(self):
        response = self.app.post('/send_message', json={'message': 'Hello'})
        self.assertEqual(response.status_code, 302)  # Redirect to login
        self.assertEqual(response.location, '/home') 

    def test_extract_info_from_endpoint_valid(self):
        url = "/projects/test-project/locations/us-central1/endpoints/mock_endpoint/operations/12345"
        result = app.extract_info_from_endpoint(url)
        self.assertEqual(result['projects'], 'test-project')
        self.assertEqual(result['locations'], 'us-central1')
        self.assertEqual(result['endpoints'], 'mock_endpoint')

    def test_extract_info_from_endpoint_invalid(self):
        url = "invalid_url"
        result = app.extract_info_from_endpoint(url)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.app()

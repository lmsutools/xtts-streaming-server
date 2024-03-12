from locust import HttpUser, task, between, events
import json
import random

class TTSTestUser(HttpUser):
    wait_time = between(1, 2)  # Specifies the wait time between executing tasks
    host = "https://c762ac7d104d1.notebooksg.jarvislabs.net"  # Predefined host URL

    def on_start(self):
        """Load the default speaker details at the start of the test"""
        with open("default_speaker.json", "r") as file:
            self.speaker = json.load(file)

    @task
    def send_tts_request(self):
        """Sends a POST request to the TTS server with a randomly chosen message"""
        messages = [
            "Short message.",
            "This is a medium length message for testing purposes.",
            "Here is a longer message intended to test the text-to-speech server's handling of more substantial inputs."
        ]
        message = random.choice(messages)

        payload = self.speaker
        payload.update({
            "text": message,
            "language": "en",
            "stream_chunk_size": "15"
        })

        # Sending the POST request
        self.client.post("/tts_stream", json=payload)

# Example of using the request event to log the response time
@events.request.add_listener
def log_response_time(request_type, name, response_time, response_length, exception, **kwargs):
    if exception:
        print(f"Request to {name} failed with exception {exception}")
    else:
        print(f"Response time for {name}: {response_time} ms")


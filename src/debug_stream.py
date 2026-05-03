import sys
import os
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.api.server import app

client = TestClient(app)

def run():
    print("Sending request...")
    with client.stream("POST", "/api/chat/stream", json={"message": "hello", "session_id": "123"}, auth=("admin", "admin")) as response:
        for line in response.iter_lines():
            print(f"RAW LINE: {line}")
            if "error" in line.lower():
                break

if __name__ == "__main__":
    run()

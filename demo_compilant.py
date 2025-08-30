# demo_compliant.py
import requests
import time

# 1. Start a compliant session
resp = requests.post(
    "http://localhost:8000/ask",
    files={},
    data={"question": "Does the login flow meet MFA requirements under Policy X?"}
)
session_id = resp.json()["session_id"]
print(f"Session ID: {session_id}")

# 2. Poll for result
for _ in range(30):
    r = requests.get(f"http://localhost:8000/result/{session_id}")
    print("Result status:", r.json())
    if r.json().get("decision") or r.json().get("status") == "completed":
        print("Final Output:", r.json())
        break
    time.sleep(2)
# demo_hitl.py
import requests
import time
import asyncio
import websockets
import json
import threading

def start_session():
    resp = requests.post(
        "http://localhost:8000/ask",
        files={},
        data={"question": "Does the login flow meet MFA requirements under Policy X?"}
    )
    return resp.json()["session_id"]

async def hitl_client(session_id):
    async with websockets.connect(f"ws://localhost:8000/connect/{session_id}") as ws:
        print(f"Connected to session {session_id}. Waiting for HITL requests...")
        hitl_count = 0
        while hitl_count < 2:
            msg = await ws.recv()
            data = json.loads(msg)
            if "hitl_request" in data:
                req = data["hitl_request"]
                print(f"\n--- HITL REQUEST ---")
                print(f"Type: {req.get('type')}")
                print(f"Prompt: {req.get('prompt')}")
                if req.get("required_artifact"):
                    print(f"Required artifact: {req['required_artifact']}")
                # Simulate responses
                if req.get('type') == "clarification":
                    response = "All users"
                elif req.get('type') == "upload_request":
                    response = "uploaded"
                else:
                    response = "approved"
                await ws.send(json.dumps({
                    "request_id": data["request_id"],
                    "payload": response
                }))
                hitl_count += 1
            elif "progress_update" in data:
                print(f"[Progress] {data['progress_update']}")
            else:
                print(f"[Other message] {data}")

def run_hitl(session_id):
    asyncio.run(hitl_client(session_id))

if __name__ == "__main__":
    session_id = start_session()
    print(f"Session ID: {session_id}")
    # Start HITL client in a thread
    t = threading.Thread(target=run_hitl, args=(session_id,))
    t.start()
    # Poll for result
    for _ in range(60):
        r = requests.get(f"http://localhost:8000/result/{session_id}")
        print("Result status:", r.json())
        if r.json().get("decision") or r.json().get("status") == "completed":
            print("Final Output:", r.json())
            break
        time.sleep(2)
    t.join()
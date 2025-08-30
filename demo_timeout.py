# demo_timeout.py
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
        # Simulate NOT responding to HITL (timeout)
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if "hitl_request" in data:
                req = data["hitl_request"]
                print(f"\n--- HITL REQUEST (will not respond, simulating timeout) ---")
                print(f"Type: {req.get('type')}")
                print(f"Prompt: {req.get('prompt')}")
                # Do not send a response, just wait for timeout
                break
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
    # Poll for result (wait for timeout)
    for _ in range(60):
        r = requests.get(f"http://localhost:8000/result/{session_id}")
        print("Result status:", r.json())
        if r.json().get("decision") or r.json().get("status") == "completed":
            print("Final Output:", r.json())
            break
        time.sleep(5)
    t.join()
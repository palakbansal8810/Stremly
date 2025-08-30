import websockets
import asyncio
import json

async def connect(session_id):
    async with websockets.connect(f"ws://localhost:8000/connect/{session_id}") as ws:
        print(f"Connected to session {session_id}. Waiting for HITL requests...")
        while True:
            msg = await ws.recv()
            print(f"Received message: {msg}")
            data = json.loads(msg)
            # If this is a HITL request, show details
            if "hitl_request" in data:
                req = data["hitl_request"]
                print(f"\n--- HITL REQUEST ---")
                print(f"Type: {req.get('type')}")
                print(f"Prompt: {req.get('prompt')}")
                if req.get("required_artifact"):
                    print(f"Required artifact: {req['required_artifact']}")
                response = input("Your response: ")
                await ws.send(json.dumps({
                    "request_id": data["request_id"],
                    "payload": response
                }))
            elif "progress_update" in data:
                print(f"[Progress] {data['progress_update']}")
            else:
                print(f"[Other message] {data}")

if __name__ == "__main__":
    session_id = input("Enter session_id: ")
    asyncio.run(connect(session_id))
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse
import uuid
import asyncio
from src.orchestrator import ComplianceOrchestrator, HitlRequest
from src.retriever import Retriever
import pymongo
from typing import Optional
import logging
from src.config import Config  # Import Config

app = FastAPI()
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
config = Config.from_env()  # Initialize Config
retriever = Retriever(config=config)  # Pass config to Retriever
orch = ComplianceOrchestrator(retriever, config, mongo_client)  # Pass config
active_connections = {}  # session_id -> list of WebSocket

@app.websocket("/connect/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in active_connections:
        active_connections[session_id] = []
    active_connections[session_id].append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            request_id = data.get("request_id")
            payload = data.get("payload")
            print(f"Received HITL response for request_id {request_id}: {payload}")
            await orch.handle_hitl_response(session_id, payload)
    except WebSocketDisconnect:
        active_connections[session_id].remove(websocket)
        logging.info(f"WS disconnected for {session_id}")

async def broadcast(session_id: str, message: dict):
    if session_id in active_connections:
        for ws in active_connections[session_id][:]:
            try:
                await ws.send_json(message)
                print(f"Sent to {session_id}: {message}")
            except:
                active_connections[session_id].remove(ws)

async def send_hitl(session_id: str, request: HitlRequest):
    request_id = str(uuid.uuid4())
    await broadcast(session_id, {"hitl_request": request.dict(), "request_id": request_id})
    await broadcast(session_id, {"progress_update": {"stage": "awaiting_human", "status": "pending"}})

orch.send_hitl_request = send_hitl
from fastapi import Form
@app.post("/ask")
async def ask(
    question: str = Form(...),
    attachment: Optional[UploadFile] = File(None)
):
    session_id = str(uuid.uuid4())
    image_data = await attachment.read() if attachment else None
    await broadcast(session_id, {"progress_update": {"stage": "planner_started", "status": "running"}})
    asyncio.create_task(orch.run(session_id, question, image_data))
    print(f"Started session {session_id} for question: {question}")
    return {"session_id": session_id, "job_id": session_id}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    session = mongo_client["compliance_db"]["sessions"].find_one({"_id": job_id})
    if session and session.get("status") == "completed":
        print(f"Returning result for job_id {job_id}")
        return JSONResponse(content=session["result"])
    return {"status": session.get("status", "pending")}

@app.post("/hitl")
async def post_hitl(session_id: str, request_id: str, response_type: str, payload: str):
    await orch.handle_hitl_response(session_id, payload)
    print(f"Received HITL response for session {session_id}, request {request_id}")
    return {"status": "received"}

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    session = mongo_client["compliance_db"]["sessions"].find_one({"_id": session_id})
    if session:
        session["costs"] = {"total_tokens": 0}  # Stub for token counting
    return session or {"error": "Not found"}
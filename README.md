# Stremly — AI Compliance & Risk Orchestrator

## Overview
Stremly answers compliance/risk queries by orchestrating parallel AI agents over your policies, specs, code, and screenshots. It supports human-in-the-loop (HITL) review via WebSocket and always returns structured, cited JSON.

---

## Quick Start

1. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

2. **Set up environment**
   - Add your API keys to `.env` (do NOT commit this file).
   - Ensure MongoDB and Redis are running locally.

3. **Ingest your data**
   - Place files in `data/policies/`, `data/product_specs/`, `data/code/`, `data/screenshots/`
   - Run:
     ```sh
     python test_ingest.py
     ```

4. **Start the server**
   ```sh
   uvicorn api.server:app --reload
   ```

5. **Start the HITL client (in a new terminal)**
   ```sh
   python client.py
   ```
   - Enter the `session_id` you get from `/ask`.

6. **Ask a question**
   ```sh
   curl -X POST -F "question=Does the login flow meet MFA requirements under Policy X?" http://localhost:8000/ask
   ```

7. **Check results**
   ```sh
   curl http://localhost:8000/result/<session_id>
   ```

---

## API

- `POST /ask` — Start a session (`question`, optional `attachment`)
- `GET /result/{job_id}` — Get final result JSON
- `WS /connect/{session_id}` — HITL client WebSocket

---

## Demo

- `python demo_compilant.py` — Normal compliant run (no HITL)
- `python demo_hitl.py` — Simulates 2 HITL steps
- `python demo_timeout.py` — Simulates HITL timeout

---

## Notes

- Never commit .env or secrets.
- All outputs are strict JSON with citations and rationale.
- HITL prompts appear in the client terminal if clarification or approval is needed.

---
```
Let me know if you want it even shorter or want to add a usage example!Let me know if you want it even shorter or want to add a usage example!
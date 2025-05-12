# Realtime Voice Agent: Project Plan & Checklist

This document outlines the plan for refactoring the existing Python voice agent into a FastAPI backend service, creating a simple Next.js frontend for testing, and deploying the entire system to Render.com.

## Phase 1: Backend Refactoring & FastAPI Implementation (Python)

*   **Goal:** Transform `QuickAgent.py` into a FastAPI backend service that handles streaming audio input, STT, LLM processing, and TTS, ready for a WebSocket connection from a frontend.

*   **Checklist:**
    *   [ ] **1.1: Project Setup & Dependencies**
        *   [ ] Create a new main application file, e.g., `main.py`, for the FastAPI app.
        *   [ ] Update `requirements.txt`:
            *   [ ] Add `fastapi`
            *   [ ] Add `uvicorn[standard]` (includes `websockets` library for `uvicorn`)
            *   [ ] Add `httpx` (for making async requests to Rime TTS API)
            *   [ ] Ensure `deepgram-sdk`, `openai` (for Dashscope), `python-dotenv` are present.
        *   [ ] Verify/update `.env` (and `sample.env`):
            *   [ ] Ensure `DASHSCOPE_API_KEY`, `DASHSCOPE_MODEL_NAME`, `DEEPGRAM_API_KEY` are there.
            *   [ ] Add `RIME_API_KEY="your_rime_api_key_here"`
            *   [ ] Add `RIME_API_URL="http://localhost:8000"` (Placeholder, will be Rime's service URL on Render or local Docker)
            *   [ ] Add `RIME_SPEAKER="joy"` (Default speaker)
            *   [ ] Add `RIME_MODEL_ID="mist"` (Default model)
    *   [ ] **1.2: Rime TTS Integration**
        *   [ ] In a new file (e.g., `tts_services.py`) or within `QuickAgent.py`, create/modify a class for Rime TTS (e.g., `RimeTTS`).
            *   [ ] Constructor: Load `RIME_API_KEY`, `RIME_API_URL`, default `RIME_SPEAKER`, `RIME_MODEL_ID` from environment variables.
            *   [ ] Implement `async def synthesize(self, text: str, speaker: str = None, model_id: str = None) -> bytes | None:`
                *   [ ] Use `httpx.AsyncClient` to make a `POST` request to `RIME_API_URL`.
                *   [ ] Headers: `Authorization: Bearer <API_KEY>`, `Content-Type: application/json`, `Accept: audio/mp3`.
                *   [ ] Body: `{"text": text, "speaker": speaker or self.default_speaker, "modelId": model_id or self.default_model_id}`.
                *   [ ] Return the raw audio content (bytes) on success.
                *   [ ] Implement error handling (log errors, return `None` or raise custom exception).
    *   [ ] **1.3: Language Model Processor (`LanguageModelProcessor` in `QuickAgent.py`)**
        *   [ ] Review `async def generate_response(self, text)`: Ensure it's suitable for taking a text input and returning the LLM's response text and processing time.
        *   [ ] The conversation history (`self.messages`) will be managed per session by the FastAPI application logic, so the `LanguageModelProcessor` instance itself will be stateful for that session.
    *   [ ] **1.4: Deepgram STT Service (Streaming)**
        *   [ ] Refactor or create a new class (e.g., `StreamingDeepgramSTT`) to handle streaming STT without direct microphone access.
            *   [ ] Constructor: Initialize `DeepgramClient`.
            *   [ ] Method: `async def process_audio_stream(self, audio_stream_source, event_callback)`:
                *   [ ] `audio_stream_source`: An async iterator or callback that yields audio chunks received from the client's WebSocket.
                *   [ ] `event_callback`: An async function to call when Deepgram events occur (e.g., interim transcript, final transcript, speech started, utterance end, error).
                *   [ ] Set up Deepgram `asynclive` connection (`dg_connection = self.client.listen.asyncwebsocket.v("1")`).
                *   [ ] Define `on_open_async`, `on_message_async`, `on_metadata_async`, `on_speech_started_async`, `on_utterance_end_async`, `on_error_async`, `on_close_async` as nested functions or methods. These will use `event_callback` to send data/events back to the main WebSocket handler in `main.py`.
                *   [ ] Start the connection: `await dg_connection.start(options, addons=addons, **kwargs)`.
                *   [ ] In a loop, receive audio from `audio_stream_source` and send it to `dg_connection.send(chunk)`.
                *   [ ] Handle connection closing.
    *   [ ] **1.5: Session and Conversation Orchestration (in `main.py`)**
        *   [ ] Define a global dictionary for active sessions: `active_sessions = {}`.
            *   Each session entry could store: `{'llm_processor': LanguageModelProcessor(), 'deepgram_stt': StreamingDeepgramSTT(), 'rime_tts': RimeTTS()}` (or TTS can be global if stateless).
        *   [ ] Develop the core logic for handling a turn (will be used within the WebSocket endpoint).
    *   [ ] **1.6: FastAPI Application (`main.py`)**
        *   [ ] Initialize FastAPI app: `app = FastAPI()`.
        *   [ ] Instantiate global services if applicable (e.g., `rime_tts = RimeTTS()` if its config doesn't change per session).
        *   [ ] **WebSocket Endpoint: `/ws/voice_chat/{session_id}`**
            *   [ ] `@app.websocket("/ws/voice_chat/{session_id}")`
            *   [ ] `async def voice_chat_endpoint(websocket: WebSocket, session_id: str):`
            *   [ ] `await websocket.accept()`
            *   [ ] Retrieve or create session state from `active_sessions` using `session_id`. Initialize `LanguageModelProcessor` for new sessions.
            *   [ ] Instantiate `StreamingDeepgramSTT`.
            *   [ ] Define an `stt_event_callback` async function:
                *   This callback will be passed to `StreamingDeepgramSTT.process_audio_stream`.
                *   It will receive events (interim/final transcript, utterance end) from `StreamingDeepgramSTT`.
                *   When an interim transcript arrives: Send `{"type": "interim_transcript", "data": "..."}` over the `websocket` to Next.js.
                *   When a final transcript arrives (e.g., from `on_message` with `is_final` or `on_utterance_end`):
                    *   Send `{"type": "final_transcript", "data": "..."}` to Next.js.
                    *   Call the session's `llm_processor.generate_response(final_transcript)`.
                    *   Send `{"type": "llm_response", "text": llm_text_response}` to Next.js.
                    *   Call `rime_tts.synthesize(llm_text_response)`.
                    *   Convert audio bytes to base64: `base64_audio = base64.b64encode(audio_bytes).decode('utf-8')`.
                    *   Send `{"type": "tts_audio", "format": "mp3", "data": base64_audio}` to Next.js.
            *   [ ] Define an async audio source generator/callback that gets data from `await websocket.receive_bytes()` and yields it to `StreamingDeepgramSTT`.
            *   [ ] Start STT processing: `await deepgram_stt_instance.process_audio_stream(audio_source_from_websocket, stt_event_callback)`.
            *   [ ] Handle WebSocket disconnection (clean up session resources if necessary).
        *   [ ] **HTTP Endpoint: `/health`**
            *   [ ] `@app.get("/health")`
            *   [ ] `async def health_check(): return {"status": "ok"}`
    *   [ ] **1.7: Configuration & Running Locally**
        *   [ ] Create `Procfile` (for Render, and guidance for local running):
            `web: uvicorn main:app --host 0.0.0.0 --port 8000 --ws websockets --reload`
            *(Note: Render provides `$PORT`. For local, you can set it or use a default like 8000).*
        *   [ ] Update `.gitignore`: Add `.venv/`, `__pycache__/`, `*.pyc`, `*.env` (if not already present and specific enough).
        *   [ ] Test running locally: `uvicorn main:app --reload --port 8000 --ws websockets`.
        *   [ ] Perform initial testing of the backend with a WebSocket client tool (e.g., Postman, or a simple Python WebSocket client script) before building the full Next.js frontend.

## Phase 2: Simple Next.js Frontend (for Local Testing)

*   **Goal:** Create a minimal Next.js application to interact with the local FastAPI backend, allowing for microphone input, streaming to the backend, and receiving/playing responses.

*   **Checklist:**
    *   [ ] **2.1: Next.js Project Setup**
        *   [ ] `npx create-next-app@latest nextjs-voice-client` (choose options, e.g., TypeScript, App Router).
        *   [ ] `cd nextjs-voice-client`.
    *   [ ] **2.2: Basic UI (`app/page.tsx` or `pages/index.js`)**
        *   [ ] Add a "Start/Stop Recording" button.
        *   [ ] Add a display area for the conversation transcript (user and agent messages).
        *   [ ] Add a status indicator (e.g., "Idle", "Connecting...", "Listening...", "Processing...", "Agent Speaking...").
    *   [ ] **2.3: Frontend Logic**
        *   [ ] **State Management (React `useState`, `useEffect`):**
            *   [ ] `sessionID`, `isRecording`, `socket (WebSocket instance)`, `transcript`, `conversationLog (array of messages)`, `statusMessage`.
        *   [ ] **Microphone Access & Recording:**
            *   [ ] On component mount or on first "Start", request microphone permission: `navigator.mediaDevices.getUserMedia({ audio: true })`.
            *   [ ] Use `MediaRecorder` API.
        *   [ ] **WebSocket Connection:**
            *   [ ] Function `connectWebSocket(currentSessionID)`:
                *   [ ] `const ws = new WebSocket(\`ws://localhost:8000/ws/voice_chat/\${currentSessionID}\`);` (Port from backend).
                *   [ ] Set up `ws.onopen`, `ws.onmessage`, `ws.onerror`, `ws.onclose` handlers.
            *   [ ] `ws.onmessage`: Parse incoming JSON.
                *   [ ] If `type === "interim_transcript"` or `type === "final_transcript"`: Update transcript display.
                *   [ ] If `type === "llm_response"`: Add agent's text to conversation log.
                *   [ ] If `type === "tts_audio"`:
                    *   [ ] Decode base64 audio data: `const audioBlob = new Blob([Uint8Array.from(atob(message.data), c => c.charCodeAt(0))], { type: 'audio/mp3' });`
                    *   [ ] Create an `Audio` object: `const audio = new Audio(URL.createObjectURL(audioBlob));`
                    *   [ ] `audio.play();`
                    *   [ ] Handle audio playback states (e.g., disable mic input while agent speaks).
        *   [ ] **Button Actions:**
            *   [ ] "Start Recording":
                *   [ ] Generate a new `sessionID` if one doesn't exist (e.g., `uuidv4`).
                *   [ ] Call `connectWebSocket(sessionID)`.
                *   [ ] Once WebSocket is open, start `MediaRecorder`.
                *   [ ] `mediaRecorder.ondataavailable = (event) => { if (socket && socket.readyState === WebSocket.OPEN) socket.send(event.data); }`.
                *   [ ] `mediaRecorder.start(500);` (send data every 500ms, adjust as needed).
            *   [ ] "Stop Recording":
                *   [ ] `mediaRecorder.stop()`.
                *   [ ] Optionally send a "stop_audio_input" message if backend needs it (Deepgram's utterance detection might handle this).
                *   [ ] Keep WebSocket open for receiving LLM/TTS response for the last utterance.
    *   [ ] **2.4: Styling and Refinements**
        *   [ ] Basic CSS for usability.
    *   [ ] **2.5: Local Testing**
        *   [ ] Run backend: `uvicorn main:app --reload --port 8000 --ws websockets`.
        *   [ ] Run frontend: `npm run dev`.
        *   [ ] Thoroughly test the flow: recording, streaming, STT, LLM, TTS playback.

## Phase 3: Deployment to Render.com

*   **Goal:** Deploy the FastAPI backend and the Next.js frontend to Render, configuring them to work together. Deploy Rime TTS if self-hosting.

*   **Checklist:**
    *   [ ] **3.1: Prepare for Deployment**
        *   [ ] Ensure all API keys and sensitive configurations are loaded from environment variables (already planned in Phase 1).
        *   [ ] Finalize `requirements.txt` for the backend.
        *   [ ] Create GitHub repositories for backend and frontend (can be separate or a monorepo).
    *   [ ] **3.2: Deploy Rime TTS (If Self-Hosting on Render)**
        *   [ ] Create a new "Docker" service on Render.
        *   [ ] Point to a repository containing your Rime `docker-compose.yml` (or a Dockerfile that sets it up). Render might require adapting the `docker-compose.yml` into its `render.yaml` format or a direct Docker deployment.
        *   [ ] Configure necessary environment variables for Rime (e.g., license keys, any specific model configs).
        *   [ ] Ensure GPU resources are allocated if Rime requires/benefits from it (check Render's instance types).
        *   [ ] Note down the internal service address (e.g., `http://rime-tts-service.onrender.com:8000`). This will be the `RIME_API_URL` for your FastAPI backend.
    *   [ ] **3.3: Deploy FastAPI Backend to Render**
        *   [ ] Create a new "Web Service" on Render.
        *   [ ] Connect Render to the backend's GitHub repository.
        *   [ ] Configure Settings:
            *   Runtime: Python.
            *   Region: Choose a region close to your users and other services.
            *   Build Command: `pip install -r requirements.txt`.
            *   Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT --ws websockets` (or use the `Procfile`). Render injects `$PORT`.
            *   Environment Variables:
                *   `DASHSCOPE_API_KEY`, `DASHSCOPE_MODEL_NAME`, `DEEPGRAM_API_KEY`.
                *   `RIME_API_KEY`.
                *   `RIME_API_URL`: (Points to your Rime TTS service on Render if self-hosted, or the public Rime API URL).
                *   `RIME_SPEAKER`, `RIME_MODEL_ID`.
                *   `PYTHON_VERSION` (if specific version needed).
        *   [ ] Set up a health check using the `/health` endpoint.
        *   [ ] Deploy and monitor logs. Note the public URL (e.g., `your-backend.onrender.com`).
    *   [ ] **3.4: Deploy Next.js Frontend to Render**
        *   [ ] Create a new "Static Site" (if fully static export) or "Web Service" (for SSR/ISR) on Render.
        *   [ ] Connect Render to the frontend's GitHub repository.
        *   [ ] Configure Settings:
            *   Build Command: `npm install && npm run build` (or `yarn equivalent`).
            *   Publish Directory (for Static Site): `out` (if using `next export`) or `.next` (check Render docs for Next.js).
            *   Start Command (for Web Service): `npm run start`.
            *   Environment Variables (set these as **public** environment variables if read by client-side Next.js code, prefixed with `NEXT_PUBLIC_`):
                *   `NEXT_PUBLIC_BACKEND_WS_URL="wss://your-backend.onrender.com"` (use `wss://` for secure WebSockets).
        *   [ ] Deploy and monitor logs.
    *   [ ] **3.5: Final Testing & DNS**
        *   [ ] Test the fully deployed application.
        *   [ ] If using custom domains, configure DNS records on Render.
        *   [ ] Implement "keep-alive" ping for Render free tier services if needed (e.g., UptimeRobot hitting the `/health` endpoint).

---

This is a comprehensive list. We can adjust and elaborate on each step as we go. 
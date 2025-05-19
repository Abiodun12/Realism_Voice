# Plan: Integrating Deepgram WebSocket Live Streaming with Next.js Frontend

This document outlines the steps to modify `QuickAgent.py` to function as a WebSocket server. This server will receive audio streamed from a Next.js frontend (using the browser's microphone), relay this audio to Deepgram's live streaming WebSocket API, and process the resulting transcripts.

This plan draws inspiration from Deepgram's official documentation and the patterns observed in their [live-streaming-starter-kit](https://github.com/deepgram/live-streaming-starter-kit.git).

## 1. Overall Goal

Transition the existing Python-based voice agent (`QuickAgent.py`) from using a local microphone to accepting real-time audio input from a web browser (Next.js frontend) via a WebSocket connection. The backend will then use Deepgram's WebSocket API for speech-to-text.

## 2. Key Architectural Changes

*   **Python Backend (`QuickAgent.py`)**:
    *   Will act as a WebSocket server, listening for connections from the Next.js frontend.
    *   Will no longer directly access the local microphone via the `deepgram-sdk`'s `Microphone` class.
    *   For each connected frontend client, it will establish a separate WebSocket connection to Deepgram's live streaming API.
*   **Next.js Frontend**:
    *   Will capture audio from the user's browser microphone.
    *   Will establish a WebSocket connection to the Python backend.
    *   Will stream raw audio data to the Python backend.
*   **Deepgram Interaction**:
    *   The Python backend will forward (proxy) the audio received from the frontend to Deepgram's WebSocket API.
    *   Transcripts and other events (like `UtteranceEnd`) from Deepgram will be received by the Python backend and processed by the `ConversationManager`.

## 3. Backend (`QuickAgent.py`) Modifications

### 3.1. Dependencies
*   Ensure `websockets` library is installed (already confirmed in `requirements.txt`).
*   The `deepgram-sdk` will still be used for client configuration (API key) but not for `asynclive` or `Microphone`.

### 3.2. `DeepgramSTT` Class Refactor

*   **`__init__(self, manager_queue, host="localhost", port=8765)`**:
    *   Store `manager_queue`.
    *   Store `host` and `port` for the Python WebSocket server.
    *   Remove initialization of `self.client = DeepgramClient(...)` if it's only used for `asynclive`. The API key can be fetched directly from `os.getenv("DEEPGRAM_API_KEY")` when needed.
    *   Remove `self.dg_connection`, `self.microphone`, `self.accumulated_transcript`, `self.connection_closed_flag`. These will be managed per-client or differently.

*   **`async def start_listening(self)`**:
    *   This method will start the Python WebSocket server using `websockets.serve()`.
    *   Example: `async with websockets.serve(self.client_handler, self.host, self.port): await asyncio.Future() # Run forever`
    *   The first argument to `websockets.serve` will be a new method, e.g., `self.client_handler`.

*   **`async def client_handler(self, client_websocket, path)` (New Method)**:
    *   This method is called for each new client connecting to our Python WebSocket server. `client_websocket` is the connection to the Next.js frontend.
    *   **Deepgram WebSocket Connection**:
        *   Construct the Deepgram WebSocket URL. Example:
            `DG_SOCKET_URL = "wss://api.deepgram.com/v1/listen?model=nova-2&language=en-US&punctuate=true&smart_format=true&encoding=linear16&sample_rate=16000&channels=1&interim_results=true&utterance_end_ms=1500&vad_events=true"`
            (Adjust parameters like `sample_rate`, `encoding`, `channels` based on what the frontend will send. These might need to be configurable or negotiated).
        *   Establish a WebSocket connection to Deepgram:
            `async with websockets.connect(DG_SOCKET_URL, extra_headers={"Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}}") as deepgram_ws:`
    *   **Two-Way Communication Tasks**:
        *   Create two concurrent tasks using `asyncio.gather` or `asyncio.create_task`:
            1.  **`forward_audio_to_deepgram(client_websocket, deepgram_ws)`**:
                *   Receives audio data messages from `client_websocket` (Next.js).
                *   Forwards these messages directly to `deepgram_ws`.
                *   Handles client disconnects gracefully.
            2.  **`receive_transcripts_from_deepgram(deepgram_ws, self.manager_queue)`**:
                *   Receives JSON messages from `deepgram_ws`.
                *   Parses messages (e.g., `Transcript`, `UtteranceEnd`, `SpeechStarted`, `Error`, `Metadata`).
                *   If a final transcript is received (based on `is_final` and `speech_final` or `UtteranceEnd` event), accumulate and put the full transcript onto `self.manager_queue`.
                *   Handle interim results if desired (e.g., for real-time feedback on the frontend, though this part focuses on backend processing first).
                *   Handles Deepgram WebSocket errors or closure.
    *   Ensure both tasks are properly awaited and connections are closed on exit/error.

*   **`async def stop_listening(self)`**:
    *   This method will need to signal the `websockets.serve` loop to shut down. This can be complex as `websockets.serve` runs until its future is cancelled or an error occurs. One common pattern is to have `start_listening` store the server object and then call `server.close()` and `await server.wait_closed()`.

*   **Remove Old Event Handlers**:
    *   The `on_open_async`, `on_message_async`, `on_utterance_end_async`, etc., designed for `deepgram-sdk`'s `asynclive` will be removed. Their logic will be incorporated into `receive_transcripts_from_deepgram`.

### 3.3. `ConversationManager`
*   Likely minimal changes initially. It will continue to get transcripts from `self.transcription_queue` (which is `DeepgramSTT`'s `manager_queue`).
*   The `DeepgramSTT` instance will be created with the appropriate `manager_queue`.

### 3.4. Main Execution Block (`if __name__ == "__main__":`)
*   Ensure `DeepgramSTT.start_listening()` is correctly `await`ed within the `asyncio.run()` context, likely as part of `manager.main_loop()` or alongside it.

## 4. Frontend (Next.js) Requirements

*   **Microphone Access**:
    *   Use `navigator.mediaDevices.getUserMedia({ audio: true })` to get access to the user's microphone.
    *   Handle permissions.
*   **WebSocket Client**:
    *   Establish a WebSocket connection to the Python backend server (e.g., `ws://localhost:8765`).
*   **Audio Streaming**:
    *   Use `MediaRecorder` API or an `AudioWorklet` to process audio from the microphone.
    *   Encode audio into a suitable format (e.g., PCM, 16-bit linear, 16000 Hz, mono). This **must** match what Deepgram expects (as configured in the `DG_SOCKET_URL`).
    *   Send audio data as binary messages (e.g., `ArrayBuffer` or `Blob`) over the WebSocket to the Python backend in chunks.
*   **UI/UX (Optional but Recommended)**:
    *   Display connection status to the WebSocket server.
    *   Optionally, display interim and final transcripts if the Python backend is enhanced to send them back to the frontend.
    *   Provide user controls (start/stop recording).

## 5. Deepgram API Integration Details

*   **WebSocket Endpoint**: `wss://api.deepgram.com/v1/listen`
*   **Authentication**: Via `Authorization: Token YOUR_DEEPGRAM_API_KEY` in `extra_headers` when connecting from Python.
*   **Key Query Parameters for the URL**:
    *   `model`: e.g., `nova-2`, `nova-3`
    *   `language`: e.g., `en-US`
    *   `encoding`: e.g., `linear16` (must match audio from frontend)
    *   `sample_rate`: e.g., `16000` (must match audio from frontend)
    *   `channels`: e.g., `1` (must match audio from frontend)
    *   `interim_results`: `true` for faster feedback.
    *   `utterance_end_ms`: e.g., `1000` or `1500` (milliseconds of silence to detect end of utterance).
    *   `vad_events`: `true` (to get `SpeechStarted`, `UtteranceEnd` events).
    *   `punctuate`: `true`
    *   `smart_format`: `true`
*   **Sending Audio**: Python backend sends binary audio frames received from the client.
*   **Receiving Messages**: Python backend receives JSON messages from Deepgram. Key message types:
    *   `Metadata`: Contains information about the stream.
    *   `SpeechStarted`: Indicates user has started speaking.
    *   `Transcript`: Contains `channel.alternatives[0].transcript`, `is_final`, `speech_final`.
    *   `UtteranceEnd`: Indicates Deepgram detected the end of an utterance.
    *   `Error`: If Deepgram encounters an error.

## 6. Workflow Summary

1.  User grants microphone permission in Next.js frontend.
2.  Next.js captures audio, connects to Python WebSocket server (`ws://localhost:PORT`).
3.  Next.js streams audio chunks to the Python server.
4.  Python server's `client_handler` receives the connection.
5.  Python server establishes a new WebSocket connection to Deepgram (`wss://api.deepgram.com/v1/listen?...`).
6.  Python server forwards audio chunks from Next.js to Deepgram.
7.  Deepgram sends JSON responses (transcripts, events) back to the Python server.
8.  Python server processes these JSONs. Final transcripts are put into `ConversationManager`'s queue.
9.  `ConversationManager` processes the transcript with LLM and TTS as before.
10. (Future Enhancement): Python server could send transcripts or LLM responses back to the Next.js frontend over the initial client WebSocket for display in the UI.

## 7. Important Considerations

*   **Error Handling**: Robustly handle WebSocket connection drops (client-to-backend, backend-to-Deepgram), API errors from Deepgram, and audio processing errors.
*   **Security**:
    *   If the Python WebSocket server is exposed publicly, consider authentication/authorization mechanisms.
    *   For local development, `localhost` is fine.
*   **Scalability**: The current `websockets.serve` model with `asyncio` is efficient but for very high numbers of concurrent users, further architectural considerations (e.g., load balancing) might be needed (though likely out of scope for initial implementation).
*   **Audio Format Consistency**: This is CRITICAL. The audio format (sample rate, encoding, channels, bit depth) sent by the Next.js frontend *must* match the parameters specified in the Deepgram WebSocket URL.
*   **Configuration**: Make host, port, and Deepgram parameters configurable (e.g., via environment variables or a config file).
*   **Logging**: Enhance logging throughout the new WebSocket handling paths for easier debugging.
*   **Graceful Shutdown**: Ensure all WebSocket connections and tasks are properly closed when the application shuts down.

This plan provides a roadmap for the integration. Implementation will require careful handling of asynchronous operations and WebSocket protocols. 
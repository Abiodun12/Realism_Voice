import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI
import signal
import websockets
import json
import aiohttp

from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

# Make sure to load environment variables with explicit path
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Check for required environment variables
if not os.getenv("DASHSCOPE_API_KEY"):
    print("Error: DASHSCOPE_API_KEY environment variable is not set.")
    print("Please create a .env file in the project root with your API keys:")
    print("DASHSCOPE_API_KEY=your_api_key_here")
    print("DEEPGRAM_API_KEY=your_deepgram_api_key_here")
    sys.exit(1)

class LanguageModelProcessor:
    def __init__(self):
        # Set up OpenAI client with DashScope compatible mode
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        
        self.model_name = os.getenv("DASHSCOPE_MODEL_NAME", "qwen-plus")
        
        # Load the system prompt from a file
        try:
            with open('system_prompt.txt', 'r') as file:
                self.system_prompt = file.read().strip()
        except FileNotFoundError:
            self.system_prompt = "You are Remi—AKA 'AB Uncle'—a fun, quick‑witted African‑American uncle who speaks in AAVE. Greet with 'Hi, how you doin', dawg?' then respond in at most 120 characters, always finish your sentence completely, use max 3 short sentences, one joke max, and add a wink 😉 if it fits."
            print("Warning: system_prompt.txt not found. Using default AB Uncle system prompt.")
        
        # Initialize conversation history with system prompt and a few-shot example
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "Tell me a joke about pizza."},
            {"role": "assistant", "content": "Hi, how you doin', dawg? Why'd the pizza apply for a job? Cause it was on a roll! 😉"}
        ]

    def process(self, text):
        # Add user message to history
        self.messages.append({"role": "user", "content": text})
        
        start_time = time.time()
        
        try:
            # Call OpenAI-compatible API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                max_tokens=60  # Increased from 30 to 60 (~120 characters of output)
            )
            
            end_time = time.time()
            elapsed_time = int((end_time - start_time) * 1000)
            
            # Extract the response text
            response_text = completion.choices[0].message.content
            
            # Add assistant response to history
            self.messages.append({"role": "assistant", "content": response_text})
            
            print(f"LLM ({elapsed_time}ms): {response_text}")
            return response_text
                
        except Exception as e:
            end_time = time.time()
            elapsed_time = int((end_time - start_time) * 1000)
            error_msg = f"Error from LLM API: {str(e)}"
            print(error_msg)
            return "I'm sorry, I encountered an issue connecting to my language model. Please try again later."

    # Add async method for compatibility with the conversation manager
    async def generate_response(self, text):
        start_time = time.time()
        print(f"Real LLM processing: {text}")
        
        # Add user message to history
        self.messages.append({"role": "user", "content": text})
        
        try:
            # Call OpenAI-compatible API using to_thread to make it async
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=self.messages,
                max_tokens=60,  # Increased from 30 to 60 (~120 characters of output)
            )
            
            # Extract the response text
            response_text = completion.choices[0].message.content
            
            # Add assistant response to history
            self.messages.append({"role": "assistant", "content": response_text})
            
            llm_time = int((time.time() - start_time) * 1000)
            return response_text, llm_time
                
        except Exception as e:
            error_msg = f"Error from LLM API: {str(e)}"
            print(error_msg)
            llm_time = int((time.time() - start_time) * 1000)
            return "I'm sorry, I encountered an issue connecting to my language model. Please try again later.", llm_time

class TextToSpeech:
    # Set your Deepgram API Key and desired voice model
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-asteria-en"  # Default model name

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        print(f"Converting to speech: {text}")
        
        if not self.DG_API_KEY:
            print("Error: DEEPGRAM_API_KEY not set")
            return

        if not self.is_installed("ffplay"):
            print("Error: ffplay not found, necessary to stream audio.")
            return

        # Fixed URL format based on Deepgram documentation
        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        # Correct payload format - only text is needed in the JSON body
        payload = {
            "text": text
        }

        print("Sending TTS request to Deepgram...")
        
        try:
            player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
            player_process = subprocess.Popen(
                player_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            start_time = time.time()  # Record the time before sending the request
            first_byte_time = None  # Initialize a variable to store the time when the first byte is received

            # Print request details for debugging
            print(f"URL: {DEEPGRAM_URL}")
            print(f"Payload: {payload}")
            
            with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
                if r.status_code != 200:
                    print(f"Error from Deepgram API: {r.status_code} - {r.text}")
                    return
                    
                print(f"Response received with status code: {r.status_code}")
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        if first_byte_time is None:  # Check if this is the first chunk received
                            first_byte_time = time.time()  # Record the time when the first byte is received
                            ttfb = int((first_byte_time - start_time)*1000)  # Calculate the time to first byte
                            print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                        player_process.stdin.write(chunk)
                        player_process.stdin.flush()

            if player_process.stdin:
                player_process.stdin.close()
            player_process.wait()
            print("Audio playback complete")
            
        except Exception as e:
            print(f"Error during TTS processing: {str(e)}")
            return

class TranscriptCollector:
    def __init__(self):
        self.reset()
        self.last_interim_transcript = ""

    def reset(self):
        self.transcript_parts = []
        self.last_interim_transcript = ""

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback, interim_callback=None):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            print("Error: DEEPGRAM_API_KEY environment variable is not set.")
            return
            
        deepgram: DeepgramClient = DeepgramClient(deepgram_api_key, config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if not sentence.strip():
                return
                
            # Handle interim results for more responsive interaction
            if not result.is_final:
                if interim_callback and sentence.strip():
                    interim_transcript = sentence.strip()
                    if interim_transcript != transcript_collector.last_interim_transcript:
                        transcript_collector.last_interim_transcript = interim_transcript
                        await interim_callback(interim_transcript)
                return
                
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before processing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-3",  # Upgrade to latest Deepgram model 
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=150,  # Reduce further from 200ms to 150ms for faster responses
            smart_format=True,
            interim_results=True  # Enable interim results for more responsive feel
        )

        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()
                
        await transcription_complete.wait()

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class DeepgramSTT:
    def __init__(self, manager_queue, manager, host="localhost", port=8765):
        self.manager_queue = manager_queue
        self.manager = manager  # Store the ConversationManager instance
        self.host = host
        self.port = port
        self.server = None  # To store the server object for graceful shutdown
        self.accumulated_transcript_per_client = {} # To store transcript per client
        self.processing_enabled = True  # Flag to control processing

    async def client_handler(self, client_websocket, path=""):
        client_id = f"{client_websocket.remote_address[0]}:{client_websocket.remote_address[1]}"
        print(f"New client connection from {client_id} on path {path}")
        self.accumulated_transcript_per_client[client_id] = ""

        DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
        if not DG_API_KEY:
            print(f"Deepgram API Key not found for client {client_id}. Closing connection.")
            try:
                await client_websocket.close(reason="Deepgram API Key not configured on server.")
            except Exception as e:
                print(f"Error closing client connection: {e}")
            return

        # Construct Deepgram WebSocket URL with the API key in the URL
        # This avoids the need for extra_headers or headers parameter
        DG_SOCKET_URL = (
            f"wss://api.deepgram.com/v1/listen?"
            f"model=nova-3&language=en-US&punctuate=true&smart_format=true&"
            f"encoding=linear16&sample_rate=16000&channels=1&"
            f"interim_results=true&utterance_end_ms=1500&vad_events=true&"
            f"endpointing=1000&filler_words=false&diarize=false"
        )
        # For a keepalive message to Deepgram every 5 seconds
        KEEPALIVE_MESSAGE = json.dumps({"type": "KeepAlive"})

        try:
            # Create a custom HTTPS request that includes the Authorization header
            
            # Use aiohttp to connect to Deepgram WebSocket
            session = aiohttp.ClientSession()
            headers = {"Authorization": f"Token {DG_API_KEY}"}
            async with session.ws_connect(DG_SOCKET_URL, headers=headers) as deepgram_ws:
                print(f"Successfully connected to Deepgram for client {client_id}")

                async def forward_audio_to_deepgram():
                    chunk_counter = 0
                    audio_logging_interval = 100
                    try:
                        while True:
                            try:
                                audio_chunk = await client_websocket.recv()
                                
                                # Check if the manager is speaking before processing/forwarding
                                if self.manager and self.manager.is_speaking:
                                    # If manager is speaking, discard the audio chunk to prevent echo
                                    # Optionally log that audio is being discarded
                                    # print(f"🎤 Discarding audio chunk from {client_id} (manager speaking)")
                                    continue # Skip to the next iteration, effectively discarding the chunk

                                # Basic validation and diagnostic logging
                                chunk_counter += 1
                                if chunk_counter % audio_logging_interval == 0:
                                    # Log audio statistics every audio_logging_interval chunks
                                    chunk_size_bytes = len(audio_chunk)
                                    
                                    # Check if it's likely to be a binary audio chunk (assuming Int16 PCM)
                                    is_valid_size = chunk_size_bytes % 2 == 0  # Int16 means even number of bytes
                                    
                                    if chunk_size_bytes > 0:
                                        # Simple check for non-zero audio
                                        # Convert first few bytes to get a sample
                                        if chunk_size_bytes >= 32:  # At least some data to sample
                                            import struct
                                            samples = []
                                            for i in range(0, min(32, chunk_size_bytes - 1), 2):
                                                sample = struct.unpack("<h", audio_chunk[i:i+2])[0]
                                                samples.append(abs(sample))
                                            
                                            # Calculate max level as percentage of full scale
                                            if samples:
                                                max_level = max(samples) / 32768.0 * 100
                                                print(f"Audio chunk #{chunk_counter}: {chunk_size_bytes} bytes, max level: {max_level:.1f}% of full scale")
                                            else:
                                                print(f"Audio chunk #{chunk_counter}: {chunk_size_bytes} bytes (no samples could be analyzed)")
                                        else:
                                            print(f"Audio chunk #{chunk_counter}: {chunk_size_bytes} bytes (too small to analyze)")
                                
                                # Forward to Deepgram
                                await deepgram_ws.send_bytes(audio_chunk)
                            except Exception as e:
                                print(f"Error in audio forwarding: {e}")
                                break
                    except Exception as e:
                        print(f"Client {client_id} error (audio forward): {e}")
                    finally:
                        # Signal Deepgram that audio stream is ending
                        try:
                            await deepgram_ws.send_json({"type": "CloseStream"})
                        except Exception as e:
                            print(f"Error sending CloseStream to Deepgram for {client_id}: {e}")
                        print(f"Audio forwarding task for {client_id} finished.")

                async def receive_transcripts_from_deepgram():
                    last_utterance_end_time = time.time()
                    utterance_in_progress = False 
                    try:
                        async for msg in deepgram_ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                message = json.loads(msg.data)
                                msg_type = message.get("type")

                                if msg_type == "Metadata":
                                    # Only log metadata on initial connection
                                    if "created" in message:
                                        print(f"Deepgram connected for client {client_id}")
                                elif msg_type == "SpeechStarted":
                                    print(f"👂 Speech detected from client {client_id}")
                                    utterance_in_progress = True
                                elif msg_type == "UtteranceEnd":
                                    print(f"🔚 Utterance ended for client {client_id}")
                                    last_utterance_end_time = time.time()
                                    utterance_in_progress = False
                                    
                                    accumulated_at_utterance_end = self.accumulated_transcript_per_client.get(client_id, "").strip()
                                    if accumulated_at_utterance_end:
                                        print(f"\n🗣️ Human (from UtteranceEnd): {accumulated_at_utterance_end}")
                                        if self.processing_enabled:
                                            await self.manager_queue.put(accumulated_at_utterance_end)
                                        else:
                                            print("⚠️ Processing disabled, not sending transcript to queue (UtteranceEnd)")
                                        self.accumulated_transcript_per_client[client_id] = "" # Clear after processing
                                    # No longer just a debug log; UtteranceEnd with content now triggers processing.
                                elif msg_type == "Results":
                                    transcript = message.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                                    is_final = message.get("is_final", False)
                                    speech_final = message.get("speech_final", False)
                                    
                                    if is_final and transcript.strip():
                                        current_transcript = self.accumulated_transcript_per_client.get(client_id, "")
                                        # Smart concatenation to avoid double spaces
                                        if current_transcript and not current_transcript.endswith(' '):
                                            self.accumulated_transcript_per_client[client_id] = current_transcript + " " + transcript
                                        else:
                                            self.accumulated_transcript_per_client[client_id] = (current_transcript + transcript).strip()
                                    
                                    # Only process and clear the accumulator when speech_final is true
                                    if speech_final:
                                        final_transcript = self.accumulated_transcript_per_client.get(client_id, "").strip()
                                        if final_transcript: # Check if there's anything new to process
                                            print(f"\n🗣️ Human (SpeechFinal): {final_transcript}")
                                            if self.processing_enabled:
                                                await self.manager_queue.put(final_transcript)
                                            else:
                                                print("⚠️ Processing disabled, not sending to queue (SpeechFinal)")
                                        
                                        # Clear the accumulator ONLY after a speech_final has been processed (or if it was empty)
                                        self.accumulated_transcript_per_client[client_id] = ""
                                        utterance_in_progress = False # Also reset this here
                                    elif transcript and not is_final: 
                                        # Show interim results without cluttering
                                        if len(transcript) > 5:  # Only show if we have meaningful content
                                            interim_display = (self.accumulated_transcript_per_client.get(client_id, "") + " " + transcript).strip()
                                            print(f"🎤 Hearing: {interim_display}...", end='\r', flush=True)
                                elif msg_type == "Error":
                                    print(f"❌ Deepgram Error for {client_id}: {message.get('description', message)}")
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                print(f"❌ Deepgram WebSocket error for {client_id}: {msg}")
                                break
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                print(f"Deepgram WebSocket closed for {client_id}")
                                break
                    except Exception as e:
                        print(f"Error receiving transcripts from Deepgram for {client_id}: {e}")
                    finally:
                        print(f"Transcript receiving task for {client_id} finished.")
                        # Ensure client connection is closed if Deepgram connection ends
                        try:
                            await client_websocket.close(reason="Deepgram connection ended")
                        except Exception as e:
                            print(f"Error closing client_websocket: {e}")
                
                async def send_keepalives_to_deepgram():
                    try:
                        while True:
                            try:
                                await deepgram_ws.send_json(json.loads(KEEPALIVE_MESSAGE))
                                await asyncio.sleep(5) # Send keepalive every 5 seconds
                            except Exception as e:
                                print(f"Error sending keepalive: {e}")
                                break
                    except Exception as e:
                        print(f"Error in Deepgram keepalive task for {client_id}: {e}")
                    finally:
                        print(f"Deepgram keepalive task for {client_id} finished.")

                # Run the three tasks concurrently
                try:
                    await asyncio.gather(
                        forward_audio_to_deepgram(),
                        receive_transcripts_from_deepgram(),
                        send_keepalives_to_deepgram()
                    )
                except Exception as e:
                    print(f"Error in gathered tasks for {client_id}: {e}")
                
            # Session automatically closed by the context manager

        except aiohttp.ClientError as e:
            print(f"AIOHTTP client error connecting to Deepgram for {client_id}: {e}")
        except Exception as e:
            print(f"Overall error in client_handler for {client_id}: {e}")
        finally:
            print(f"Client {client_id} handler finished.")
            if client_id in self.accumulated_transcript_per_client:
                del self.accumulated_transcript_per_client[client_id]
            try:
                await client_websocket.close()
            except Exception as e:
                print(f"Final error during client_websocket close for {client_id}: {e}")
            
            # Ensure aiohttp session is closed if still exists
            if 'session' in locals() and not session.closed:
                await session.close()

    async def start_listening(self):
        if not os.getenv("DEEPGRAM_API_KEY"):
            print("Error: DEEPGRAM_API_KEY environment variable not set. STT will not function.")
            # Optionally, prevent server from starting or run in a degraded mode
            # For now, we let it start but client_handler will fail connections.

        # Create a wrapper function that handles the websockets library's parameter passing
        async def handler_wrapper(websocket):
            await self.client_handler(websocket)

        print(f"Starting WebSocket server on {self.host}:{self.port}...")
        try:
            self.server = await websockets.serve(handler_wrapper, self.host, self.port)
            print(f"WebSocket server listening on {self.host}:{self.port}")
            await self.server.wait_closed() # Keep server running until it's closed
        except OSError as e:
            print(f"Failed to start WebSocket server on {self.host}:{self.port}: {e}")
            print("This might be due to the port already being in use or insufficient permissions.")
        except Exception as e:
            print(f"An unexpected error occurred while starting or running the WebSocket server: {e}")
        finally:
            print("WebSocket server has shut down.")

    async def stop_listening(self):
        if self.server:
            print("Attempting to stop WebSocket server...")
            self.server.close()
            try:
                # Give it a moment to close gracefully
                await asyncio.wait_for(self.server.wait_closed(), timeout=5.0)
                print("WebSocket server stopped.")
            except asyncio.TimeoutError:
                print("WebSocket server did not stop gracefully within timeout.")
            except Exception as e:
                print(f"Error stopping WebSocket server: {e}")
        self.server = None
        # No specific client connections to close here, client_handler should manage its own resources.
        # Clearing accumulated transcripts on stop might be good if server restarts are expected
        self.accumulated_transcript_per_client.clear()
        print("Deepgram STT (WebSocket mode) stopped.")

class ConversationManager:
    def __init__(self, llm, tts, ws_host="localhost", ws_port=8765):
        self.llm = llm
        self.tts = tts
        self.transcription_queue = asyncio.Queue() 
        self.stt = DeepgramSTT(self.transcription_queue, self, ws_host, ws_port) # Pass self (manager instance)
        self.is_running = True
        self.current_llm_task = None
        self.user_spoke_again = asyncio.Event()
        self.is_speaking = False  # Flag to track if system is speaking
        self.speaking_lock = asyncio.Lock()  # Lock to prevent overlapping speech

    async def process_llm_and_speak(self, text_input):
        print(f"⚙️ Processing: '{text_input}'")
        # Reset the event before starting LLM/TTS
        self.user_spoke_again.clear()
        
        try:
            llm_response_text, llm_time = await self.llm.generate_response(text_input)
            print(f"\n🤖 Assistant ({llm_time}ms): {llm_response_text}")

            if self.user_spoke_again.is_set():
                print("User spoke again during LLM response generation. Not speaking this response.")
                return

            if llm_response_text:
                # Use lock to prevent overlapping speech
                async with self.speaking_lock:
                    # Disable speech processing before speaking
                    self.is_speaking = True
                    self.stt.processing_enabled = False
                    print("🔇 Speech processing disabled during TTS output")
                    
                    try:
                        # Speak the response
                        await self.tts.speak(llm_response_text, self.user_spoke_again)
                        
                        # Add a delay after speech before resuming speech processing
                        # This prevents any audio echo/feedback from being processed
                        if not self.user_spoke_again.is_set():
                            try:
                                # Delay removed as per user request
                                pass # await asyncio.sleep(2.5)  # 2.5 second delay to avoid echo detection - REMOVED
                            except asyncio.CancelledError:
                                # Task was cancelled during sleep - that's fine
                                return
                    except Exception as e:
                        print(f"❌ Error during TTS: {e}")
                    finally:
                        # Even if there's an error, reset these flags
                        self.is_speaking = False
                        self.stt.processing_enabled = True
                        print("🔊 Speech processing enabled")
            else:
                print("LLM returned no response.")
            
            if not self.user_spoke_again.is_set() and self.is_running:
                print("\n🎤 Ready! Click record and speak when you want to continue the conversation.\n")
        except asyncio.CancelledError:
            print("LLM/TTS processing was cancelled")
            raise
        except Exception as e:
            print(f"❌ Error in process_llm_and_speak: {e}")
            # Still reset flags on error
            self.is_speaking = False
            self.stt.processing_enabled = True

    async def main_loop(self):
        # Instead of self.stt.start_listening() directly here, 
        # we will create a task for it to run the server concurrently.
        stt_server_task = asyncio.create_task(self.stt.start_listening())
        print("ConversationManager: STT server task created")

        try:
            print("ConversationManager: Entering main loop")
            print("\n🎤 Waiting for you to click record and speak...\n")
            last_queue_log_time = 0
            queue_log_interval = 60  # Only log waiting message every 60 seconds (reduced frequency)
            
            while self.is_running:
                try:
                    # Use a short timeout to allow for regular checks
                    current_time = time.time()
                    if current_time - last_queue_log_time >= queue_log_interval:
                        print("🎤 Waiting for you to click record and speak...")
                        last_queue_log_time = current_time
                        
                    user_input = await asyncio.wait_for(self.transcription_queue.get(), timeout=0.1)
                    queue_size = self.transcription_queue.qsize()
                    # Reset timer when we get input
                    last_queue_log_time = current_time
                    
                    print(f"ConversationManager: Received user input: '{user_input}'")
                    print(f"DEBUG: Queue size after get: {queue_size}")
                    
                    # Handle user input
                    if user_input:
                        print(f"ConversationManager: Processing user input: '{user_input}'")
                        # Create a task for LLM processing to not block the main loop
                        print(f"ConversationManager: Creating LLM task for: '{user_input}'")
                        # Process LLM and speak in the current task
                        await self.process_llm_and_speak(user_input)
                        
                except asyncio.TimeoutError:
                    # No input available within timeout, continue loop
                    pass
                except Exception as e:
                    print(f"Error in ConversationManager.main_loop: {e}")
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(1)  # Avoid tight error loop
                    
            print("ConversationManager: Main loop exited")
                
        except asyncio.CancelledError:
            print("Main loop: Attempting to stop STT WebSocket server task...")
            stt_server_task.cancel()
            try:
                await stt_server_task
            except asyncio.CancelledError:
                print("Main loop: STT WebSocket server task has been cancelled.")
            except Exception as e:
                print(f"Main loop: Error during STT WebSocket server task cancellation: {e}")
            finally:
                print("Main loop: STT WebSocket server task has completed.")
            raise

class DeepgramTTS:
    def __init__(self, api_key):
        self.api_key = api_key
        # self.client = DeepgramClient(api_key) # Not used if using requests directly
        self.player_process = None
        self.play_command = ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", "-"]
        # Check for ffplay availability or use afplay/aplay
        # For macOS: self.play_command = ["afplay", "-"]
        # For Linux (aplay): self.play_command = ["aplay", "-f", "S16_LE", "-r", "24000", "-"] # Example, adjust format/rate

    async def speak(self, text, interrupt_event=None):
        if not text:
            print("TTS: No text to speak.")
            return

        DEEPGRAM_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"text": text}

        print(f"🔊 Speaking: {text}")
        start_time = time.time()

        try:
            # Using asyncio.to_thread for the blocking requests call
            response = await asyncio.to_thread(requests.post, DEEPGRAM_URL, headers=headers, json=payload, stream=True)
            
            response.raise_for_status()

            if self.player_process and self.player_process.poll() is None:
                print("⚠️ Previous player process still running. Terminating.")
                await self.stop_playback()

            self.player_process = subprocess.Popen(self.play_command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            audio_received_time = None
            chunk_count = 0
            
            # Wrap the audio processing in a try block to catch pipe errors
            try:
                for chunk in response.iter_content(chunk_size=4096): # Increased chunk_size
                    if interrupt_event and interrupt_event.is_set():
                        print("⏹️ TTS: Interrupt received, stopping playback.")
                        response.close() # Close the response stream
                        await self.stop_playback()
                        return

                    if chunk:
                        if audio_received_time is None: # First chunk of audio
                            audio_received_time = time.time()
                            actual_ttfb = int((audio_received_time - start_time) * 1000)
                            print(f"⏱️ TTS Time to First Audio Byte: {actual_ttfb}ms")
                        
                        try:
                            if self.player_process and self.player_process.stdin:
                                self.player_process.stdin.write(chunk)
                            else: # Player closed prematurely
                                print("⚠️ TTS: Player process stdin is not available. Stopping.")
                                response.close()
                                break 
                        except BrokenPipeError:
                            # This is common and can be handled silently
                            response.close()
                            break 
                        except Exception as e: # Catch other potential stdin write errors
                            print(f"❌ TTS: Error writing to player stdin: {e}")
                            response.close()
                            break
                    chunk_count +=1
            except BrokenPipeError:
                # Handle silently - this is expected during interruptions
                response.close()
            except Exception as e:
                print(f"❌ TTS: Error during audio chunk processing: {e}")
                response.close()
            
            if chunk_count == 0 and audio_received_time is None: # No audio data received
                print(f"⚠️ TTS: No audio data received from Deepgram for: {text}")

            if self.player_process and self.player_process.stdin:
                try:
                    self.player_process.stdin.close()
                except BrokenPipeError:
                    # Expected during cleanup
                    pass
                except Exception as e:
                    print(f"❌ TTS: Error closing player stdin: {e}")
            
            if self.player_process:
                # Wait for playback to complete in a way that doesn't block
                rc = await asyncio.to_thread(self.player_process.wait)
                if rc != 0:
                    print(f"⚠️ TTS: Player process exited with code {rc}")
                self.player_process = None
                
                print("✅ Audio playback complete")

        except requests.exceptions.RequestException as e:
            print(f"❌ TTS Request failed: {e}")
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            await self.stop_playback() 
        finally:
            if self.player_process and self.player_process.poll() is None:
                try:
                    await self.stop_playback()
                except Exception as e:
                    print(f"❌ TTS: Error during cleanup in finally block: {e}")

    async def stop_playback(self):
        if self.player_process and self.player_process.poll() is None: 
            print("Stopping TTS playback...")
            try:
                if self.player_process.stdin:
                    self.player_process.stdin.close()
            except BrokenPipeError:
                print("Pipe was already broken during stop - this is expected during interruption")
            except Exception as e:
                print(f"TTS stop: Error closing stdin: {e}")

            try:
                self.player_process.terminate() 
                try:
                    await asyncio.to_thread(self.player_process.wait, timeout=1.0)
                except subprocess.TimeoutExpired:
                    print("Player did not terminate gracefully, killing.")
                    self.player_process.kill() 
                    try:
                        await asyncio.to_thread(self.player_process.wait, timeout=1.0)
                    except subprocess.TimeoutExpired:
                        print("Player did not die even after kill. Giving up.")
                    except Exception as e:
                         print(f"TTS stop wait after kill: Error during player wait: {e}")
                except Exception as e:
                     print(f"TTS stop: Error during player termination: {e}")
            except Exception as e:
                print(f"TTS stop: Error during termination process: {e}")
            
            self.player_process = None
            print("TTS playback stopped.")
        elif self.player_process and self.player_process.poll() is not None:
            self.player_process = None 

async def main_async(text_only=False):
    # Initialize your LLM and TTS with real implementations
    llm = LanguageModelProcessor()
    
    if text_only:
        tts = TextToSpeech()
    else:
        print("Initializing Rime TTS...")
        from realism_voice.io.rime_tts_async import RimeTTS
        try:
            tts = RimeTTS()
            print("Successfully initialized Rime TTS")
        except Exception as e:
            print(f"Failed to initialize Rime TTS: {e}")
            print("Falling back to text-only mode.")
            tts = TextToSpeech()

    manager = ConversationManager(llm, tts, ws_host=args.ws_host, ws_port=args.ws_port)
    
    print(f"\n🎧 System ready! Rime will speak after you click record and say something. WebSocket server listening on {args.ws_host}:{args.ws_port}\n")
    
    # Set up proper signal handling for cleaner shutdown
    loop = asyncio.get_running_loop()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(manager)))
    except NotImplementedError:
        # Windows doesn't support signals properly
        pass

    try:
        # Start the main conversation loop
        await manager.main_loop()
    except Exception as e:
        print(f"Critical exception in main_async: {e}")
        import traceback
        traceback.print_exc()

async def shutdown(manager):
    print("\nShutdown signal received, cleaning up...")
    manager.is_running = False
    # The manager's main loop will handle the rest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Voice Assistant")
    parser.add_argument("--text_only", action="store_true", help="Run in text-only mode (no TTS).")
    parser.add_argument("--ws_host", type=str, default="0.0.0.0", help="WebSocket server host (default: 0.0.0.0, accessible on LAN)")
    parser.add_argument("--ws_port", type=int, default=8765, help="WebSocket server port (default: 8765)")
    args = parser.parse_args() 

    try:
        asyncio.run(main_async(text_only=args.text_only))
    except KeyboardInterrupt:
        # This handler is now redundant with our signal handling, but kept for safety
        print("Application terminated by user.")
    except RuntimeError as e:
        # Most common runtime errors during shutdown are related to event loops
        # We'll handle them gracefully
        if "Event loop is closed" in str(e):
            print("Application successfully shut down.")
        else:
            print(f"Runtime error: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        # This top-level exception handler is good for catching unexpected issues
        print(f"An unhandled critical exception occurred: {e}")
        import traceback
        traceback.print_exc() 
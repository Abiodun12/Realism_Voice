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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import json
import starlette.websockets

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

# Create FastAPI app
app = FastAPI(title="Realism Voice API", description="API for voice interactions using Deepgram")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",          # local Next.js dev
        "https://your-site.vercel.app",   # replace with your real domain
        "*",                              # Allow any origin for testing (you can remove this in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a health check endpoint for Render to prevent spinning down
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}

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

        # Using the correct pattern for SDK 4.x
        dg_connection = deepgram.listen.websocket.v("1")
        print("Listening...")

        def on_message(client, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if not sentence.strip():
                return
                
            # Handle interim results for more responsive interaction
            if not result.is_final:
                if interim_callback and sentence.strip():
                    interim_transcript = sentence.strip()
                    if interim_transcript != transcript_collector.last_interim_transcript:
                        transcript_collector.last_interim_transcript = interim_transcript
                        # Using a thread to handle the async callback
                        threading.Thread(target=lambda: asyncio.run(interim_callback(interim_transcript))).start()
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

        # Start is not awaitable in SDK 4.x
        dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()
                
        await transcription_complete.wait()

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished - not awaitable in SDK 4.x
        dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class DeepgramSTT:
    def __init__(self, manager_queue):
        self.manager_queue = manager_queue
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.deepgram_api_key:
            print("ERROR: DEEPGRAM_API_KEY environment variable is not set")
            raise ValueError("Missing Deepgram API key")
            
        # Initialize Deepgram client with proper options
        config = DeepgramClientOptions(options={"keepalive": "true"})
        self.client = DeepgramClient(self.deepgram_api_key, config)
        self.dg_connection = None
        self.microphone = None
        self.is_speaking = False
        self.accumulated_transcript = ""
        self.processing_enabled = True  # Flag to enable/disable transcript processing
        self.microphone_enabled = False  # Flag to control whether microphone should be active
        # Store the event loop that this object was created on
        self.loop = asyncio.get_event_loop()
        
        # Check if we're running in a server environment (like Render.com)
        self.is_server_env = os.getenv("PORT") is not None
        if self.is_server_env:
            print("Running in server environment - microphone access will be disabled")

    async def start_listening(self, enable_microphone=False):
        # Ensure we're on the same event loop
        if asyncio.get_event_loop() != self.loop:
            print("Warning: Event loop mismatch. This might cause issues.")
        
        # Set microphone flag
        self.microphone_enabled = enable_microphone
        
        # Create the Deepgram connection using the documented approach for SDK 4.1.0
        try:
            print("Creating Deepgram live connection")
            
            # Reinitialize client with keepalive option
            config = DeepgramClientOptions(options={"keepalive": "true"})
            self.client = DeepgramClient(self.deepgram_api_key, config)
            
            # Prepare connection options
            options = LiveOptions(
                model="nova-3",
                language="en-US", 
                smart_format=True,
                encoding="linear16",
                channels=1, 
                sample_rate=16000,
                interim_results=True,
                utterance_end_ms=1000,  # Must be at least 1000ms
                vad_events=True,
                endpointing=800
            )
            
            # Create API connection using the pattern for SDK 4.x
            try:
                # For SDK 4.x, the correct pattern is to use websocket, not live
                connection = self.client.listen.websocket.v("1")
                self.dg_connection = connection
                
                # Set up a keepalive task to prevent the Deepgram connection from timing out
                # We'll run this in a separate thread to not block the main event loop
                def start_keepalive():
                    print("Starting Deepgram keepalive thread")
                    while True:
                        try:
                            time.sleep(5)  # Send keepalive every 5 seconds
                            # Only send if connection exists and we're not in the process of closing it
                            if self.dg_connection and hasattr(self.dg_connection, 'send_message'):
                                try:
                                    # Try to send a keepalive message
                                    self.dg_connection.send_message({"type": "KeepAlive"})
                                    print("Sent Deepgram keepalive ping")
                                except:
                                    # If it fails, the connection might be closed or in an invalid state
                                    # This is expected during shutdown so we'll just pass
                                    pass
                        except:
                            # If thread is interrupted or connection is closed, break the loop
                            break
                
                # Start the keepalive thread if we're in a server environment 
                # where the connection needs to stay open for long periods
                if self.is_server_env:
                    keepalive_thread = threading.Thread(target=start_keepalive, daemon=True)
                    keepalive_thread.start()
                
                # Define event handlers - callback style for SDK 4.x
                def on_open(client, event, **kwargs):
                    print(f"Deepgram connection opened successfully: {event}")
                
                def on_message(client, result, **kwargs):
                    # Store reference to self for callback processing
                    outer_self = self
                    
                    # Only process if speech processing is enabled
                    if not outer_self.processing_enabled:
                        return
                        
                    try:
                        if not hasattr(result, 'channel') or not hasattr(result.channel, 'alternatives') or len(result.channel.alternatives) == 0:
                            print("Warning: Received message with no alternatives")
                            return
                            
                        sentence = result.channel.alternatives[0].transcript
                        
                        if result.is_final and result.speech_final:
                            if sentence.strip():
                                outer_self.accumulated_transcript = sentence.strip()
                                print(f"Final transcript: {outer_self.accumulated_transcript}")
                                # Instead of using create_task, add to the queue directly in non-async context
                                outer_self.manager_queue.put_nowait({
                                    "text": outer_self.accumulated_transcript.strip(),
                                    "is_final": True,
                                    "timestamp": time.time()
                                })
                                outer_self.accumulated_transcript = ""  # Reset transcript
                        elif not result.is_final and sentence.strip(): # Interim result
                            current_transcript_interim = sentence.strip()
                            print(f"Hearing: {current_transcript_interim}...", end='\r', flush=True)
                            outer_self.accumulated_transcript = current_transcript_interim
                    except Exception as e:
                        print(f"Error processing transcription message: {e}")
                
                def on_utterance_end(client, utterance_end, **kwargs):
                    # Store reference to self for callback processing
                    outer_self = self
                    
                    # Only process if speech processing is enabled
                    if not outer_self.processing_enabled:
                        return
                        
                    print("\nUser likely finished speaking (UtteranceEnd).")
                    outer_self.is_speaking = False # User has stopped.
                    if len(outer_self.accumulated_transcript.strip()) > 0:
                        try:
                            # Instead of using create_task, add to the queue directly in non-async context
                            outer_self.manager_queue.put_nowait({
                                "text": outer_self.accumulated_transcript.strip(),
                                "is_final": True,
                                "timestamp": time.time()
                            })
                            outer_self.accumulated_transcript = ""  # Reset transcript
                        except Exception as e:
                            print(f"Error sending transcript to manager: {e}")
                    else:
                        print("UtteranceEnd received, but no transcript accumulated to send.")
                
                def on_speech_started(client, speech_started, **kwargs):
                    # Store reference to self for callback processing
                    outer_self = self
                    
                    # Only process if speech processing is enabled
                    if not outer_self.processing_enabled:
                        return
                        
                    print("User started speaking.")
                    outer_self.is_speaking = True
                    
                def on_error(client, error, **kwargs):
                    print(f"Deepgram error: {error}")
                    
                def on_close(client, close, **kwargs):
                    print(f"Deepgram connection closed: {close}")
                    
                    # Store reference to self for callback processing
                    outer_self = self
                    
                    if outer_self.microphone and hasattr(outer_self.microphone, 'is_alive') and outer_self.microphone.is_alive():
                        try:
                            outer_self.microphone.finish()
                        except Exception as e:
                            print(f"Error finishing microphone in on_close: {e}")
                            
                # Register event handlers
                self.dg_connection.on(LiveTranscriptionEvents.Open, on_open)
                self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
                self.dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
                self.dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
                self.dg_connection.on(LiveTranscriptionEvents.Error, on_error)
                self.dg_connection.on(LiveTranscriptionEvents.Close, on_close)
                
                # Start the connection
                print("Starting Deepgram connection...")
                self.dg_connection.start(options)
                print("Deepgram connection started successfully")
                
                # Only create microphone if explicitly enabled and not in server environment
                if self.microphone_enabled and not self.is_server_env:
                    try:
                        # Store the current event loop to ensure consistency
                        self.microphone = Microphone(self.dg_connection.send)
                        self.microphone.start()
                        print("Microphone started successfully.")
                    except Exception as e:
                        print(f"Failed to start microphone: {e}")
                        # Don't return, as we can still use WebSocket audio
                else:
                    print("Microphone not enabled or running in server environment - waiting for audio via WebSocket")
                
                return True
            except Exception as e:
                print(f"Failed to create Deepgram connection with SDK: {e}")
                import traceback
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"Failed to create Deepgram connection: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    async def enable_microphone(self):
        """Enable microphone if connection is established"""
        if not self.dg_connection:
            print("Cannot enable microphone: Deepgram connection not established")
            return False
            
        if self.microphone and hasattr(self.microphone, 'is_alive') and self.microphone.is_alive():
            print("Microphone already active")
            return True
            
        # Check if we're in a server environment (e.g., Render.com)
        if self.is_server_env:
            # In server environment, we don't need a physical mic but should still accept audio via WebSocket
            print("Server environment detected - continuing without physical microphone")
            self.microphone_enabled = True  # Mark as enabled to process incoming audio
            return True  # Return success
        
        # For local environment, try to access the physical microphone
        try:
            self.microphone = Microphone(self.dg_connection.send)
            self.microphone.start()
            self.microphone_enabled = True
            print("Microphone enabled and started successfully")
            return True
        except Exception as e:
            print(f"Failed to start microphone: {e}")
            return False
            
    async def disable_microphone(self):
        """Disable microphone if active"""
        if self.microphone and hasattr(self.microphone, 'finish'):
            try:
                self.microphone.finish()
                self.microphone = None
                self.microphone_enabled = False
                print("Microphone disabled")
                return True
            except Exception as e:
                print(f"Error stopping microphone: {e}")
                return False
        return True  # Already disabled
        
    async def send_to_manager_async(self):
        transcript_to_send = self.accumulated_transcript.strip()
        if transcript_to_send:
            print(f"\nHuman: {transcript_to_send}")
            # Format the transcript as a dictionary with additional metadata
            await self.manager_queue.put({
                "text": transcript_to_send,
                "is_final": True,
                "timestamp": time.time()
            })
            self.accumulated_transcript = "" # Reset for next utterance
        else:
            print("Send_to_manager called with empty transcript.")

    async def stop_listening(self):
        print("Stopping Deepgram STT...")
        
        if self.dg_connection:
            try:
                if self.microphone and hasattr(self.microphone, 'finish'):
                    try:
                        self.microphone.finish()
                        # Allow a brief moment for the microphone to clean up
                        await asyncio.sleep(0.2)
                    except Exception as e:
                        print(f"Error stopping microphone: {e}")
                        
                # According to Deepgram Python SDK 4.x, finish() is not awaitable
                try:
                    # Send a CloseStream message before finishing
                    if hasattr(self.dg_connection, 'send_message'):
                        try:
                            self.dg_connection.send_message({"type": "CloseStream"})
                            print("Sent CloseStream message")
                        except Exception as e:
                            print(f"Error sending CloseStream message: {e}")
                    
                    # Properly finish the connection - not awaitable in SDK 4.x
                    self.dg_connection.finish()
                    print("Closed Deepgram connection properly")
                except Exception as e:
                    print(f"Error during Deepgram connection finish: {e}")
                    
                print("Stopped Deepgram STT.")
            except Exception as e:
                print(f"Error during stop_listening: {e}")
        
        # Reset state regardless of errors
        self.accumulated_transcript = ""
        self.dg_connection = None
        self.microphone = None

class ConversationManager:
    def __init__(self, llm, tts, stt=None):
        """Initialize conversation manager with LLM and TTS"""
        self.llm = llm  # Language model processor
        self.tts = tts  # Text-to-speech processor
        self.stt = stt  # Speech-to-text processor (optional)
        self.transcription_queue = asyncio.Queue() if stt else None  # Queue for transcriptions from STT
        self.is_running = True  # Flag to control the main loop
        self.is_speaking = False  # Flag to track if system is speaking
        self.interrupt_event = asyncio.Event()  # Event to interrupt TTS
        self.current_llm_task = None  # Track the current LLM task
        
        # Method to cancel any background tasks
        self.tasks = []
    
    def cancel_tasks(self):
        """Cancel all background tasks"""
        for task in self.tasks:
            if not task.done():
                task.cancel()
        self.is_running = False

    async def process_llm_and_speak(self, text_input):
        """Process user input with LLM and speak the response"""
        start_time = time.time()
        
        # Skip LLM processing if input is too short or empty
        if len(text_input.strip()) <= 1:
            return None, None
            
        print(f"Processing input: {text_input}")
        
        # Process with LLM
        llm_response, llm_time = await self.llm.generate_response(text_input)
        
        if llm_response:
            # Speak the response
            self.is_speaking = True
            
            # Notify STT to disable processing if it exists
            if self.stt:
                self.stt.processing_enabled = False
            
            # Get the TTS audio data - use the TTS instance passed to the constructor
            audio_chunks = []
            async for chunk in self.tts.get_audio_data(llm_response):
                audio_chunks.append(chunk)
            
            # Concatenate audio chunks
            audio_data = b''.join(audio_chunks)
            
            # Speak the response (for local playback)
            await self.tts.speak(llm_response, self.interrupt_event)
            
            # Calculate and display total time
            total_time = int((time.time() - start_time) * 1000)
            print(f"Total processing time: {total_time}ms")
            
            # Mark speaking as done
            self.is_speaking = False
            
            # Re-enable STT processing if it exists
            if self.stt:
                self.stt.processing_enabled = True
            
            # Return response and audio data
            return llm_response, audio_data
            
        else:
            print("LLM returned no response.")
            return None, None

    async def main_loop(self):
        # Only start STT if it exists
        if self.stt:
            await self.stt.start_listening()
            self.transcription_queue = self.stt.manager_queue
        else:
            print("No speech-to-text processor provided, running in text-only mode")
            self.transcription_queue = asyncio.Queue()  # Create a queue anyway for potential manual input
            
        try:
            while self.is_running:
                try:
                    # Use a short timeout to allow for regular checks
                    user_input = await asyncio.wait_for(self.transcription_queue.get(), timeout=0.1)
                    
                    # Only process if we're not in the middle of speaking
                    # or if the user explicitly interrupted
                    if self.is_speaking:
                        print("User spoke while system was speaking.")
                        self.interrupt_event.set()  # Signal that new user input has arrived
                    
                        if self.current_llm_task and not self.current_llm_task.done():
                            print("Cancelling previous task due to user interruption.")
                            try:
                                self.current_llm_task.cancel()
                                # Allow some time for the task to actually cancel
                                await asyncio.sleep(0.1)
                            except Exception as e:
                                print(f"Error cancelling LLM task: {e}")
                            
                            try:
                                await self.tts.stop_playback()  # Ensure TTS stops
                            except Exception as e:
                                print(f"Error stopping TTS playback: {e}")
                    else:
                        # Normal flow - not speaking, so process the user input
                        if user_input:
                            # Handle both string and dictionary formats
                            if isinstance(user_input, dict):
                                input_text = user_input.get("text", "")
                            else:
                                input_text = str(user_input)
                                
                            if "that's enough" in input_text.lower() or \
                            "stop listening" in input_text.lower() or \
                            "goodbye" in input_text.lower():
                                print("Exit phrase detected. Shutting down.")
                                if not self.interrupt_event.is_set():  # Don't speak if immediately interrupted
                                    try:
                                        await self.tts.speak("Okay, goodbye!", self.interrupt_event)
                                    except Exception as e:
                                        print(f"Error during goodbye message: {e}")
                                self.is_running = False
                                break
                            
                            self.interrupt_event.clear()  # Clear before starting new task
                            try:
                                self.current_llm_task = asyncio.create_task(self.process_llm_and_speak(input_text))
                            except Exception as e:
                                print(f"Error creating LLM task: {e}")

                except asyncio.TimeoutError:
                    pass  # No new transcript, continue listening
                except asyncio.CancelledError:
                    print("Main loop task was cancelled.")
                    break
                except Exception as e:
                    print(f"Unexpected error in main loop: {e}")
                
                if self.current_llm_task and self.current_llm_task.done():
                    try:
                        await self.current_llm_task  # Propagate exceptions
                    except asyncio.CancelledError:
                        print("LLM task finished (was cancelled).")
                    except Exception as e:
                        print(f"LLM task finished with error: {e}")
                    self.current_llm_task = None
                    if self.is_running and not self.interrupt_event.is_set():  # Only print Listening if not shutting down and not just interrupted
                         print("Listening...")

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Shutting down...")
            self.is_running = False
        except Exception as e:
            print(f"Critical error in main loop: {e}")
        finally:
            print("Cleaning up...")
            self.interrupt_event.set()  # Signal to any ongoing tasks to stop
            if self.current_llm_task and not self.current_llm_task.done():
                try:
                    self.current_llm_task.cancel()
                    # Wait for cancellation to complete with timeout
                    try:
                        await asyncio.wait_for(asyncio.shield(self.current_llm_task), timeout=2.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass  # Expected
                except Exception as e:
                    print(f"Error during LLM task cancellation: {e}")
            
            if self.stt:
                try:
                    await self.stt.stop_listening()
                except Exception as e:
                    print(f"Error stopping STT: {e}")
                
            try:
                await self.tts.stop_playback()  # Ensure TTS is stopped on cleanup
            except Exception as e:
                print(f"Error stopping TTS: {e}")
                
            print("Shutdown complete.")

class DeepgramTTS:
    def __init__(self, api_key):
        self.api_key = api_key
        # self.client = DeepgramClient(api_key) # Not used if using requests directly
        self.temp_file = None
        self.player_process = None
        self.is_playing = False
        self.audio_data = None
        self.play_command = ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", "-"]
        
    async def get_audio_data(self, text):
        """Get audio data for text without playing it"""
        # URL for Deepgram TTS API
        DEEPGRAM_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
        
        # Set up headers and payload for the TTS request
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"text": text}
        
        # Send the request and get the response
        response = await asyncio.to_thread(requests.post, DEEPGRAM_URL, headers=headers, json=payload, stream=True)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"Error: Failed to get audio from Deepgram TTS. Status code: {response.status_code}")
            return b""
        
        # Return audio data as bytes iterator for streaming
        async def audio_stream():
            chunks = []
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
                    yield chunk
            # Store audio data for later use
            if chunks:
                self.audio_data = b''.join(chunks)
            else:
                print(f"TTS: No audio data received from Deepgram for: {text}")
                self.audio_data = None
        
        return audio_stream()
        
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

        print(f"Converting to speech: {text}")
        # print(f"Sending TTS request to Deepgram...") # A bit verbose
        start_time = time.time()

        try:
            # Using asyncio.to_thread for the blocking requests call
            response = await asyncio.to_thread(requests.post, DEEPGRAM_URL, headers=headers, json=payload, stream=True)
            
            response.raise_for_status()

            if self.player_process and self.player_process.poll() is None:
                print("Warning: Previous player process still running. Terminating.")
                await self.stop_playback()

            self.player_process = subprocess.Popen(self.play_command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            audio_received_time = None
            chunk_count = 0
            
            # Wrap the audio processing in a try block to catch pipe errors
            try:
                for chunk in response.iter_content(chunk_size=4096): # Increased chunk_size
                    if interrupt_event and interrupt_event.is_set():
                        print("TTS: Interrupt received, stopping playback.")
                        response.close() # Close the response stream
                        await self.stop_playback()
                        return

                    if chunk:
                        if audio_received_time is None: # First chunk of audio
                            audio_received_time = time.time()
                            actual_ttfb = int((audio_received_time - start_time) * 1000)
                            print(f"TTS Time to First Audio Byte: {actual_ttfb}ms")
                        
                        try:
                            if self.player_process and self.player_process.stdin:
                                self.player_process.stdin.write(chunk)
                            else: # Player closed prematurely
                                print("TTS: Player process stdin is not available. Stopping.")
                                response.close()
                                break 
                        except BrokenPipeError:
                            print("TTS: Playback pipe broken. Player likely closed or interrupted.")
                            response.close()
                            break 
                        except Exception as e: # Catch other potential stdin write errors
                            print(f"TTS: Error writing to player stdin: {e}")
                            response.close()
                            break
                    chunk_count +=1
            except BrokenPipeError:
                print("TTS: BrokenPipeError during audio chunk processing - handling gracefully")
                response.close()
            except Exception as e:
                print(f"TTS: Error during audio chunk processing: {e}")
                response.close()
            
            if chunk_count == 0 and audio_received_time is None: # No audio data received
                print(f"TTS: No audio data received from Deepgram for: {text}")

            if self.player_process and self.player_process.stdin:
                try:
                    self.player_process.stdin.close()
                except BrokenPipeError:
                    print("TTS: Pipe already broken during cleanup - this is expected")
                except Exception as e:
                    print(f"TTS: Error closing player stdin: {e}")
            
            if self.player_process:
                try:
                    # Wait for the player to finish naturally
                    start_wait_time = time.time()
                    max_wait_time = 30.0  # Maximum time to wait for audio to finish (30 seconds)
                    
                    # Wait for playback to complete or for an interrupt
                    while self.player_process.poll() is None:
                        if interrupt_event and interrupt_event.is_set():
                            print("TTS: Interrupt received while waiting for player. Terminating.")
                            await self.stop_playback()
                            return
                            
                        # Check if we've been waiting too long
                        current_wait_time = time.time() - start_wait_time
                        if current_wait_time > max_wait_time:
                            print(f"TTS: Playback exceeded maximum wait time ({max_wait_time}s). Forcing termination.")
                            await self.stop_playback()
                            break
                            
                        # Short wait to avoid blocking the event loop
                        await asyncio.sleep(0.05)
                    
                    rc = self.player_process.returncode if self.player_process else None
                    if rc is not None and rc != 0:
                        print(f"TTS: Player process exited with code {rc}")
                    
                    print("Audio playback complete")
                    self.player_process = None
                except Exception as e:
                    print(f"TTS: Error while waiting for player process: {e}")
                    await self.stop_playback()

        except requests.exceptions.RequestException as e:
            print(f"TTS Request failed: {e}")
        except Exception as e:
            print(f"TTS Error: {e}")
            await self.stop_playback() 
        finally:
            if self.player_process and self.player_process.poll() is None:
                try:
                    await self.stop_playback()
                except Exception as e:
                    print(f"TTS: Error during cleanup in finally block: {e}")

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

async def main_async(first_message=None, text_only=False):
    # Initialize your LLM and TTS with real implementations
    llm = LanguageModelProcessor() 
    
    # Use Rime TTS instead of Deepgram TTS
    try:
        from realism_voice.io.rime_tts_async import RimeTTS
        tts = RimeTTS()
        print("Successfully initialized Rime TTS")
    except Exception as e:
        print(f"Error initializing Rime TTS: {e}")
        print("Falling back to Deepgram TTS")
        # Use Deepgram TTS as fallback
        deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            print("Warning: DEEPGRAM_API_KEY not set. TTS will not work.")
            print("Please add DEEPGRAM_API_KEY to your .env file.")
        tts = DeepgramTTS(deepgram_api_key)
    
    # If no first message provided, use the greeting
    if not first_message:
        first_message = "Hey how you doing my dawg!!"
    
    # Only initialize STT if not in text_only mode
    stt = None
    if not text_only:
        try:
            print("Initializing speech-to-text...")
            stt_queue = asyncio.Queue()
            stt = DeepgramSTT(stt_queue)
            print("STT initialized successfully")
        except Exception as e:
            print(f"Error initializing STT: {e}")
            print("Running in text-only mode")
            stt = None  # Explicitly set to None in case of error

    manager = ConversationManager(llm, tts, stt)
    
    if first_message:
        print(f"Agent: {first_message}")
        if not text_only:
            try:
                # Create a dummy event for the first message if needed, or handle None
                await tts.speak(first_message, interrupt_event=asyncio.Event())
            except Exception as e:
                print(f"Error during initial greeting: {e}")
                print("Continuing without initial speech output")

    # Set up proper signal handling for cleaner shutdown
    loop = asyncio.get_running_loop()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(manager)))
    except NotImplementedError:
        # Windows doesn't support signals properly
        pass

    try:
        await manager.main_loop()
    except Exception as e:
        print(f"Critical exception in main_async: {e}")
        import traceback
        traceback.print_exc()

async def shutdown(manager):
    # Cancel all tasks
    print("Shutting down... Cleaning up resources.")
    if manager:
        manager.cancel_tasks()
    
    # Sleep a bit to allow tasks to clean up
    await asyncio.sleep(1)

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Realism Voice API is running", "endpoints": [
        "/chat - POST: Send text to chat with the AI",
        "/tts - POST: Convert text to speech",
        "/voice/ws - WebSocket: Real-time voice conversation"
    ]}

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.body()
    text = body.decode()
    
    # Initialize language model
    llm = LanguageModelProcessor()
    response, _ = await llm.generate_response(text)
    return JSONResponse({"response": response})

@app.post("/tts")
async def tts_endpoint(request: Request):
    body = await request.body()
    text = body.decode()
    
    # Try to use Rime TTS first, fall back to Deepgram if needed
    try:
        from realism_voice.io.rime_tts_async import RimeTTS
        if os.getenv("RIME_API_KEY"):
            print(f"Using Rime TTS for: {text}")
            tts = RimeTTS()
            
            # Use the new get_audio_data method which returns a generator of audio chunks
            return StreamingResponse(tts.get_audio_data(text), media_type="audio/mp3")
    except Exception as e:
        print(f"Error using Rime TTS: {e}")
        print("Falling back to Deepgram TTS")
    
    # Fallback to Deepgram TTS
    if not os.getenv("DEEPGRAM_API_KEY"):
        return JSONResponse({"error": "No TTS providers available. Set DEEPGRAM_API_KEY or RIME_API_KEY."}, status_code=500)
    
    tts = DeepgramTTS(os.getenv("DEEPGRAM_API_KEY"))
    audio_data = await tts.get_audio_data(text)
    
    # Return audio as streaming response
    return StreamingResponse(audio_data, media_type="audio/mp3")

# WebSocket for voice streaming
@app.websocket("/voice/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    print("WebSocket connection accepted")
    
    # Create a queue for communication between STT and this handler
    transcription_queue = asyncio.Queue()
    
    # Initialize STT with the queue
    try:
        stt = DeepgramSTT(transcription_queue)
    except ValueError as e:
        print(f"Error initializing STT: {e}")
        await websocket.close(1008, "Failed to initialize STT - API key missing")
        return
    
    # Initialize LLM
    llm = LanguageModelProcessor()
    
    # Try to use Rime TTS first, fall back to Deepgram if needed
    try:
        from realism_voice.io.rime_tts_async import RimeTTS
        if os.getenv("RIME_API_KEY"):
            print("Using Rime TTS for WebSocket")
            tts = RimeTTS()
        else:
            # Fall back to Deepgram TTS
            print("RIME_API_KEY not set, falling back to Deepgram TTS")
            if not os.getenv("DEEPGRAM_API_KEY"):
                await websocket.close(1008, "No TTS providers available")
                return
            tts = DeepgramTTS(os.getenv("DEEPGRAM_API_KEY"))
    except Exception as e:
        print(f"Error initializing Rime TTS: {e}")
        print("Falling back to Deepgram TTS")
        if not os.getenv("DEEPGRAM_API_KEY"):
            await websocket.close(1008, "No TTS providers available")
            return
        tts = DeepgramTTS(os.getenv("DEEPGRAM_API_KEY"))
    
    # Create conversation manager
    manager = ConversationManager(llm, tts, stt)
    
    # Start STT listening but DON'T enable microphone by default
    try:
        print("Starting STT listening...")
        # Pass False to not enable microphone automatically
        success = await stt.start_listening(enable_microphone=False)
        
        # Check if connection was properly initialized
        if not success:
            print("Error: Failed to start STT listening")
            await websocket.close(1011, "Failed to initialize STT connection")
            return
            
        print("STT listening started successfully (microphone disabled)")
        
        # Send initialization confirmation to client
        await websocket.send_json({"type": "connection", "status": "ready", "microphone": "disabled"})
    except Exception as e:
        print(f"Error starting STT: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close(1011, "Failed to start speech recognition")
        except Exception as close_err:
            print(f"Error during close after initialization error: {close_err}")
        return
    
    # Create a task for processing transcriptions
    process_task = None
    
    try:
        # Start a background task to process transcription queue
        process_task = asyncio.create_task(process_transcriptions(websocket, transcription_queue, manager))
        print("Started background task for processing transcriptions")
        
        # Receive messages from client
        while True:
            try:
                # Wait for data with a timeout to allow for checks
                try:
                    data = await asyncio.wait_for(websocket.receive(), timeout=0.5)
                    message_type = data.get("type", "unknown")
                    print(f"Received message type: {message_type}")
                    
                    # Check for disconnect message
                    if message_type == "websocket.disconnect":
                        print(f"Received disconnect message: {data}")
                        break
                    
                    # Handle text messages (commands and text input)
                    if message_type == "websocket.receive":
                        if "text" in data:
                            # Process JSON messages
                            try:
                                text_data = data["text"]
                                print(f"Raw text data: {text_data}")
                                message_data = json.loads(text_data)
                                message = message_data.get("text", "")
                                print(f"Processed text message: {message}")
                                
                                if message.startswith('/'):
                                    # Command handling
                                    command = message[1:].strip().lower()
                                    print(f"Processing command: {command}")
                                    
                                    if command == "mic_on" or command == "start_recording":
                                        # Enable microphone
                                        success = await stt.enable_microphone()
                                        if success:
                                            print("Microphone enabled successfully")
                                            await websocket.send_json({"type": "status", "message": "Microphone enabled", "microphone": "enabled"})
                                        else:
                                            print("Failed to enable microphone")
                                            await websocket.send_json({"type": "error", "message": "Failed to enable microphone"})
                                    
                                    elif command == "mic_off" or command == "stop_recording":
                                        # Disable microphone
                                        success = await stt.disable_microphone()
                                        if success:
                                            print("Microphone disabled successfully")
                                            await websocket.send_json({"type": "status", "message": "Microphone disabled", "microphone": "disabled"})
                                        else:
                                            print("Failed to disable microphone")
                                            await websocket.send_json({"type": "error", "message": "Failed to disable microphone"})
                                            
                                    else:
                                        print(f"Unknown command: {command}")
                                        await websocket.send_json({"type": "error", "message": f"Unknown command: {command}"})
                                else:
                                    # Regular text message - process as user input
                                    print(f"Adding text to transcription queue: {message}")
                                    transcription_queue.put_nowait({
                                        "text": message,
                                        "is_final": True,
                                        "timestamp": time.time()
                                    })
                            except json.JSONDecodeError as e:
                                print(f"Error decoding JSON from websocket: {e}")
                                print(f"Raw text was: {data.get('text', 'unknown')}")
                                await websocket.send_json({"type": "error", "message": f"Invalid JSON: {str(e)}"})
                        elif "bytes" in data:
                            # Process binary data (audio)
                            audio_data = data["bytes"]
                            print(f"Received binary audio data: {len(audio_data)} bytes")
                            
                            if not stt.dg_connection:
                                print("Error: Deepgram connection is None, cannot send data")
                                break
                                
                            try:
                                # Send audio data to Deepgram
                                # With SDK 4.x, send is not awaitable
                                stt.dg_connection.send(audio_data)
                                
                                # Print occasional heartbeat to confirm data flow (not every chunk to avoid spamming logs)
                                if len(audio_data) % 10000 < 100:  # Print roughly every 10KB
                                    print(f"Sent audio chunk to Deepgram: {len(audio_data)} bytes")
                            except Exception as e:
                                print(f"Error sending data to Deepgram: {e}")
                                import traceback
                                traceback.print_exc()
                                break
                        else:
                            print(f"Unsupported message content: {data}")
                    else:
                        print(f"Received unknown message type: {message_type}")
                        print(f"Full message: {data}")
                except asyncio.TimeoutError:
                    # No data received in timeout period - this is normal
                    # Check if process_task is still running
                    if process_task and process_task.done():
                        print("Processing task has ended unexpectedly")
                        break
                    continue
            except WebSocketDisconnect:
                print("WebSocket disconnected during receive")
                break
            except starlette.websockets.WebSocketDisconnect:
                # Also catch starlette's WebSocketDisconnect specifically
                print("Starlette WebSocket disconnected during receive")
                break
            except Exception as e:
                print(f"Error receiving data from WebSocket: {e}")
                import traceback
                traceback.print_exc()
                break
    
    except WebSocketDisconnect:
        print("WebSocket disconnected by client")
    except Exception as e:
        print(f"Error in WebSocket: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up - make sure these operations are safe even if the websocket is already closed
        print("Cleaning up WebSocket resources...")
        
        # Cancel the processing task if it's running
        if process_task and not process_task.done():
            process_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(process_task), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                print(f"Error during process task cancellation (non-critical): {e}")
        
        # Stop STT listening
        try:
            await stt.stop_listening()
        except Exception as e:
            print(f"Error stopping STT (non-critical): {e}")
        
        # Attempt to close the websocket if it's not already closed
        try:
            await websocket.close()
        except RuntimeError as e:
            # This is expected if the websocket is already closed
            print(f"Note: WebSocket was already closed: {e}")
        except Exception as e:
            print(f"Error during websocket.close (non-critical): {e}")
            
        print("WebSocket cleanup complete")

async def process_transcriptions(websocket, queue, manager):
    print("Starting transcription processing task")
    try:
        await websocket.send_json({"type": "status", "message": "Transcription service ready"})
    except Exception as e:
        print(f"Error sending initial status message: {e}")
    
    while True:
        try:
            # Get transcription from queue
            transcription = await queue.get()
            print(f"Received transcription: {transcription}")
            
            # Format for sending to client
            if isinstance(transcription, dict):
                # Send transcription to client
                await websocket.send_json({
                    "type": "transcript", 
                    "text": transcription.get("text", ""), 
                    "is_final": transcription.get("is_final", False),
                    "timestamp": transcription.get("timestamp", time.time())
                })
                print(f"Sent transcript to client: {transcription.get('text', '')}")
                
                # Process with LLM and TTS if it's a final transcription
                if transcription.get("is_final", False):
                    try:
                        # Get text from transcription
                        input_text = transcription.get("text", "")
                        if not input_text.strip():
                            print("Empty transcript, skipping LLM processing")
                            continue
                            
                        # Send processing status to client
                        await websocket.send_json({"type": "status", "message": "Processing with AI..."})
                            
                        print(f"Processing with LLM: {input_text}")
                        response, audio_data = await manager.process_llm_and_speak(input_text)
                        
                        if response:
                            # Send response text to client
                            await websocket.send_json({"type": "response", "text": response})
                            print(f"Sent response to client: {response}")
                            
                            # Send audio data in chunks if available
                            if audio_data:
                                try:
                                    audio_blob = audio_data  # Already binary data
                                    chunk_size = 8192
                                    
                                    # First, send a JSON message indicating audio is coming
                                    await websocket.send_json({"type": "audio_start", "size": len(audio_blob)})
                                    
                                    # Then send the audio in chunks
                                    for i in range(0, len(audio_blob), chunk_size):
                                        chunk = audio_blob[i:i+chunk_size]
                                        await websocket.send_bytes(chunk)
                                    
                                    # Signal that audio transmission is complete
                                    await websocket.send_json({"type": "audio_end"})
                                    print(f"Sent {len(audio_data)} bytes of audio data in chunks")
                                    
                                    # Tell client we're ready for more input
                                    await websocket.send_json({"type": "status", "message": "Ready for more input"})
                                except Exception as e:
                                    print(f"Error sending audio data: {e}")
                        else:
                            print("No response from LLM")
                            await websocket.send_json({"type": "status", "message": "No response generated"})
                    except Exception as e:
                        print(f"Error processing with LLM: {e}")
                        import traceback
                        traceback.print_exc()
                        await websocket.send_json({"type": "error", "message": "Error processing your request"})
            else:
                # Regular text message - process as user input
                # This allows testing with typed inputs even when mic is off
                print(f"Adding text to transcription queue: {transcription}")
                queue.put_nowait({
                    "text": transcription,
                    "is_final": True,
                    "timestamp": time.time()
                })
        except asyncio.CancelledError:
            print("Transcription processing task cancelled")
            break
        except Exception as e:
            print(f"Error processing transcription: {e}")
            import traceback
            traceback.print_exc()
            # Continue processing next transcription
            continue
            
    print("Transcription processing task ended")

if __name__ == "__main__":
    # Check if running directly or through uvicorn
    if os.getenv("PORT"):
        # Running on Render.com, use the PORT environment variable
        port = int(os.getenv("PORT", 10000))
        host = "0.0.0.0"
        print(f"Starting API server on {host}:{port}")
        # Run in API server mode only (no microphone)
        uvicorn.run(app, host=host, port=port)
    else:
        # Local development - can use interactive mode with microphone
        try:
            # Try to run the interactive agent
            asyncio.run(main_async())
        except ModuleNotFoundError as e:
            if "pyaudio" in str(e).lower():
                print(f"Warning: {e}")
                print("PyAudio is required for microphone access.")
                print("Install with: pip install pyaudio")
                print("On Linux: sudo apt-get install python3-pyaudio portaudio19-dev")
                print("On macOS: brew install portaudio && pip install pyaudio")
            print("Falling back to API-only mode (no microphone)")
            port = 8000
            host = "127.0.0.1"
            uvicorn.run(app, host=host, port=port) 
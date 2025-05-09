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
    def __init__(self, manager_queue):
        self.manager_queue = manager_queue
        config = DeepgramClientOptions(options={"keepalive": "true"})
        self.client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)
        self.dg_connection = None
        self.microphone = None
        self.is_speaking = False
        self.accumulated_transcript = ""
        self.processing_enabled = True  # Flag to enable/disable transcript processing
        # Store the event loop that this object was created on
        self.loop = asyncio.get_event_loop()
        
        # Check if we're running in a server environment (like Render.com)
        self.is_server_env = os.getenv("PORT") is not None
        if self.is_server_env:
            print("Running in server environment - microphone access will be disabled")

    async def start_listening(self):
        # Skip microphone initialization in server environments
        if self.is_server_env:
            print("Microphone initialization skipped (server environment)")
            return
            
        # Ensure we're on the same event loop
        if asyncio.get_event_loop() != self.loop:
            print("Warning: Event loop mismatch. This might cause issues.")
        
        # The configuration was moved to the client initialization in __init__
        self.dg_connection = self.client.listen.asynclive.v("1")

        # Fix: Define async event handlers properly
        async def on_open_async(connection, **kwargs):
            print("Deepgram connection opened.")
            
        async def on_message_async(connection, result, **kwargs):
            # Only process if speech processing is enabled (not speaking)
            if not self.processing_enabled:
                return
                
            sentence = result.channel.alternatives[0].transcript
            
            if result.is_final and result.speech_final:
                self.accumulated_transcript = result.channel.alternatives[0].transcript
                pass
            elif not result.is_final and len(sentence) > 0: # Interim result
                current_transcript_interim = result.channel.alternatives[0].transcript
                print(f"Hearing: {current_transcript_interim}...", end='\r', flush=True)
                self.accumulated_transcript = current_transcript_interim
        
        async def on_utterance_end_async(connection, utterance_end, **kwargs):
            # Only process if speech processing is enabled (not speaking)
            if not self.processing_enabled:
                return
                
            print("\nUser likely finished speaking (UtteranceEnd).")
            self.is_speaking = False # User has stopped.
            if len(self.accumulated_transcript.strip()) > 0:
                try:
                    await self.send_to_manager_async()
                except Exception as e:
                    print(f"Error sending transcript to manager: {e}")
            else:
                print("UtteranceEnd received, but no transcript accumulated to send.")
        
        async def on_speech_started_async(connection, speech_started, **kwargs):
            # Only process if speech processing is enabled (not speaking)
            if not self.processing_enabled:
                return
                
            print("User started speaking.")
            self.is_speaking = True
            
        async def on_error_async(connection, error, **kwargs):
            print(f"Deepgram error: {error}")
            
        async def on_close_async(connection, **kwargs):
            print("Deepgram connection closed.")
            if self.microphone and hasattr(self.microphone, 'is_alive') and self.microphone.is_alive():
                try:
                    self.microphone.finish()
                except Exception as e:
                    print(f"Error finishing microphone in on_close: {e}")

        # Register the async handlers
        self.dg_connection.on(LiveTranscriptionEvents.Open, on_open_async)
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message_async)
        self.dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end_async)
        self.dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started_async)
        self.dg_connection.on(LiveTranscriptionEvents.Error, on_error_async)
        self.dg_connection.on(LiveTranscriptionEvents.Close, on_close_async)

        # More focused options for accurate speech detection
        options = LiveOptions(
            model="nova-2", # or nova-3 if available and preferred
            language="en-US",
            smart_format=True,
            encoding="linear16", # Make sure this matches your microphone
            channels=1,
            sample_rate=16000,   # Make sure this matches your microphone
            interim_results=True,
            utterance_end_ms=1500, # Increased from 1000ms to 1500ms to avoid premature end detection
            vad_events=True,
            endpointing=500,     # Increased from 300ms to 500ms to wait longer for more speech
        )

        print("Listening...")
        # The start method in AsyncLive version does need an await
        try:
            await self.dg_connection.start(options)
            
            # Create and start microphone after successful connection
            try:
                # Store the current event loop to ensure consistency
                self.microphone = Microphone(self.dg_connection.send)
                self.microphone.start()
                print("Microphone started successfully.")
            except Exception as e:
                print(f"Failed to start microphone: {e}")
                await self.dg_connection.finish()  # Clean up connection if microphone fails
                return
                
        except Exception as e:
            print(f"Failed to start Deepgram connection: {e}")
            return

    # Remove old instance methods as they're now handled by the local async functions
    async def send_to_manager_async(self):
        transcript_to_send = self.accumulated_transcript.strip()
        if transcript_to_send:
            print(f"\nHuman: {transcript_to_send}")
            await self.manager_queue.put(transcript_to_send)
            self.accumulated_transcript = "" # Reset for next utterance
        else:
            print("Send_to_manager called with empty transcript.")

    async def stop_listening(self):
        if self.dg_connection:
            try:
                if self.microphone and hasattr(self.microphone, 'finish'):
                    try:
                        self.microphone.finish()
                        # Allow a brief moment for the microphone to clean up
                        await asyncio.sleep(0.2)
                    except Exception as e:
                        print(f"Error stopping microphone: {e}")
                        
                # According to Deepgram Python SDK, finish() is the correct way to close.
                # It handles closing the websocket and stopping the listener.
                try:
                    await self.dg_connection.finish()
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
        self.transcription_queue = None  # Queue for transcriptions from STT
        self.is_running = True  # Flag to control the main loop
        self.is_speaking = False  # Flag to track if system is speaking
        self.interrupt_event = asyncio.Event()  # Event to interrupt TTS
        
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
            
            # Notify STT to disable processing
            self.stt.processing_enabled = False
            
            # Get the TTS audio data
            tts = DeepgramTTS(os.getenv("DEEPGRAM_API_KEY"))
            audio_stream = await tts.get_audio_data(llm_response)
            
            # Collect audio chunks
            audio_chunks = []
            async for chunk in audio_stream:
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
            
            # Re-enable STT processing
            self.stt.processing_enabled = True
            
            # Return response and audio data
            return llm_response, audio_data
            
        else:
            print("LLM returned no response.")
            return None, None

    async def main_loop(self):
        await self.stt.start_listening()
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
                            if "that's enough" in user_input.lower() or \
                            "stop listening" in user_input.lower() or \
                            "goodbye" in user_input.lower():
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
                                self.current_llm_task = asyncio.create_task(self.process_llm_and_speak(user_input))
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

    manager = ConversationManager(llm, tts)
    
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
    
    # Use Deepgram TTS
    if not os.getenv("DEEPGRAM_API_KEY"):
        return JSONResponse({"error": "DEEPGRAM_API_KEY not set"}, status_code=500)
    
    tts = DeepgramTTS(os.getenv("DEEPGRAM_API_KEY"))
    audio_data = await tts.get_audio_data(text)
    
    # Return audio as streaming response
    return StreamingResponse(audio_data, media_type="audio/mp3")

# WebSocket for voice streaming
@app.websocket("/voice/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create a queue for communication between STT and this handler
    transcription_queue = asyncio.Queue()
    
    # Initialize STT with the queue
    stt = DeepgramSTT(transcription_queue)
    
    # Initialize LLM and TTS
    llm = LanguageModelProcessor()
    tts = DeepgramTTS(os.getenv("DEEPGRAM_API_KEY"))
    
    # Create conversation manager
    manager = ConversationManager(llm, tts)
    
    # Start STT listening
    await stt.start_listening()
    
    try:
        # Start a background task to process transcription queue
        process_task = asyncio.create_task(process_transcriptions(websocket, transcription_queue, manager))
        
        # Receive audio data from client and send to STT
        while True:
            data = await websocket.receive_bytes()
            stt.connection.send(data)
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error in WebSocket: {e}")
    finally:
        # Clean up
        process_task.cancel()
        await stt.stop_listening()
        await websocket.close()

async def process_transcriptions(websocket, queue, manager):
    while True:
        try:
            # Get transcription from queue
            transcription = await queue.get()
            
            # Format for sending to client
            if isinstance(transcription, dict):
                # Send transcription to client
                await websocket.send_json({"type": "transcript", "text": transcription.get("text", ""), "is_final": transcription.get("is_final", False)})
                
                # Process with LLM and TTS if it's a final transcription
                if transcription.get("is_final", False):
                    response, audio_data = await manager.process_llm_and_speak(transcription.get("text", ""))
                    
                    if response and audio_data:
                        # Send response text to client
                        await websocket.send_json({"type": "response", "text": response})
                        
                        # Send audio data in chunks
                        chunk_size = 8192
                        for i in range(0, len(audio_data), chunk_size):
                            chunk = audio_data[i:i+chunk_size]
                            await websocket.send_bytes(chunk)
            else:
                # Simple string transcription
                await websocket.send_json({"type": "transcript", "text": str(transcription), "is_final": True})
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error processing transcription: {e}")
            # Continue processing next transcription
            continue

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
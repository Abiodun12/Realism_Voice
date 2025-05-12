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
            model="nova-3",  # Changed from nova-2 to nova-3
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
        # Ensure DEEPGRAM_API_KEY is loaded and available
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            print("Error: DEEPGRAM_API_KEY environment variable is not set. STT will not function.")
            # Consider raising an error or having a more robust way to handle this
        
        # Configure Deepgram client
        # Options for keepalive, etc., can be set here
        self.client_config = DeepgramClientOptions(
            options={"keepalive": "true"} 
        )
        self.client = DeepgramClient(self.api_key if self.api_key else "", self.client_config) # Pass empty string if no key, though SDK might handle None

        self.manager_queue = manager_queue
        self.is_listening = False
        self.microphone = None
        self.dg_connection = None
        self.transcript_collector = TranscriptCollector()
        self.utterance_end_timer = None
        self.speech_started = False
        self.speech_processing_enabled = True # Control flag
        self.active_audio_task = None # To keep track of the audio processing task

    async def start_listening(self):
        if not self.api_key:
            print("Deepgram API key not set. STT cannot start.")
            return

        # Use the recommended asyncwebsocket
        self.dg_connection = self.client.listen.asyncwebsocket.v("1")

        # Define event handlers. The SDK passes the client instance as the first argument
        # to these callbacks, even if they are nested functions.
        async def on_open_async(client_instance, open_payload, **kwargs):
            print(f"Deepgram connection opened via: {client_instance}")
            print(f"Deepgram on_open_async PAYLOAD: {open_payload}")
            self.is_listening = True # Explicitly set is_listening to True on successful open
            # request_id = open_payload.headers.get("dg-request-id") # Example if payload has headers
            # print(f"Request ID: {request_id}")

        async def on_message_async(client_instance, result, **kwargs):
            if not self.is_listening or not self.speech_processing_enabled:
                return # Ignore transcripts if not actively listening or speech processing is disabled

            sentence = result.channel.alternatives[0].transcript
            if not sentence.strip():
                return

            if result.is_final:
                self.transcript_collector.add_part(sentence)
                if result.speech_final:
                    full_transcript = self.transcript_collector.get_full_transcript().strip()
                    if full_transcript:
                        print(f"Human: {full_transcript}")
                        await self.manager_queue.put({'type': 'user_speech', 'data': full_transcript})
                        self.transcript_collector.reset()
                        self.speech_started = False
                    if self.utterance_end_timer:
                        self.utterance_end_timer.cancel()
                        self.utterance_end_timer = None
                else:
                    pass # is_final but not speech_final, collected.
            else: # Interim result
                interim_transcript = sentence.strip()
                if interim_transcript:
                    if self.utterance_end_timer:
                        self.utterance_end_timer.cancel()
                    self.utterance_end_timer = asyncio.create_task(self.send_to_manager_after_delay(1.0))

        async def on_utterance_end_async(client_instance, utterance_end, **kwargs):
            if not self.speech_processing_enabled:
                return
            # Add guard for is_listening here as well
            if not self.is_listening:
                return
            print("User likely finished speaking (UtteranceEnd).")
            if self.speech_started and not self.utterance_end_timer:
                full_transcript = self.transcript_collector.get_full_transcript().strip()
                if full_transcript:
                    print(f"Human (from UtteranceEnd): {full_transcript}")
                    await self.manager_queue.put({'type': 'user_speech', 'data': full_transcript})
                    self.transcript_collector.reset()
                self.speech_started = False

        async def on_speech_started_async(client_instance, *, speech_started, **kwargs):
            # Signature based on observed SDK behavior: client_instance as pos arg,
            # and speech_started_payload (SpeechStartedResponse) as a kwarg.
            if not self.speech_processing_enabled:
                return
            print(f"User started speaking (client: {client_instance}, payload: {speech_started}).")
            self.speech_started = True
            if self.utterance_end_timer:
                self.utterance_end_timer.cancel()
                self.utterance_end_timer = None

        async def on_error_async(client_instance, error, **kwargs):
            print(f"Deepgram error (via client {client_instance}): {error}")
            if hasattr(error, 'message'):
                print(f"Error message: {error.message}")
            # For websockets.exceptions.InvalidStatusCode, headers are on the exception object itself
            if hasattr(error, 'headers'):
                 dg_error_msg = error.headers.get("dg-error")
                 dg_request_id = error.headers.get("dg-request-id")
                 if dg_error_msg:
                     print(f"dg-error header: {dg_error_msg}")
                 if dg_request_id:
                     print(f"dg-request-id header: {dg_request_id}")
            # For other types of errors, the structure might be different.
            # Consider logging the type of error: print(f"Error type: {type(error)}")

        async def on_close_async(client_instance, *, close, **kwargs):
            # Signature based on observed SDK behavior: client_instance as pos arg,
            # and close_payload (CloseResponse) as a kwarg.
            print(f"Deepgram connection closed by {client_instance}. Payload: {close}")
            self.is_listening = False

        # Register event handlers
        self.dg_connection.on(LiveTranscriptionEvents.Open, on_open_async)
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message_async)
        self.dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end_async)
        self.dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started_async)
        self.dg_connection.on(LiveTranscriptionEvents.Error, on_error_async)
        self.dg_connection.on(LiveTranscriptionEvents.Close, on_close_async)

        options = LiveOptions(
            model="nova-3", # Changed from nova-2 to nova-3
            language="en-US",
            smart_format=True,
            punctuate=True,
            interim_results=True, # Get interim results for faster feedback
            utterance_end_ms="1000", # How long to wait for UtteranceEnd after speech
            vad_events=True, # Voice Activity Detection events (SpeechStarted, UtteranceEnd)
            sample_rate=16000, # Ensure this matches your microphone's sample rate
            encoding="linear16",
            channels=1
        )

        try:
            if await self.dg_connection.start(options) is False:
                print("Failed to connect to Deepgram with new settings.")
                self.is_listening = False
                return
            
            print("Deepgram connection started successfully with asyncwebsocket.")
            self.is_listening = True

            # Initialize and start the microphone
            self.microphone = Microphone(self.dg_connection.send)
            self.microphone.start()
            print("Microphone started successfully.")

        except Exception as e:
            print(f"Error starting Deepgram connection or microphone: {e}")
            self.is_listening = False
            if self.dg_connection:
                await self.dg_connection.finish() # Ensure connection is closed on error

    async def send_to_manager_after_delay(self, delay):
        await asyncio.sleep(delay)
        # Add guard for is_listening and speech_processing_enabled
        if not self.is_listening or not self.speech_processing_enabled:
            self.utterance_end_timer = None # Ensure timer is cleared if we bail early
            return
        
        if self.transcript_collector.get_full_transcript().strip():
            full_transcript = self.transcript_collector.get_full_transcript().strip()
            print(f"Human (after delay): {full_transcript}")
            await self.manager_queue.put({'type': 'user_speech', 'data': full_transcript})
            self.transcript_collector.reset()
            self.speech_started = False
        self.utterance_end_timer = None


    async def stop_listening(self):
        if self.dg_connection:
            # Set is_listening to false early in the stop process
            self.is_listening = False 
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
        self.transcript_collector.reset()
        self.dg_connection = None
        self.microphone = None

class ConversationManager:
    def __init__(self, llm, tts):
        self.llm = llm
        self.tts = tts
        self.transcription_queue = asyncio.Queue() 
        self.stt = DeepgramSTT(self.transcription_queue)
        self.is_running = True
        self.current_llm_task = None
        self.user_spoke_again = asyncio.Event()
        self.is_speaking = False  # Flag to track if system is speaking
        self.speaking_lock = asyncio.Lock()  # Lock to prevent overlapping speech

    async def process_llm_and_speak(self, text_input):
        print(f"LLM processing: '{text_input}'")
        # Reset the event before starting LLM/TTS
        self.user_spoke_again.clear()
        
        try:
            llm_response_text, llm_time = await self.llm.generate_response(text_input)
            print(f"LLM ({llm_time}ms): {llm_response_text}")

            if self.user_spoke_again.is_set():
                print("User spoke again during LLM response generation. Not speaking this response.")
                return

            if llm_response_text:
                # Use lock to prevent overlapping speech
                async with self.speaking_lock:
                    # Disable speech processing before speaking
                    self.is_speaking = True
                    self.stt.speech_processing_enabled = False
                    print("Speech processing disabled during TTS output...")
                    
                    try:
                        # Speak the response
                        await self.tts.speak(llm_response_text, self.user_spoke_again)
                        
                        # Add a delay after speech before resuming speech processing
                        # This prevents any audio echo/feedback from being processed
                        if not self.user_spoke_again.is_set():
                            try:
                                # Increased delay to ensure we don't hear echo
                                await asyncio.sleep(1.5)  # 1.5 second delay to avoid echo detection
                            except asyncio.CancelledError:
                                # Task was cancelled during sleep - that's fine
                                return
                    except Exception as e:
                        print(f"Error during TTS: {e}")
                    finally:
                        # Even if there's an error, reset these flags
                        self.is_speaking = False
                        self.stt.speech_processing_enabled = True
                        print("Speech processing resumed.")
            else:
                print("LLM returned no response.")
            
        except asyncio.CancelledError:
            print("LLM/TTS processing was cancelled")
            raise
        except Exception as e:
            print(f"Error in process_llm_and_speak: {e}")
            # Still reset flags on error
            self.is_speaking = False
            self.stt.speech_processing_enabled = True

    async def main_loop(self):
        await self.stt.start_listening()
        try:
            while self.is_running:
                try:
                    # Use a short timeout to allow for regular checks
                    queue_item = await asyncio.wait_for(self.transcription_queue.get(), timeout=0.1)
                    
                    user_speech_data = None
                    if isinstance(queue_item, dict) and queue_item.get('type') == 'user_speech':
                        user_speech_data = queue_item.get('data')
                    else:
                        # Handle other types of queue items or log unexpected items
                        print(f"Unexpected item in queue: {queue_item}")
                        continue

                    # Only process if we're not in the middle of speaking
                    # or if the user explicitly interrupted
                    if self.is_speaking:
                        print("User spoke while system was speaking.")
                        self.user_spoke_again.set()  # Signal that new user input has arrived
                    
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
                        if user_speech_data:
                            transcript = user_speech_data # This is the actual string now
                            if "that's enough" in transcript.lower() or \
                            "stop listening" in transcript.lower() or \
                            "goodbye" in transcript.lower():
                                print("Exit phrase detected. Shutting down.")
                                if not self.user_spoke_again.is_set():  # Don't speak if immediately interrupted
                                    try:
                                        await self.tts.speak("Okay, goodbye!", self.user_spoke_again)
                                    except Exception as e:
                                        print(f"Error during goodbye message: {e}")
                                self.is_running = False
                                break
                            
                            self.user_spoke_again.clear()  # Clear before starting new task
                            try:
                                self.current_llm_task = asyncio.create_task(self.process_llm_and_speak(transcript))
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
                    if self.is_running and not self.user_spoke_again.is_set():  # Only print Listening if not shutting down and not just interrupted
                         print("Listening...")

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Shutting down...")
            self.is_running = False
        except Exception as e:
            print(f"Critical error in main loop: {e}")
        finally:
            print("Cleaning up...")
            self.user_spoke_again.set()  # Signal to any ongoing tasks to stop
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
    print("\nShutdown signal received, cleaning up...")
    manager.is_running = False
    # The manager's main loop will handle the rest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Voice Assistant")
    parser.add_argument("--first_message", type=str, default=None, help="A message for the agent to speak at the beginning.")
    parser.add_argument("--text_only", action="store_true", help="Run in text-only mode (no TTS).")
    # early_processing arg is removed as the new structure inherently aims for responsiveness
    args = parser.parse_args() 

    try:
        asyncio.run(main_async(args.first_message, args.text_only))
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
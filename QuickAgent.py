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
            self.system_prompt = "You are a helpful assistant."
            print("Warning: system_prompt.txt not found. Using default system prompt.")
        
        # Initialize conversation history
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def process(self, text):
        # Add user message to history
        self.messages.append({"role": "user", "content": text})
        
        start_time = time.time()
        
        try:
            # Call OpenAI-compatible API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages
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
                messages=self.messages
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

    async def start_listening(self):
        # The configuration was moved to the client initialization in __init__
        self.dg_connection = self.client.listen.asynclive.v("1")

        # Fix: Define async event handlers properly
        async def on_open_async(connection, **kwargs):
            print("Deepgram connection opened.")
            
        async def on_message_async(connection, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if result.is_final and result.speech_final:
                self.accumulated_transcript = result.channel.alternatives[0].transcript
                pass
            elif not result.is_final and len(sentence) > 0: # Interim result
                current_transcript_interim = result.channel.alternatives[0].transcript
                print(f"Hearing: {current_transcript_interim}...", end='\r', flush=True)
                self.accumulated_transcript = current_transcript_interim
        
        async def on_utterance_end_async(connection, utterance_end, **kwargs):
            print("\nUser likely finished speaking (UtteranceEnd).")
            self.is_speaking = False # User has stopped.
            if len(self.accumulated_transcript.strip()) > 0:
                await self.send_to_manager_async()
            else:
                print("UtteranceEnd received, but no transcript accumulated to send.")
        
        async def on_speech_started_async(connection, speech_started, **kwargs):
            print("User started speaking.")
            self.is_speaking = True
            
        async def on_error_async(connection, error, **kwargs):
            print(f"Deepgram error: {error}")
            
        async def on_close_async(connection, **kwargs):
            print("Deepgram connection closed.")
            if self.microphone and self.microphone.is_alive():
                self.microphone.finish()

        # Register the async handlers
        self.dg_connection.on(LiveTranscriptionEvents.Open, on_open_async)
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message_async)
        self.dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end_async)
        self.dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started_async)
        self.dg_connection.on(LiveTranscriptionEvents.Error, on_error_async)
        self.dg_connection.on(LiveTranscriptionEvents.Close, on_close_async)

        # More aggressive options for faster interaction
        options = LiveOptions(
            model="nova-2", # or nova-3 if available and preferred
            language="en-US",
            smart_format=True,
            encoding="linear16", # Make sure this matches your microphone
            channels=1,
            sample_rate=16000,   # Make sure this matches your microphone
            interim_results=True,
            utterance_end_ms=1000, # Changed to integer
            vad_events=True,
            endpointing=300, # Changed to integer
        )

        print("Listening...")
        # The start method in AsyncLive version does need an await
        try:
            await self.dg_connection.start(options)
            
            # Create and start microphone after successful connection
            try:
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
            if self.microphone and self.microphone.is_alive():
                self.microphone.finish() 
            # According to Deepgram Python SDK, finish() is the correct way to close.
            # It handles closing the websocket and stopping the listener.
            await self.dg_connection.finish() 
            print("Stopped Deepgram STT.")
        self.accumulated_transcript = ""

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

    async def process_llm_and_speak(self, text_input):
        print(f"LLM processing: '{text_input}'")
        # Reset the event before starting LLM/TTS
        self.user_spoke_again.clear()
        
        llm_response_text, llm_time = await self.llm.generate_response(text_input)
        print(f"LLM ({llm_time}ms): {llm_response_text}")

        if self.user_spoke_again.is_set():
            print("User spoke again during LLM response generation. Not speaking this response.")
            return

        if llm_response_text:
            # Disable speech processing before speaking
            self.is_speaking = True
            self.stt.processing_enabled = False
            print("Speech processing disabled during TTS output...")
            
            # Speak the response
            await self.tts.speak(llm_response_text, self.user_spoke_again)
            
            # Add a small delay after speech before resuming speech processing
            # This prevents any audio echo/feedback from being processed
            if not self.user_spoke_again.is_set():
                await asyncio.sleep(1.0)  # 1 second delay to avoid echo detection
            
            # Reset flags
            self.is_speaking = False
            self.stt.processing_enabled = True
            print("Speech processing resumed.")
        else:
            print("LLM returned no response.")
        
        if not self.user_spoke_again.is_set() and self.is_running:
            print("Audio playback complete")

    async def main_loop(self):
        await self.stt.start_listening()
        try:
            while self.is_running:
                try:
                    # Use a short timeout to allow for regular checks
                    user_input = await asyncio.wait_for(self.transcription_queue.get(), timeout=0.1)
                    
                    self.user_spoke_again.set() # Signal that new user input has arrived

                    if self.current_llm_task and not self.current_llm_task.done():
                        print("User spoke while LLM was processing/speaking. Cancelling previous task.")
                        self.current_llm_task.cancel()
                        await self.tts.stop_playback() # Ensure TTS stops

                    if user_input:
                        if "that's enough" in user_input.lower() or \
                           "stop listening" in user_input.lower() or \
                           "goodbye" in user_input.lower():
                            print("Exit phrase detected. Shutting down.")
                            if not self.user_spoke_again.is_set(): # Don't speak if immediately interrupted
                                await self.tts.speak("Okay, goodbye!", self.user_spoke_again)
                            self.is_running = False
                            break
                        
                        self.user_spoke_again.clear() # Clear before starting new task
                        self.current_llm_task = asyncio.create_task(self.process_llm_and_speak(user_input))

                except asyncio.TimeoutError:
                    pass # No new transcript, continue listening
                except asyncio.CancelledError:
                    print("LLM task was cancelled due to new input or shutdown.")
                
                if self.current_llm_task and self.current_llm_task.done():
                    try:
                        await self.current_llm_task # Propagate exceptions
                    except asyncio.CancelledError:
                        print("LLM task finished (was cancelled).")
                    except Exception as e:
                        print(f"LLM task finished with error: {e}")
                    self.current_llm_task = None
                    if self.is_running and not self.user_spoke_again.is_set(): # Only print Listening if not shutting down and not just interrupted
                         print("Listening...")


        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Shutting down...")
            self.is_running = False
        finally:
            print("Cleaning up...")
            self.user_spoke_again.set() # Signal to any ongoing tasks to stop
            if self.current_llm_task and not self.current_llm_task.done():
                self.current_llm_task.cancel()
                # Wait for cancellation to complete
                try:
                    await self.current_llm_task
                except asyncio.CancelledError:
                    pass # Expected
            await self.stt.stop_listening()
            await self.tts.stop_playback() # Ensure TTS is stopped on cleanup
            # if self.llm and hasattr(self.llm, 'close'): # If your LLM client needs closing
            #     await self.llm.close() # This depends on your LLM client library
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
            
            if chunk_count == 0 and audio_received_time is None: # No audio data received
                print(f"TTS: No audio data received from Deepgram for: {text}")

            if self.player_process and self.player_process.stdin:
                try:
                    self.player_process.stdin.close()
                except Exception as e:
                    print(f"TTS: Error closing player stdin: {e}")
            
            if self.player_process:
                while self.player_process.poll() is None:
                    if interrupt_event and interrupt_event.is_set():
                        print("TTS: Interrupt received while waiting for player. Terminating.")
                        await self.stop_playback()
                        return
                    await asyncio.sleep(0.05) # Non-blocking wait
                
                rc = self.player_process.returncode
                if rc != 0:
                    print(f"TTS: Player process exited with code {rc}")
                self.player_process = None

        except requests.exceptions.RequestException as e:
            print(f"TTS Request failed: {e}")
        except Exception as e:
            print(f"TTS Error: {e}")
            await self.stop_playback() 
        finally:
            if self.player_process and self.player_process.poll() is None:
                print("TTS: Forcing player stop in finally block.")
                await self.stop_playback()

    async def stop_playback(self):
        if self.player_process and self.player_process.poll() is None: 
            print("Stopping TTS playback...")
            try:
                if self.player_process.stdin:
                    self.player_process.stdin.close()
            except Exception as e:
                print(f"TTS stop: Error closing stdin: {e}")

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
                 print(f"TTS stop: Error during player termination: {e}")
            self.player_process = None
            print("TTS playback stopped.")
        elif self.player_process and self.player_process.poll() is not None:
            self.player_process = None 

async def main_async(first_message=None, text_only=False):
    # Initialize your LLM and TTS with real implementations
    llm = LanguageModelProcessor() 
    
    # Use Rime TTS instead of Deepgram TTS
    from realism_voice.io.rime_tts_async import RimeTTS
    tts = RimeTTS()
    
    # If no first message provided, use the greeting
    if not first_message:
        first_message = "Hey how you doing my dawg!!"

    manager = ConversationManager(llm, tts)
    
    if first_message:
        print(f"Agent: {first_message}")
        if not text_only:
            # Create a dummy event for the first message if needed, or handle None
            await tts.speak(first_message, interrupt_event=asyncio.Event()) 

    await manager.main_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Voice Assistant")
    parser.add_argument("--first_message", type=str, default=None, help="A message for the agent to speak at the beginning.")
    parser.add_argument("--text_only", action="store_true", help="Run in text-only mode (no TTS).")
    # early_processing arg is removed as the new structure inherently aims for responsiveness
    args = parser.parse_args() 

    try:
        asyncio.run(main_async(args.first_message, args.text_only))
    except KeyboardInterrupt:
        print("Application terminated by user.")
    except Exception as e:
        # This top-level exception handler is good for catching unexpected issues
        print(f"An unhandled critical exception occurred: {e}")
        import traceback
        traceback.print_exc() 
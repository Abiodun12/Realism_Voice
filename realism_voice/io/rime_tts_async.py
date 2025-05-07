import os
import asyncio
import requests
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────────
RIME_TTS_URL = os.getenv("RIME_API_BASE", "https://users.rime.ai/v1/rime-tts")
RIME_API_KEY = os.getenv("RIME_API_KEY")
DEFAULT_SPK = os.getenv("RIME_SPEAKER", "orion")

class RimeTTS:
    def __init__(self, api_key=None):
        self.api_key = api_key or RIME_API_KEY
        self.player_process = None
        self.play_command = ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", "-"]
        # For macOS: self.play_command = ["afplay", "-"] 
        # For Linux: self.play_command = ["aplay", "-f", "S16_LE", "-r", "24000", "-"]

    async def speak(self, text, interrupt_event=None, speaker=DEFAULT_SPK):
        """
        Streams MP3 bytes from Rime Arcana and plays them chunk-by-chunk.
        
        Args:
            text: The text to be spoken
            interrupt_event: An asyncio Event that can be set to interrupt playback
            speaker: The speaker voice to use
        """
        if not text:
            print("TTS: No text to speak.")
            return

        headers = {
            "Accept": "audio/mp3",  # trigger streaming MP3
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "speaker": speaker,
            "text": text,
            "modelId": "arcana",  # Arcana for lifelike voice
            "audioFormat": "mp3",  # ensure MP3 payload
            "reduceLatency": True,  # shave off extra processing
            "samplingRate": 22050,  # or lower for telephony
            "repetition_penalty": 1.2,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1200,
        }

        print(f"Converting to speech: {text}")
        start_time = asyncio.get_event_loop().time()

        try:
            # Using asyncio.to_thread for the blocking requests call
            response = await asyncio.to_thread(
                requests.post, 
                RIME_TTS_URL, 
                headers=headers, 
                json=payload, 
                stream=True
            )
            
            response.raise_for_status()

            if self.player_process and self.player_process.poll() is None:
                print("Warning: Previous player process still running. Terminating.")
                await self.stop_playback()

            self.player_process = await asyncio.create_subprocess_exec(
                *self.play_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            audio_received_time = None
            chunk_count = 0
            
            for chunk in response.iter_content(chunk_size=4096):
                if interrupt_event and interrupt_event.is_set():
                    print("TTS: Interrupt received, stopping playback.")
                    response.close()
                    await self.stop_playback()
                    return

                if chunk:
                    if audio_received_time is None:  # First chunk of audio
                        audio_received_time = asyncio.get_event_loop().time()
                        actual_ttfb = int((audio_received_time - start_time) * 1000)
                        print(f"TTS Time to First Audio Byte: {actual_ttfb}ms")
                    
                    try:
                        if self.player_process and not self.player_process.stdin.is_closing():
                            self.player_process.stdin.write(chunk)
                            await self.player_process.stdin.drain()
                        else:  # Player closed prematurely
                            print("TTS: Player process stdin is not available. Stopping.")
                            response.close()
                            break
                    except (BrokenPipeError, ConnectionResetError):
                        print("TTS: Playback pipe broken. Player likely closed or interrupted.")
                        response.close()
                        break
                    except Exception as e:  # Catch other potential stdin write errors
                        print(f"TTS: Error writing to player stdin: {e}")
                        response.close()
                        break
                    
                    # Check for interruption frequently
                    if interrupt_event and interrupt_event.is_set():
                        print("TTS: Interrupt received while processing chunk. Stopping.")
                        response.close()
                        await self.stop_playback()
                        return
                
                chunk_count += 1
            
            if chunk_count == 0 and audio_received_time is None:  # No audio data received
                print(f"TTS: No audio data received from Rime for: {text}")

            if self.player_process and not self.player_process.stdin.is_closing():
                try:
                    self.player_process.stdin.close()
                except Exception as e:
                    print(f"TTS: Error closing player stdin: {e}")
            
            if self.player_process:
                try:
                    await asyncio.wait_for(self.player_process.wait(), 10.0)  # Wait up to 10 seconds
                    print("Audio playback complete")
                except asyncio.TimeoutError:
                    print("TTS: Player process taking too long to complete. Forcing termination.")
                    await self.stop_playback()
                
                self.player_process = None

        except requests.exceptions.RequestException as e:
            print(f"TTS Request failed: {e}")
        except Exception as e:
            print(f"TTS Error: {e}")
            await self.stop_playback()
        finally:
            if self.player_process and self.player_process.returncode is None:
                print("TTS: Forcing player stop in finally block.")
                await self.stop_playback()

    async def stop_playback(self):
        """Stop any ongoing playback."""
        if self.player_process and self.player_process.returncode is None:
            print("Stopping TTS playback...")
            try:
                if not self.player_process.stdin.is_closing():
                    self.player_process.stdin.close()
            except Exception as e:
                print(f"TTS stop: Error closing stdin: {e}")

            try:
                self.player_process.terminate()
                await asyncio.wait_for(self.player_process.wait(), 1.0)
            except asyncio.TimeoutError:
                print("Player did not terminate gracefully, killing.")
                self.player_process.kill()
                try:
                    await asyncio.wait_for(self.player_process.wait(), 1.0)
                except asyncio.TimeoutError:
                    print("Player did not die even after kill. Giving up.")
            except Exception as e:
                print(f"TTS stop: Error during player termination: {e}")
            
            self.player_process = None
            print("TTS playback stopped.")
        elif self.player_process and self.player_process.returncode is not None:
            self.player_process = None 
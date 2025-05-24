import os
import asyncio
import requests
import json
from pathlib import Path
from .rime_tts_async import RimeTTS

class StreamingRimeTTS(RimeTTS):
    def __init__(self, api_key=None, websocket_clients=None):
        """
        Enhanced RimeTTS that can stream audio to WebSocket clients.
        
        Args:
            api_key: Rime API key
            websocket_clients: Dictionary of active WebSocket clients to stream audio to
        """
        super().__init__(api_key)
        self.websocket_clients = websocket_clients or {}
    
    async def speak(self, text, interrupt_event=None, speaker=None, client_id=None):
        """
        Streams MP3 bytes from Rime Arcana and either:
        1. Plays them locally (if running locally)
        2. Streams them to connected WebSocket clients (if running on Render)
        
        Args:
            text: The text to be spoken
            interrupt_event: An asyncio Event that can be set to interrupt playback
            speaker: The speaker voice to use
            client_id: Specific client ID to send audio to (if None, send to all)
        """
        if not text:
            print("TTS: No text to speak.")
            return

        # Use default speaker if not specified
        speaker = speaker or os.getenv("RIME_SPEAKER", "orion")

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
        
        # Check if we're streaming to clients
        has_clients = bool(self.websocket_clients)
        local_playback = not has_clients or os.getenv("FORCE_LOCAL_PLAYBACK", "0") == "1"

        try:
            # Using asyncio.to_thread for the blocking requests call
            response = await asyncio.to_thread(
                requests.post, 
                os.getenv("RIME_API_BASE", "https://users.rime.ai/v1/rime-tts"), 
                headers=headers, 
                json=payload, 
                stream=True
            )
            
            response.raise_for_status()

            # If doing local playback, set up the player process
            if local_playback:
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
            first_chunk = True
            
            for chunk in response.iter_content(chunk_size=4096):
                if interrupt_event and interrupt_event.is_set():
                    print("TTS: Interrupt received, stopping playback.")
                    response.close()
                    if local_playback:
                        await self.stop_playback()
                    return

                if chunk:
                    if audio_received_time is None:  # First chunk of audio
                        audio_received_time = asyncio.get_event_loop().time()
                        actual_ttfb = int((audio_received_time - start_time) * 1000)
                        print(f"TTS Time to First Audio Byte: {actual_ttfb}ms")
                    
                    # Stream to WebSocket clients if available
                    if has_clients:
                        clients_to_stream = []
                        if client_id and client_id in self.websocket_clients:
                            clients_to_stream = [self.websocket_clients[client_id]]
                        else:
                            clients_to_stream = list(self.websocket_clients.values())
                        
                        # Send audio data to clients
                        for ws in clients_to_stream:
                            try:
                                # First chunk includes metadata
                                if first_chunk:
                                    # Send a JSON message first to indicate audio is coming
                                    await ws.send_json({
                                        "type": "tts_audio_start",
                                        "format": "mp3",
                                        "text": text
                                    })
                                
                                # Send the audio data as binary
                                await ws.send_bytes(chunk)
                            except Exception as e:
                                print(f"Error sending audio to client: {e}")
                        
                        first_chunk = False
                    
                    # Also play locally if needed
                    if local_playback:
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
                        if local_playback:
                            await self.stop_playback()
                        return
                
                chunk_count += 1
            
            if chunk_count == 0 and audio_received_time is None:  # No audio data received
                print(f"TTS: No audio data received from Rime for: {text}")

            # Send end marker to clients
            if has_clients:
                for ws in self.websocket_clients.values():
                    try:
                        await ws.send_json({"type": "tts_audio_end"})
                    except Exception as e:
                        print(f"Error sending audio end marker: {e}")

            # Clean up local playback if active
            if local_playback and self.player_process:
                if not self.player_process.stdin.is_closing():
                    try:
                        self.player_process.stdin.close()
                    except Exception as e:
                        print(f"TTS: Error closing player stdin: {e}")
                
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
            if local_playback:
                await self.stop_playback()
        finally:
            if local_playback and self.player_process and self.player_process.returncode is None:
                print("TTS: Forcing player stop in finally block.")
                await self.stop_playback()
            
            # Final end marker for clients in case of error
            if has_clients:
                for ws in self.websocket_clients.values():
                    try:
                        await ws.send_json({"type": "tts_audio_end"})
                    except:
                        pass

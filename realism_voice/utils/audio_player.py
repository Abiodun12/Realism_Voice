import sounddevice as sd
import soundfile as sf
from pathlib import Path

def play_audio_file(file_path: Path):
    """
    Play an audio file using sounddevice and soundfile.
    
    Args:
        file_path: Path to the audio file
    """
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()  # Wait until the audio is finished playing 
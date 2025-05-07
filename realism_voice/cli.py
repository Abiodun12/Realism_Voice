from realism_voice.io.tts_rime import rime_tts_stream
from realism_voice.utils.audio_player import play_audio_file

def run_realism_conversation():
    """
    Run the main conversation loop.
    This function will be implemented later.
    """
    pass

def main():
    # Fire off the streaming greeting
    greeting = rime_tts_stream("Hey how you doing my dawg!!")
    play_audio_file(greeting)

    # Then enter the STT ↔ LLM ↔ TTS loop
    run_realism_conversation()

if __name__ == "__main__":
    main() 
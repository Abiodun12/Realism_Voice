# Real-Time Voice Assistant

A conversational voice assistant with natural speech interaction capabilities, powered by Deepgram's speech technologies and Alibaba Cloud's Qwen language model.

## Features

- **Real-time Speech Recognition**: Uses Deepgram's Nova-2/Nova-3 models for accurate and fast speech-to-text.
- **Natural Language Understanding**: Processes queries with Alibaba Cloud's Qwen language model.
- **Text-to-Speech**: Converts responses to natural-sounding speech using Deepgram's Aura voice.
- **Interruption Handling**: Allows users to interrupt the assistant while it's speaking.
- **Self-speech Filtering**: Prevents the assistant from hearing and responding to its own voice output.

## How It Works

The system combines several components to create a natural conversation flow:

1. **Speech Input**: Captures audio from your microphone in real-time.
2. **Speech Recognition**: Transcribes speech to text using Deepgram's streaming API.
3. **Language Processing**: Sends transcribed text to Alibaba Cloud's Qwen LLM for understanding and response generation.
4. **Speech Synthesis**: Converts the LLM's text response to speech using Deepgram's TTS API.
5. **Conversation Management**: Handles the flow between these components, including interruptions and speech filtering.

## Requirements

- Python 3.10+
- FFmpeg installed for audio playback
- Deepgram API Key
- Alibaba Cloud DashScope API Key

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd Realism_voice_
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   DEEPGRAM_API_KEY=your_deepgram_api_key
   DASHSCOPE_API_KEY=your_alibaba_dashscope_api_key
   DASHSCOPE_MODEL_NAME=qwen-plus  # Or another available model
   ```

4. (Optional) Create a `system_prompt.txt` file to customize the LLM's behavior.

## Usage

Run the voice assistant with:

```
python QuickAgent.py
```

To start with a welcome message:

```
python QuickAgent.py --first_message "Hello! I'm your voice assistant. How can I help you today?"
```

For text-only mode (no speech output):

```
python QuickAgent.py --text_only
```

### Conversation Flow

1. Wait for the "Listening..." prompt.
2. Speak naturally - you'll see interim transcripts as you speak.
3. When you pause, the system will process your speech and respond.
4. You can interrupt the assistant at any time by speaking again.
5. Say "goodbye", "that's enough", or "stop listening" to exit.

## Technical Details

### Key Components

- **DeepgramSTT**: Handles speech-to-text processing with real-time transcription.
- **LanguageModelProcessor**: Manages the connection to the Alibaba Cloud Qwen LLM.
- **DeepgramTTS**: Converts text responses to speech.
- **ConversationManager**: Orchestrates the conversation flow between components.

### Advanced Features

- **Voice Activity Detection**: Uses Deepgram's VAD to detect when the user has finished speaking.
- **Endpointing**: Automatically segments speech for better processing.
- **Interim Results**: Shows what the system is hearing in real-time for feedback.
- **Error Handling**: Robust error handling for network issues and other failures.
- **Self-speech Filtering**: Temporarily disables STT processing while TTS is active to prevent feedback loops.

## Troubleshooting

- **Audio Playback Issues**: Ensure FFmpeg is installed and accessible in your PATH.
- **Microphone Access**: Make sure your system has permission to access the microphone.
- **API Key Errors**: Verify that your API keys are correctly set in the `.env` file.
- **Speech Recognition Problems**: Check your microphone settings and ensure you're in a quiet environment.

## License

[Add your license information here]

## Acknowledgements

- [Deepgram](https://deepgram.com/) for speech recognition and synthesis APIs
- [Alibaba Cloud](https://www.alibabacloud.com/) for the Qwen language model 
# Realism Voice Assistant

A real-time voice assistant that combines state-of-the-art speech-to-text, language understanding, and text-to-speech technologies for natural conversations.

## Features

- **Speech Recognition**: Uses Deepgram's Nova-2/3 model for high-quality, low-latency speech recognition
- **Natural Language Understanding**: Powered by Alibaba Cloud's Qwen model for intelligent responses
- **Ultra-Realistic Voice Synthesis**: Integrates Rime's Arcana TTS for lifelike voice output
- **Responsive Interaction**: Handles interruptions and provides real-time feedback during conversations
- **Smart Echo Cancellation**: Automatically prevents the assistant from responding to its own speech
- **Enhanced Speech Completion**: Improved handling of TTS speech to prevent self-interruptions
- **Resilient Error Handling**: Graceful fallback mechanisms for all components

## Components

- **Speech-to-Text (STT)**: Deepgram for fast, accurate speech recognition
- **Language Model (LLM)**: Alibaba Cloud Qwen via DashScope API
- **Text-to-Speech (TTS)**: Rime Arcana for ultra-realistic voice synthesis

## Setup

### Prerequisites

- Python 3.10+
- FFmpeg (for audio playback)
- API keys for Deepgram, DashScope, and Rime

### Installation

1. Clone the repository:
```
git clone <repository-url>
cd realism_voice
```

2. Set up a virtual environment:
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
# API Keys
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DEEPGRAM_API_KEY=your_deepgram_api_key_here
RIME_API_KEY=your_rime_api_key_here

# Optional configurations
DASHSCOPE_MODEL_NAME=qwen-plus
RIME_SPEAKER=orion
RIME_API_BASE=https://users.rime.ai/v1/rime-tts
```

## Usage

Run the assistant:

```
python QuickAgent.py
```

Optional arguments:
- `--first_message "Your custom greeting"` - Set a custom greeting
- `--text_only` - Run without TTS output

## Voice Options

The assistant uses Rime's Arcana TTS, which offers a variety of realistic voices:

- **orion** (default): A natural male voice
- Other voices available through Rime's voice catalog

To change the voice, set the `RIME_SPEAKER` environment variable in your `.env` file.

## Architecture

- `QuickAgent.py`: Main application
- `realism_voice/io/`: Input/output modules (TTS, STT)
- `realism_voice/utils/`: Utility functions

### Key Components

1. **Conversation Manager**: Orchestrates the conversation flow
2. **DeepgramSTT**: Handles speech recognition
3. **RimeTTS**: Generates ultra-realistic speech output
4. **LanguageModelProcessor**: Processes natural language understanding

### Detailed System Architecture

#### Core Components and Data Flow

1. **Main Application (`QuickAgent.py`)**
   - Initializes all components and starts the conversation loop
   - Handles command-line arguments and environment variables
   - Sets up global exception handling

2. **Conversation Manager (`ConversationManager` class)**
   - Coordinates data flow between STT, LLM, and TTS
   - Manages conversation state and user interruptions
   - Uses asyncio to handle concurrent operations
   - Main methods:
     - `main_loop()`: Primary event loop
     - `process_llm_and_speak()`: Processes user input and generates responses

3. **Speech Recognition (`DeepgramSTT` class)**
   - Uses Deepgram's WebSocket API for real-time transcription
   - Manages microphone input and speech processing
   - Detects speech start/end events for responsive interaction
   - Key features:
     - Interim results for real-time feedback
     - Utterance end detection for natural conversation flow
     - Processing toggle during TTS to prevent self-feedback

4. **Language Understanding (`LanguageModelProcessor` class)**
   - Connects to Alibaba Cloud's DashScope API using OpenAI-compatible interface
   - Maintains conversation history for context
   - Handles prompt construction and response parsing
   - Configurable through system prompt and model settings

5. **Voice Synthesis (`RimeTTS` class in `realism_voice/io/rime_tts_async.py`)**
   - Connects to Rime's Arcana API for ultra-realistic voice synthesis
   - Streams audio chunks for low-latency playback
   - Handles playback interruptions and cleanup
   - Voice customization through speaker parameter

#### System Communication

```
┌───────────────┐     Transcribed Text     ┌───────────────┐
│               │───────────────────────>  │               │
│  DeepgramSTT  │                          │ LanguageModel │
│    (Input)    │                          │   Processor   │
│               │  <───────────────────────│               │
└───────────────┘      Response Text       └───────────────┘
        │                                           │
        │                                           │
        │                                           │
        │                                           ▼
        │                                  ┌───────────────┐
        │                                  │               │
        │  Speech Processing Toggle        │    RimeTTS    │
        └─────────────────────────────────>│   (Output)    │
                                           │               │
                                           └───────────────┘
```

### Key Implementation Details

1. **Asynchronous Architecture**
   - The system uses asyncio throughout for non-blocking operations
   - Allows handling multiple streams (STT, LLM, TTS) concurrently
   - Enables interruption handling during any processing phase

2. **Interruption Handling**
   - Uses asyncio.Event objects to signal interruptions
   - Task cancellation pattern to stop in-progress operations
   - Audio stream cleanup to prevent resource leaks
   - Speaking lock mechanism to prevent overlapping speech
   - Improved control flow to allow LLM to finish speaking sentences

3. **Echo Cancellation Strategy**
   - STT processing disabled during TTS output
   - Extended delay (1.5 seconds) after TTS completes before re-enabling STT
   - Processing flags to track system speaking state
   - Enhanced utterance detection to reduce false positives

4. **Error Handling and Resilience**
   - Robust exception handling at multiple levels
   - Auto-reconnection for transient network issues
   - Graceful degradation on API failures
   - TTS service fallback mechanism (switches from Rime to Deepgram if needed)
   - More comprehensive BrokenPipe error handling
   - Proper cleanup of resources in all error scenarios

5. **Speech Detection Tuning**
   - Optimized Deepgram parameters for more natural conversation
   - Extended utterance thresholds (1500ms) to prevent premature speech endings
   - Increased endpointing delay (500ms) for improved sentence detection
   - Better handling of interim results to reduce processing of incomplete phrases

## Extending the System

### Adding New Voices

1. To use a different Rime voice:
   - Update the `RIME_SPEAKER` value in your `.env` file
   - Or modify `DEFAULT_SPK` in `realism_voice/io/rime_tts_async.py`
   - Alternatively, pass the desired speaker name to the `speak()` method directly

2. To add support for a different TTS provider:
   - Create a new TTS class implementing the same interface as `RimeTTS`
   - Ensure it provides `speak()` and `stop_playback()` async methods
   - Update the TTS initialization in `main_async()` function

### Customizing Language Model Behavior

1. Modifying system prompts:
   - Create or edit `system_prompt.txt` in the project root
   - The system will automatically load this file to configure the LLM
   - To modify the AB Uncle persona, edit this file while maintaining the key instructions (brevity, AAVE style, etc.)
   - Example alternative: "You are Coach T, a no-nonsense sports trainer who speaks in short, motivational phrases."

2. Changing LLM providers:
   - Update the `LanguageModelProcessor` class to use a different LLM provider
   - Ensure it provides the same interface as the current implementation
   - Update the LLM initialization in `main_async()` function

### Adding New Features

1. Wake word detection:
   - Could be implemented in `DeepgramSTT` class to only process after a trigger phrase
   - Add activation state to prevent processing until wake word detected

2. Multi-turn conversation improvements:
   - Enhance the `LanguageModelProcessor` with better context management
   - Add conversation summarization to maintain longer histories

3. Multi-modal capabilities:
   - Integrate with vision APIs to add image understanding
   - Extend the `ConversationManager` to handle different input types

## Troubleshooting

- **Audio issues**: Make sure FFmpeg is installed and your microphone is working
- **API errors**: Check your API keys in the `.env` file
- **Voice not working**: Verify the `RIME_SPEAKER` value and internet connection
- **TTS Failures**: If Rime TTS fails, the system will automatically fall back to Deepgram TTS
- **Self-interruptions**: If the assistant interrupts itself, try increasing the endpointing and utterance_end_ms values
- **Microphone errors**: Ensure your microphone is properly connected and not being used by another application
- **Persona issues**: If responses don't match the expected AB Uncle style, check `system_prompt.txt` and ensure it's being loaded correctly

## Voice Persona

The assistant now uses the "AB Uncle" persona - a fun, quick-witted African-American uncle who speaks in AAVE style with concise responses.

### AB Uncle Features:
- Fun, conversational AAVE speech style
- Greets with "Hi, how you doin', dawg?"
- Includes light humor (1-2 jokes max)
- Keeps responses concise (limited to ~60 tokens/120 characters)
- Uses 3 short sentences maximum
- Occasionally adds a wink emoji 😉 when appropriate

You can customize this persona by editing the `system_prompt.txt` file in the project root.

### Token Limits

The system now enforces a 60-token limit (approximately 120 characters) on responses to:
- Keep interactions brief and engaging
- Ensure complete sentences without abrupt cutoffs
- Optimize for voice output without lengthy monologues

This token limit can be adjusted in the `QuickAgent.py` file by modifying the `max_tokens` parameter in the LLM calls.

## License

[Your License]
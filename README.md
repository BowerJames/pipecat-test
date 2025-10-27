# Pipecat Local Voice Bot

A voice conversational bot built with the Pipecat framework that uses your computer's microphone and speakers - no external services required beyond OpenAI.

## Features

- **Local Voice Bot** (`local_voice_bot.py`): Voice interaction using your computer's mic/speakers
- **No Daily.co needed**: Uses local audio transport instead of external services
- **OpenAI Integration**: Uses OpenAI for both language model and text-to-speech
- **Environment Variables**: Automatically loads from `.env` file

## Quick Start

### 1. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

### 2. Set Up Environment Variables

Copy the example environment file:
```bash
cp env.example .env
```

Edit `.env` and add your API keys:
```bash
# Required: OpenAI API key for LLM and TTS
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom OpenAI TTS voice (alloy, echo, fable, onyx, nova, shimmer)
OPENAI_TTS_VOICE=alloy
```

**Note:** The bot automatically loads environment variables from the `.env` file using `python-dotenv`.

### 3. Run the Bot

```bash
uv run python local_voice_bot.py
```

## API Keys Required

### OpenAI API Key
- Get from: https://platform.openai.com/api-keys
- Required for: LLM responses and text-to-speech
- Used for: Language model responses and speech synthesis

## Usage

### Local Voice Bot (`local_voice_bot.py`)
- Only requires OpenAI API key
- Uses your computer's microphone and speakers
- No external services needed
- Speak naturally and get voice responses
- Press Ctrl+C to stop

## Development

### Adding New Features
- Modify the pipeline in `local_voice_bot.py`
- Add new services to the pipeline
- Update the system prompt for different behavior

### Dependencies
- `pipecat-ai[local]`: Core framework with local audio support
- `python-dotenv`: Environment variable loading
- `websockets`: WebSocket communication

### Project Structure
```
pipecat-test/
├── pyproject.toml      # Project configuration
├── local_voice_bot.py  # Local voice chat bot
├── env.example        # Environment variables template
└── README.md          # This file
```

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY environment variable is required"**
   - Make sure you've set the environment variable or created a `.env` file
   - Check that the API key is valid

2. **Audio not working**
   - Ensure your computer has a working microphone and speakers
   - Check that no other applications are using the audio devices
   - On macOS, you may need to grant microphone permissions

3. **Import errors**
   - Run `uv sync` to install dependencies
   - Make sure you're using Python 3.11+

4. **No voice responses**
   - Check that your OpenAI API key has TTS access
   - Verify the voice setting in your `.env` file
   - Ensure your speakers are working

### Getting Help

- Check the [Pipecat documentation](https://pipecat.ai/)
- Review the example environment file for required variables
- Ensure all API keys are valid and have proper permissions
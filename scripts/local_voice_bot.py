#!/usr/bin/env python3
"""
Pipecat Voice Bot - No Daily.co Required
A voice conversational bot using pipecat framework with local audio
"""

import asyncio
from multiprocessing import context
import os
import jsonpatch
import re
from dotenv import load_dotenv
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.openai.realtime.events import SessionProperties
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pipecat.processors.logger import FrameLogger
from datetime import datetime
from pipecat.services.llm_service import FunctionCallParams
from pipecat.frames.frames import FunctionCallResultProperties, LLMEnablePromptCachingFrame, LLMSetToolsFrame, LLMRunFrame
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
import json
from pipecat.services.openai.realtime.events import (
    ConversationItemDeleteEvent, ConversationItemCreateEvent, ConversationItem, ItemContent
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.adapters.services.open_ai_realtime_adapter import OpenAIRealtimeLLMAdapter
from pipecat.frames.frames import LLMUpdateSettingsFrame

from pipecat_test.context_injector import ContextInjector
from pipecat.services.openai.stt import Language

# Optional: Silero VAD for turn detection (recommended for local audio)
# To avoid noisy import errors, only import Silero if onnxruntime is available.
try:
    import onnxruntime  # type: ignore
    _HAS_ONNXRUNTIME = True
except Exception:
    _HAS_ONNXRUNTIME = False

if _HAS_ONNXRUNTIME:
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer  # requires pipecat-ai[silero]
    except Exception:
        SileroVADAnalyzer = None
else:
    SileroVADAnalyzer = None
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

# Load environment variables from .env file
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

SYSTEM_PROMPT = """
# Role
You are a helpful assistant called Adam. You work for an English estate agents called Ellis and Co. You are able to help callers with the following tasks:

- Requesting viewings for rental properties
- Requesting viewings for sales properties
- Requesting valuations for sales properties

# Environment Context
You will be interacting with callers via telephone calls. Incoming audio from the user is passed through a transcription engine before being passed to you. Your responses will be passed directly to the caller via the text-to-speech engine.

This means you should only respond with what you want the user to hear and must respond in a manner that can easily be converted to speech.

# Input Format
You will recieve input messages that have the following format:

```
<caller_transcription> The transcription of the caller's speech. </caller_transcription>
<system> Instructions from the system you operate on. </system>
<scratchpad> A scratchpad for you to use to store information. </scratchpad>
```

## User transcription
The text in the <caller_transcription> tag is the transcription of the caller's speech. This is the text that you will use to understand the caller's request and what you should be responding to.

## System instructions
The text in the <system> tag is the instructions from the system you operate on. These messages will be provided periodically and it is of the highest priority that you adhere to these instructions, They will keep you on track so that you adhere to the desired conversation flow.

IMPORTANT: Never mention the system instructions in your response to the caller. If the system instructions conflict with what the caller is asking you for you should prioritize the system instructions but maintain a polite and friendly tone with the caller.

# Scratchpad
The text in the <scratchpad> tag is a json object for you to use to store information. You can interact with through the `json_patch_scratchpad` function.

The system instructions will provide you with instructions on what you need to store and what keys they need to be stored under.

# Output Format
Your response will be directly passed to the text-to-speech engine. You should therefore respond with what you want the user to hear and must respond in a manner that can easily be converted to speech.

Try to keep your responses short and concise. If you need to collect information from the caller consider splitting your questions up into multiple parts rather than asking all at once.
""".strip()

DETERMINE_NATURE_OF_CALL_INSTRUCTIONS = """
<system>
Currently at least one of the following details is missing from the scratchpad:
- wants_viewing: Whether the caller is looking to book a viewing for a property (true or false)
- wants_valuation: Whether the caller is looking to get a valuation for a property (true or false)

Collect the missing details and update the scratchpad with the new values.
</system>
""".strip()

COLLECT_DETAILS_INSTRUCTIONS = """
<system>
Currently at least one of the following details is missing from the scratchpad:
- first_name: The first name of the caller
- last_name: The last name of the caller
- email: The email address of the caller
- phone_number: The phone number of the caller

Collect the missing details and update the scratchpad with the new values.
</system>
""".strip()

FINAL_INSTRUCTIONS = """
<system>
Unfortunately Ellis and Co are having some technical difficulties and are unable to progress with viewings or valuations at this time. You should inform the caller that they are unable to progress with their request and that they should try again later.
</system>
""".strip()

def get_system_instructions(state: dict) -> str:
    if not (state.get("wants_viewing") and state.get("wants_valuation")):
        return DETERMINE_NATURE_OF_CALL_INSTRUCTIONS
    if not (state.get("first_name") and state.get("last_name") and state.get("email") and state.get("phone_number")):
        return COLLECT_DETAILS_INSTRUCTIONS
    else:
        return FINAL_INSTRUCTIONS


async def main():
    # Check for required environment variables
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set it in your environment or create a .env file with your API key")
        return

    stt = OpenAISTTService(
        language=Language.EN,
        prompt="Expect the user to speak in English."
    )
    llm = OpenAILLMService(
        model="gpt-5-mini",
        reasoning_effort="minimal",
    )
    tts = OpenAITTSService()
    

    scratchpad = {}

    async def json_patch_scratchpad(function_call_params: FunctionCallParams):
        args = function_call_params.arguments
        op = args.get("op")
        path = args.get("path")
        value = args.get("value")
        patch = jsonpatch.JsonPatch([{
            "op": op,
            "path": path,
            "value": value
        }])
        try:
            patch.apply(scratchpad, in_place=True)
            response = f"The JSON patch was applied successfully."
        except Exception as e:
            response = f"There was an error applying the JSON patch to the scratchpad: {e}"
        await function_call_params.result_callback(
            response,
            properties=FunctionCallResultProperties(
                run_llm=False,
                on_context_updated=lambda: function_call_params.llm.push_frame(LLMRunFrame(), FrameDirection.UPSTREAM)
            )
        )

    json_patch_schema = FunctionSchema(
        name="json_patch_scratchpad",
        description="Apply a JSON patch to the scratchpad following the JSON patch RFC 6902 specification",
        properties={
            "op": {
                "type": "string",
                "enum": ["add", "remove", "replace", "move", "copy", "test"]
            },
            "path": {
                "type": "string",
                "description": "The path to the property to apply the patch to"
            },
            "value": {
                "type": "string",
                "description": "The value to apply to the property"
            }
        },
        required=["op", "path"]
    )
    tools = ToolsSchema(standard_tools=[json_patch_schema])

    context = OpenAILLMContext(
        messages=[],
        tools=tools
    )
    context_aggregator = llm.create_context_aggregator(context)
    llm.register_function(
        "json_patch_scratchpad",
        json_patch_scratchpad,
        cancel_on_interruption=True
    )

    context_injector = ContextInjector()

    @context_injector.register_context_handler(mutable=False)
    def context_handler(context: OpenAILLMContext) -> OpenAILLMContext:
        # Developer Prompt
        system_message = {
            "role": "developer",
            "content": SYSTEM_PROMPT
        }
        context.messages.insert(0, system_message)

        for message in context.get_messages():
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    if not re.match(r"<.*>.*</.*>", content):
                        message["content"] = f"<caller_transcription> {content} </caller_transcription>"
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            if not re.match(r"<.*>.*</.*>", item.get("text")):
                                item["text"] = f"<caller_transcription> {item.get('text')} </caller_transcription>"


        # System Instructions
        system_instructions = get_system_instructions(scratchpad)

        # Scratchpad message
        
        scratchpad_message = f"<scratchpad> {json.dumps(scratchpad)} </scratchpad>"

        context.add_message(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": system_instructions
                    },
                    {
                        "type": "text",
                        "text": scratchpad_message
                    }
                ]

            }
        )
        return context

    logger = FrameLogger()
    
    # Create local audio transport (uses your computer's mic and speakers)
    # Use Silero VAD if available to detect user speech turns.
    vad = SileroVADAnalyzer() if SileroVADAnalyzer else None
    if vad is None:
        print("Warning: VAD not enabled. Install 'pipecat-ai[silero]' and re-run for best results.")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            vad_analyzer=vad
        )
    )
    
    # Create pipeline with context aggregators around the LLM
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        logger,
        context_injector,
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])
    
    # Create and run task
    task = PipelineTask(pipeline)
    # Kick off an initial run so context/settings are sent
    await task.queue_frames([LLMRunFrame()])
    
    # Log LLM-specific activity (start/end, generated text, function calls, context/messages)
    runner = PipelineRunner()
    
    print("🎤 Pipecat Voice Bot")
    print("=" * 40)
    print("Using OpenAI Realtime API for speech-to-speech")
    print("Speak naturally - the bot will respond with voice")
    print("Press Ctrl+C to stop")
    print("=" * 40)
    
    try:
        await runner.run(task)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"Error running bot: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            await transport.cleanup()
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

if __name__ == "__main__":
    asyncio.run(main())

from ast import arguments
import os
import sys
from pathlib import Path
import asyncio

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from deepgram import LiveOptions
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
import pipecat.audio.turn.smart_turn.base_smart_turn as base_smart_turn_module
base_smart_turn_module.USE_ONLY_LAST_VAD_SEGMENT = False
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel, Language
from pipecat.transports.base_input import InputAudioRawFrame, VADState, EndOfTurnState, BaseInputTransport

async def my_run_turn_analyzer(
        self, frame: InputAudioRawFrame, vad_state: VADState, previous_vad_state: VADState
    ):
        """Run turn analysis on audio frame and handle results."""
        is_speech = vad_state == VADState.SPEAKING or vad_state == VADState.STARTING
        # If silence exceeds threshold, we are going to receive EndOfTurnState.COMPLETE
        end_of_turn_state = self._params.turn_analyzer.append_audio(frame.audio, is_speech)
        if end_of_turn_state == EndOfTurnState.COMPLETE:
            await self._handle_end_of_turn_complete(end_of_turn_state)
        # Otherwise we are going to trigger to check if the turn is completed based on the VAD
        elif vad_state == VADState.QUIET and vad_state != previous_vad_state:
            await self._handle_end_of_turn()
        elif self.awaiting_end_of_turn and (self.frame_count + 1) % 10 == 0:
            await self._handle_end_of_turn()
        elif self.awaiting_end_of_turn and (self.frame_count + 1) % 10 != 0:
            self.frame_count += 1

async def my_handle_end_of_turn_complete(self, state: EndOfTurnState):
        """Handle completion of end-of-turn analysis."""
        audio_buffer_length = len(self._params.turn_analyzer._audio_buffer)
        print("--------------------------------")
        print(f"Audio buffer length: {audio_buffer_length}")
        print("--------------------------------")
        if state == EndOfTurnState.COMPLETE:
            await self._handle_user_interruption(VADState.QUIET)
            self.awaiting_end_of_turn = False
            self.frame_count = 0
        else:
            self.awaiting_end_of_turn = True
            self.frame_count += 1
BaseInputTransport._run_turn_analyzer = my_run_turn_analyzer
BaseInputTransport._handle_end_of_turn_complete = my_handle_end_of_turn_complete

async def run_bot(transport: BaseTransport, handle_sigint: bool):
    # Configure logger early, before creating analyzers
    logger.remove()
    def console_filter(record):
        # record["name"] is the name where the log call was made
        names = {"pipecat.audio.turn", "pipecat.audio.vad"}
        return any(record["name"].startswith(name) for name in names)
    
    logger.add(sys.stderr, level="TRACE", filter=console_filter)
    logger.add(sys.stderr, level="ERROR")

    questionnaire = {
        "1.1": {
            "question_text": "What is the callers First Name?",
            "spelling_sensitive": False,
            "answer": None
        },
        "1.2": {
            "question_text": "What is the callers Last Name?",
            "spelling_sensitive": False,
            "answer": None
        },
        "1.3": {
            "question_text": "What is the callers email address?",
            "spelling_sensitive": True,
            "answer": None
        },
        "1.4": {
            "question_text": "What is the callers phone number?",
            "spelling_sensitive": True,
            "answer": None
        },
        "2.1": {
            "question_text": "What is the callers issue?",
            "spelling_sensitive": False,
            "answer": None
        }
    }

    set_answer_function_schema = FunctionSchema(
        name="set_answer",
        description=(
            "Set the answer to a question. "
        ),
        properties={
            "question_id": {
                "type": "string",
                "description": "The id of the question to set the answer for"
            },
            "answer": {
                "type": "string",
                "description": "The answer to the question"
            }
        },
        required=["question_id", "answer"]
    )
    async def set_answer(params: FunctionCallParams):
        try:
            arguments = params.arguments
            question_id = arguments.get("question_id", None)
            assert isinstance(question_id, str) and question_id in questionnaire, "Question ID must be a string and must be a valid question ID form the form"
            answer = arguments.get("answer", None)
            assert isinstance(answer, str), "Answer must be a string"
            question_id = question_id.strip()
            answer = answer.strip()
            question = questionnaire[question_id]
            question["answer"] = answer
            if question["spelling_sensitive"]:
                response = {
                    "status": "COMPLETED",
                    "result": f"Question {question_id} has been set to: {answer}",
                    "instructions": (
                        f"'{question_id}' is a spelling sensitive field, please confirm with the caller that it is correct by reading it back to them. In your response represent the value as {"-".join(list(answer.upper()))}. "
                        f"You must do this every time you attempt to set the value for question {question_id}."
                    )
                }
            else:
                response = {
                    "status": "COMPLETED",
                    "result": f"Question {question_id} has been set to: {answer}",
                    "instructions": None
                }
            await params.result_callback(response)
            return
        except AssertionError as e:
            await params.result_callback(
                {"status": "ERROR", "reason": e.args[0]}
            )
            return
        except Exception as e:
            await params.result_callback(
                {"status": "CRIRICAL_ERROR"}
            )
            logger.error(f"An unexpected error occurred while setting the answer for question {question_id}: {e}")
            return
    
    tools = ToolsSchema(
        standard_tools=[set_answer_function_schema]
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5-mini",
        params=OpenAILLMService.InputParams(
            extra={
                "reasoning_effort": "minimal"
            }
        )
    )
    llm.register_function("set_answer", set_answer)
    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "# Role\n"
                            "You are a telephone operator that's job is is to help the caller log complaints or issues. "
                            "To do this you will need to fill out a form on behalf of the caller. "
                            "You should always be friendly and helpful. "
                            "You should keep your responses short and concise to maintain a natural conversation flow."
                        )
                    },
                    {
                        "type": "text",
                        "text": (
                            "# Speech to Text\n"
                            "The caller will be speaking into a microphone and the speech will be transcribed into text before it is provided to you. "
                            "You should bear in mind the the transcription layer can make mistakes and do your best to intuit the intended meaning of the caller. "
                            "\n\n## Context Awareness\n"
                            "The transcription engine is not context aware, this means it only has access to the final piece of audio provided by the caller. "
                            "Therefore, it is likely that it could make the same mistake even after it was corrected by the caller. "
                            "Do not be alarmed by this, it is a limitation of the technology and you should do your best to deal with it gracefully and not allow it to disrupt the flow of the conversation. "
                            "If you are really unsure about the intended meaning of the caller you should find a natural way to ask them some clarifying questions. "
                            "\n\nImportant: Do not mention the speech to text process to the caller. Deal with any issues naturally and take responsibility for any mistakes made by the transcription layer. "
                        )
                    },
                    {
                        "type": "text",
                        "text": (
                            "# Text to Speech\n"
                            "The text to speech layer will be used to convert the text you provide to speech. "
                            "## Formatting Responses\n"
                            "You must provide responses in a format that converts to natural language speech. "
                            "You should avoid any structure that lends itself to document formatting such as lists, bullet points, headers, etc. "
                            "You should pay close attention to the formatting instructions provided by the tool responses. "
                            "These are likely given so the text to speech layer can handle the text appropriately. "
                            "\n\n## Context Awareness\n"
                            "The text to speech layer is not context aware, this means it only has access to the final piece of text provided to it. "
                            "Therefore, it is likely that it could make the same mistake even after it was corrected by the caller. "
                            "Do not be alarmed by this, it is a limitation of the technology and you should do your best to deal with it gracefully and not allow it to disrupt the flow of the conversation. "
                            "If the caller complains or attempts to correct pronounciation patiently explain that you are doing your best but you are unable to adapt your pronounciation. "
                            "If this becomes a significant issue and the callers seems particularly angry you should apologize to the caller and explain that you will need to redirect the call to a human operator. "
                            "\n\nImportant: Do not mention the text to speech process to the caller. Deal with any issues naturally and take responsibility for any mistakes made by the text to speech layer. "
                        )
                    },
                    {
                        "type": "text",
                        "text": (
                            "# Voice Activity Detection\n"
                            "There is a voice activity detection (VAD) layer in the pipeline that will detect when the caller is speaking and when they are not. "
                            "There are limitations to the technology and this may cause issued. "
                            "For instance, the VAD is triggered whenever any speaking is detected. It is not able to differentiate between the caller speaking and people speaking in the background. "
                            "Therefore it is possible the VAD will be triggered by background noise or other people speaking. "
                            "You should be able to tell when this is the case as the text provided by the speech to text layer will be out of place and seem out of context or the speech will be poorly formed and jumbled. "
                            "If you believe this to be the case you should inform the caller that you believe background noise is interferring with the conversation and politely ask them to find a quieter place to continue the conversation. "
                            "\n\nImportant: Do not mention the VAD layer to the caller. Deal with any issues naturlly and take responsibility for any mistakes made by the VAD layer. "
                        )
                    },
                    {
                        "type": "text",
                        "text": (
                            "# System Messages\n"
                            "Throughout the conversation you may receive system messages from the pipeline. "
                            "These messages are not from the caller and should NEVER be mentioned to the caller. "
                            "These are provided to you by the pipeline to help you navigate the conversation and take the appropriate actions. "
                            "You must follow the instructions provided by system messages. "
                            "You will know text is a system message message when it is enclosed in <system> tags with the format `<system>...</system>`. "
                            "The caller has no knowledge of system messages and should not be aware of them. "
                        )
                    },
                    {
                        "type": "text",
                        "text": (
                            "# Tool Usage\n"
                            "You will need to use the tools provided to you by the pipeline to help you throughout the call. "
                            "\n\n## Standard Tool Responses\n"
                            "Most tools will respond in json format with the format `{\"status\": \"COMPLETED\", \"result\": ..., \"instructions\": ...}`. "
                            "The status \"COMPLETED\" indicates that the tool has completed its task and the result is ready to be used. "
                            "The result will be provided in the result field and could be in string or JSON format. "
                            "The instructions field will be either a string or null. "
                            "If the instructions field is a string, it will be natural language instructions you must follow as soon as you get the opportunity. "
                            "For instance, if you have just filled out a field of the form that is spelling sensitive you may recieve instructions such as \"'email' is a spelling senstivie field, please confirm with the caller that it is correct by reading it back to them. In your response represent the email as J-A-M-E-S-@-E-X-A-M-P-L-E-.C-O-M\". "
                            "You MUST follow the instructions provided by the tool but maintain a natural conversation flow. "
                            "Pay close attention to any formatting instructions provided, they are there to make sure the text to speech layer handles the text appropriately. "
                            "\n\n## Pending Tool Responses\n"
                            "Some tools may respond in the format `{\"status\": \"PENDING\"}`. "
                            "This means that is will take a few seconds for the tool to complete its task and return the result. "
                            "The final result will be provided in special user text enclosed in <tool_result> tags with the format `<tool_result>{\"tool_name\": <tool_name>, \"result\": <result>}</tool_result>`. "
                            "This text is not from the caller and should NEVER be mentioned to the caller. "
                            "You should just naturally use the results as if they were a function output. "
                            "\n\n## Tool Usage Error Responses\n"
                            "If you have used a tool incorrectly which causes an error and the reason for the error is known you will receive a response in the format `{\"status\": \"ERROR\", \"reason\": \"...\"}`. "
                            "The reason field will be a natural language explanation of what caused the error. "
                            "If this happens you should attempt to address the error and continue the conversation as normal. "
                            "You may need to collect more information from the caller to address the error or you may be able to address it immediately with another tool call. "
                            "If this happens multiple times in a row and you are unable to address the error you should apologize to the caller and explain that you will need to redirect the call to a human operator. "
                            "\n\n## Critical Errors\n"
                            "If you receive a response in the format `{\"status\": \"CRITICAL_ERROR\"}`. "
                            "This means that the error is critical and you are unable to continue the conversation. "
                            "You MUST apologize to the caller and explain that you will need to redirect the call to a human operator. "
                        )
                    },
                    {
                        "type": "text",
                        "text": (
                            "# Form Filling\n"
                            "To log a complaint or issue for the user you will need to fill out a form. "
                            "The whole form must be filled out for the complaint to be logged. "
                            "You can fill out the form in any order you see fit, you should try to fill out the form in a way that is natural and goes with the flow of the conversation with the caller. "
                            "The form you are required to fill in will be provided in the next section. "
                            "The questions will be provided in the following format:"
                            "\n\n<question><question_id>...</question_id><question_text>...</question_text><spelling_sensitive>...</spelling_sensitive></question>\n"
                        )
                    },
                    {
                        "type": "text",
                        "text": (
                            "# Form\n"
                            "<form>\n"
                            "<question><question_id>1.1</question_id><question_text>What is the callers First Name?</question_text><spelling_sensitive>false</spelling_sensitive></question>\n"
                            "<question><question_id>1.2</question_id><question_text>What is the callers Last Name?</question_text><spelling_sensitive>false</spelling_sensitive></question>\n"
                            "<question><question_id>1.3</question_id><question_text>What is the callers email address?</question_text><spelling_sensitive>true</spelling_sensitive></question>\n"
                            "<question><question_id>1.4</question_id><question_text>What is the callers phone number?</question_text><spelling_sensitive>true</spelling_sensitive></question>\n"
                            "<question><question_id>2.1</question_id><question_text>What is the callers issue?</question_text><spelling_sensitive>false</spelling_sensitive></question>\n"
                            "</form>"     
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<system>Say \"Hello, my name is Hunter's Digital and I'm here to help you log your issue.\"</system>"
                        )
                    }
                ]
            }
        ],
        tools=tools
    )
    context_aggregator = LLMContextAggregatorPair(context)


    stt = WhisperSTTServiceMLX(
        model=MLXModel.LARGE_V3_TURBO_Q4,  # or MEDIUM, LARGE_V3, etc.
        language=Language.EN
    )

    tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-2-andromeda-en")
    input_processor = transport.input()
    input_processor.awaiting_end_of_turn = False
    input_processor.frame_count = 0
    pipeline = Pipeline(
        [
            input_processor,  # Websocket input from client
            stt,
            context_aggregator.user(),
            #llm,  # LLM
            #tts,
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        )
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        print(questionnaire)
        for item in context.messages:
            print("--------------------------------")
            print(item)
            print("--------------------------------")
        await task.cancel()


    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""

    

    _, call_data = await parse_telephony_websocket(runner_args.websocket)
    from_number = call_data["from"]

    # Extract the from number from the call data, which allows you to identify the caller.
    # With this information, you can make a request to your API to get the user's information
    # and inject that information into your bot's configuration.
    logger.info(f"From number: {from_number}")
    

    serializer = TelnyxFrameSerializer(
        stream_id=call_data["stream_id"],
        outbound_encoding=call_data["outbound_encoding"],
        inbound_encoding="PCMU",
        call_control_id=call_data["call_control_id"],
        api_key=os.getenv("TELNYX_API_KEY"),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2, min_volume=0.5)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(
                params=SmartTurnParams(
                    stop_secs=2,
                    pre_speech_ms=400,
                )
            ),
            serializer=serializer,
        ),
    )

    handle_sigint = runner_args.handle_sigint

    await run_bot(transport, handle_sigint)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
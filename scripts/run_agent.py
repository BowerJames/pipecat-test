import asyncio
from dotenv import load_dotenv
import os

from pipecat.frames.frames import LLMRunFrame
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.services.openai.realtime.context import OpenAILLMContext
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.openai.realtime.events import SessionProperties
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.tools_schema import ToolsSchema
import pipecat.services.openai.realtime.events as events

from pipecat_extension.services.openai_realtime_llm_service import OpenAIRealtimeLLMServiceExt

load_dotenv()

get_users_name_schema = {
    "name": "get_users_name",
    "description": "Get the users name",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
    "type": "function",
}

get_users_name_function = FunctionSchema(
    name="get_users_name",
    description="Get the users name",
    properties={},
    required=[],
)

async def get_users_name(params: FunctionCallParams):
    await params.result_callback(
        {"status": "PENDING"}
    )



async def run_agent():
    transport_params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=24000,
        audio_in_channels=1,
        audio_out_sample_rate=24000,
        audio_out_channels=1,
    )
    transport = LocalAudioTransport(
        transport_params,
    )


    session_properties = SessionProperties(
        instructions=(
            "You are a helpful assistant that can answer questions and help with tasks. "
            "You have access to tools that can help you with your tasks. "
            "Most tools will respond with `{\"status\": \"PENDING\"}` after they have been called. "
            "This means that is will take a few seconds for the tool to complete its task and return the result. "
            "The final result will be provided in special user messages enclosed in <tool_result> tags with the format `{\"tool_name\": <tool_name>, \"result\": <result>}`. "
            "These messages are not from the caller and should NEVER be mentioned to the caller. "
            "You should just naturally use the results as if they were a function output. "
        ),
        tools=[get_users_name_schema]
    )
    llm = OpenAIRealtimeLLMServiceExt(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-realtime-mini",
        session_properties=session_properties,
    )
    llm.register_function("get_users_name", get_users_name)

    context = OpenAILLMContext()
    context_aggregator = llm.create_context_aggregator(context=context)

    @llm.event_handler("after_function_call_output_sent")
    async def after_function_call_output_sent_handler(processor: OpenAIRealtimeLLMServiceExt, frame):
        await processor.send_client_event(
            events.ConversationItemCreateEvent(
                item=events.ConversationItem(
                    type="message",
                    role="user",
                    content=[
                        events.ItemContent(
                            type="input_text",
                            text="<tool_result>{\"tool_name\": \"get_users_name\", \"result\": \"James Bower\"}</tool_result>"
                        )
                    ]
                )
            )
        )

    

    pipeline = Pipeline(
        [
            transport.input(),
            llm,
            context_aggregator.assistant(),
            transport.output(),
        ]
    )

    conversation = {
        "items": {}
    }

    # Create and run the pipeline task
    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    @llm.event_handler("on_conversation_item_created")
    async def on_conversation_item_created_handler(processor: OpenAIRealtimeLLMServiceExt, item_id: str, item: events.ConversationItem):
        conversation["items"][item_id] = item

    @llm.event_handler("on_conversation_item_updated")
    async def on_conversation_item_updated_handler(processor: OpenAIRealtimeLLMServiceExt, item_id: str, item: events.ConversationItem):
        conversation["items"][item_id] = item

    @llm.event_handler("on_conversation_item_deleted")
    async def on_conversation_item_deleted_handler(processor: OpenAIRealtimeLLMServiceExt, item_id: str):
        del conversation["items"][item_id]

    @transport.event_handler("on_client_connected")
    async def on_client_connected_handler(tranport, client):
        await task.queue_frame(
            [
                LLMRunFrame()
            ]
        )
    

    print("Starting agent... Press Ctrl+C to stop.")
    await runner.run(task)

    for item in conversation["items"].values():
        print("--------------------------------")
        print(item)
        print("--------------------------------")
        
if __name__ == "__main__":
    asyncio.run(run_agent())


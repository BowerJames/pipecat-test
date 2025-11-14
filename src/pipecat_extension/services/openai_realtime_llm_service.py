from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


class OpenAIRealtimeLLMServiceExt(OpenAIRealtimeLLMService):
    """Extended OpenAI Realtime LLM Service that registers additional event handlers."""

    def __init__(self, *args, **kwargs):
        """Initialize the extended service and register the after_function_call_output_sent handler."""
        super().__init__(*args, **kwargs)
        self._register_event_handler("after_function_call_output_sent")

    async def _handle_function_call_result(self, frame):
        """Handle function call result and trigger the after_function_call_output_sent event handler."""
        await super()._handle_function_call_result(frame)
        await self._call_event_handler("after_function_call_output_sent")


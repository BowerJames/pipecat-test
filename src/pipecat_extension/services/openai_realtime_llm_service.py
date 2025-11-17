from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.openai.realtime import events


class OpenAIRealtimeLLMServiceExt(OpenAIRealtimeLLMService):
    """Extended OpenAI Realtime LLM Service that registers additional event handlers."""

    def __init__(self, *args, **kwargs):
        """Initialize the extended service and register the after_function_call_output_sent and on_conversation_item_deleted handlers."""
        super().__init__(*args, **kwargs)
        self._register_event_handler("after_function_call_output_sent", sync=True)
        self._register_event_handler("on_conversation_item_deleted")
        self._register_event_handler("on_session_updated")

    async def _handle_function_call_result(self, frame):
        """Handle function call result and trigger the after_function_call_output_sent event handler."""
        await super()._handle_function_call_result(frame)
        await self._call_event_handler("after_function_call_output_sent", frame)

    async def _receive_task_handler(self):
        """Override to handle conversation.item.deleted events."""
        async for message in self._websocket:
            evt = events.parse_server_event(message)
            if evt.type == "session.created":
                await self._handle_evt_session_created(evt)
            elif evt.type == "session.updated":
                await self._handle_evt_session_updated(evt)
            elif evt.type == "response.output_audio.delta":
                await self._handle_evt_audio_delta(evt)
            elif evt.type == "response.output_audio.done":
                await self._handle_evt_audio_done(evt)
            elif evt.type == "conversation.item.added":
                await self._handle_evt_conversation_item_added(evt)
            elif evt.type == "conversation.item.done":
                await self._handle_evt_conversation_item_done(evt)
            elif evt.type == "conversation.item.input_audio_transcription.delta":
                await self._handle_evt_input_audio_transcription_delta(evt)
            elif evt.type == "conversation.item.input_audio_transcription.completed":
                await self.handle_evt_input_audio_transcription_completed(evt)
            elif evt.type == "conversation.item.retrieved":
                await self._handle_conversation_item_retrieved(evt)
            elif evt.type == "conversation.item.deleted":
                await self._handle_evt_conversation_item_deleted(evt)
            elif evt.type == "response.done":
                await self._handle_evt_response_done(evt)
            elif evt.type == "input_audio_buffer.speech_started":
                await self._handle_evt_speech_started(evt)
            elif evt.type == "input_audio_buffer.speech_stopped":
                await self._handle_evt_speech_stopped(evt)
            elif evt.type == "response.output_text.delta":
                await self._handle_evt_text_delta(evt)
            elif evt.type == "response.output_audio_transcript.delta":
                await self._handle_evt_audio_transcript_delta(evt)
            elif evt.type == "response.function_call_arguments.done":
                await self._handle_evt_function_call_arguments_done(evt)
            elif evt.type == "error":
                if not await self._maybe_handle_evt_retrieve_conversation_item_error(evt):
                    await self._handle_evt_error(evt)
                    # errors are fatal, so exit the receive loop
                    return

    async def _handle_evt_conversation_item_deleted(self, evt):
        """Handle conversation.item.deleted event and trigger the on_conversation_item_deleted event handler."""
        await self._call_event_handler("on_conversation_item_deleted", evt.item_id)


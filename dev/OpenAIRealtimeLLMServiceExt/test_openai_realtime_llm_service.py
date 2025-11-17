import pytest
from unittest.mock import Mock, AsyncMock, create_autospec, call
import asyncio

from pipecat_extension.services.openai_realtime_llm_service import OpenAIRealtimeLLMServiceExt
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
import pipecat.services.openai.realtime.events as events

from dev.utils import wait_for_mock_awaited

def test_openai_realtime_llm_service_ext_init(
    monkeypatch: pytest.MonkeyPatch,
    mock_pipecat_service_init: Mock,
    mock_register_event_handler: Mock,
    after_function_call_output_event_handler_name: str,
    on_conversation_item_deleted_event_handler_name: str,
):
    with monkeypatch.context() as m:
        m.setattr(OpenAIRealtimeLLMService, "__init__", mock_pipecat_service_init)
        m.setattr(OpenAIRealtimeLLMService, "_register_event_handler", mock_register_event_handler)
        args = (Mock(), Mock())
        kwargs = {
            "name": "test",
            "lorem": "ipsum",
        }
        openai_realtime_llm_service_ext = OpenAIRealtimeLLMServiceExt(*args, **kwargs)
        mock_pipecat_service_init.assert_called_once_with(*args, **kwargs)
        mock_register_event_handler.assert_has_calls([call(after_function_call_output_event_handler_name, sync=True), call(on_conversation_item_deleted_event_handler_name)], any_order=True)


@pytest.mark.asyncio
async def test_openai_realtime_llm_service_ext_after_function_call_output_sent_event_handler_called(
    monkeypatch: pytest.MonkeyPatch,
    connected_processor: OpenAIRealtimeLLMServiceExt,
    mock_after_function_call_output_event_handler: AsyncMock,
):
    mock_handle_function_call_result = AsyncMock()
    with monkeypatch.context() as m:
        m.setattr(OpenAIRealtimeLLMService, "_handle_function_call_result", mock_handle_function_call_result)

        processor = connected_processor
        frame = Mock()
        await processor._handle_function_call_result(frame)
        mock_handle_function_call_result.assert_awaited_once_with(frame)
        mock_after_function_call_output_event_handler.assert_awaited_once_with(processor, frame)

@pytest.mark.asyncio
async def test_openai_realtime_llm_service_ext_on_conversation_item_deleted_event_handler_called(
    monkeypatch: pytest.MonkeyPatch,
    connected_processor: OpenAIRealtimeLLMServiceExt,
    inbound_websocket_queue: asyncio.Queue,
    mock_on_conversation_item_deleted_event_handler: AsyncMock,
):
    delete_event = events.ConversationItemDeleted(
        event_id="test_event_id",
        type="conversation.item.deleted",
        item_id="test_item_id",
    )
    inbound_websocket_queue.put_nowait(
        delete_event.model_dump_json()
    )
    await connected_processor._receive_task_handler()
    await wait_for_mock_awaited(mock_on_conversation_item_deleted_event_handler, 1)
    mock_on_conversation_item_deleted_event_handler.assert_awaited_once_with(connected_processor, delete_event.item_id)

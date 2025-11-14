import pytest
from unittest.mock import Mock, AsyncMock, create_autospec

from pipecat_extension.services.openai_realtime_llm_service import OpenAIRealtimeLLMServiceExt
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

def test_openai_realtime_llm_service_ext_init(
    monkeypatch: pytest.MonkeyPatch,
    mock_pipecat_service_init: Mock,
    mock_register_event_handler: Mock,
    after_function_call_output_event_handler_name: str,
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
        mock_register_event_handler.assert_called_once_with(after_function_call_output_event_handler_name)


@pytest.mark.asyncio
async def test_openai_realtime_llm_service_ext_after_function_call_output_sent_event_handler_called(
    monkeypatch: pytest.MonkeyPatch,
    mock_pipecat_service_init: Mock,
    mock_register_event_handler: Mock,
    after_function_call_output_event_handler_name: str,
):
    mock_call_event_handler = AsyncMock()
    mock_handle_function_call_result = AsyncMock()
    with monkeypatch.context() as m:
        m.setattr(OpenAIRealtimeLLMService, "__init__", mock_pipecat_service_init)
        m.setattr(OpenAIRealtimeLLMService, "_register_event_handler", mock_register_event_handler)
        m.setattr(OpenAIRealtimeLLMService, "_call_event_handler", mock_call_event_handler)
        m.setattr(OpenAIRealtimeLLMService, "_handle_function_call_result", mock_handle_function_call_result)

        processor = OpenAIRealtimeLLMServiceExt()
        frame = Mock()
        await processor._handle_function_call_result(frame)
        mock_handle_function_call_result.assert_awaited_once_with(frame)
        mock_call_event_handler.assert_awaited_once_with(after_function_call_output_event_handler_name)


import pytest
from unittest.mock import Mock

from pipecat_extension.services.openai_realtime_llm_service import OpenAIRealtimeLLMServiceExt
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

@pytest.fixture
def after_function_call_output_event_handler_name() -> str:
    return "after_function_call_output_sent"

@pytest.fixture
def mock_pipecat_service_init(monkeypatch: pytest.MonkeyPatch) -> Mock:
    return Mock(return_value=None)

@pytest.fixture
def mock_register_event_handler(after_function_call_output_event_handler_name: str) -> Mock:
    return Mock()

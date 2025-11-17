import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock
import asyncio

from pipecat_extension.services.openai_realtime_llm_service import OpenAIRealtimeLLMServiceExt
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

@pytest.fixture
def api_key() -> str:
    return "test_api_key"

@pytest.fixture
def after_function_call_output_event_handler_name() -> str:
    return "after_function_call_output_sent"

@pytest.fixture
def on_conversation_item_deleted_event_handler_name() -> str:
    return "on_conversation_item_deleted"

@pytest.fixture
def mock_after_function_call_output_event_handler() -> AsyncMock:
    return AsyncMock()

@pytest.fixture
def mock_on_conversation_item_deleted_event_handler() -> AsyncMock:
    return AsyncMock()

@pytest.fixture
def mock_pipecat_service_init() -> Mock:
    return Mock(return_value=None)

@pytest.fixture
def inbound_websocket_queue() -> asyncio.Queue:
    return asyncio.Queue()

@pytest.fixture
def mock_websocket(monkeypatch: pytest.MonkeyPatch, inbound_websocket_queue: asyncio.Queue):
    mock_websocket = AsyncMock()
    def __aiter__(self):
        return self
    async def __anext__(self):
        if inbound_websocket_queue.empty():
            raise StopAsyncIteration
        event = await inbound_websocket_queue.get()
        return event
    mock_websocket.__aiter__ = __aiter__
    mock_websocket.__anext__ = __anext__

    return mock_websocket


@pytest.fixture
def mock_register_event_handler(after_function_call_output_event_handler_name: str) -> Mock:
    return Mock()

@pytest_asyncio.fixture
async def connected_processor(
    monkeypatch: pytest.MonkeyPatch,
    mock_websocket: AsyncMock,
    mock_after_function_call_output_event_handler: AsyncMock,
    mock_on_conversation_item_deleted_event_handler: AsyncMock,
    after_function_call_output_event_handler_name: str,
    on_conversation_item_deleted_event_handler_name: str,
    api_key: str,
):
    with monkeypatch.context() as m:
        async def _connect(self, *args, **kwargs): 
            self._websocket = mock_websocket
        m.setattr(OpenAIRealtimeLLMService, "_connect", _connect)
        processor = OpenAIRealtimeLLMServiceExt(api_key=api_key)

        @processor.event_handler(after_function_call_output_event_handler_name)
        async def after_function_call_output_event_handler(processor, frame):
            await mock_after_function_call_output_event_handler(processor, frame)

        @processor.event_handler(on_conversation_item_deleted_event_handler_name)
        async def on_conversation_item_deleted_event_handler(processor, item_id):
            await mock_on_conversation_item_deleted_event_handler(processor, item_id)

        await processor._connect()
    return processor

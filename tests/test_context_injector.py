import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext
import copy

from pipecat_test.context_injector import ContextInjector


@pytest.fixture(autouse=True)
def mock_push_frame():
    """Autouse fixture to mock push_frame on all ContextInjector instances."""
    with patch.object(ContextInjector, 'push_frame', new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_super_process_frame():
    """Autouse fixture to mock the superclass's process_frame method."""
    with patch.object(
        ContextInjector.__bases__[0],
        "process_frame",
        new_callable=AsyncMock
    ) as mock:
        yield mock

@pytest.fixture()
def mock_handle_llm_context_frame():
    with patch.object(ContextInjector, '_handle_llm_context_frame', new_callable=AsyncMock, create=True) as mock:
        yield mock

@pytest.fixture()
def mock_context_handler():
    with patch.object(ContextInjector, '_context_handler', new_callable=MagicMock, create=True) as mock:
        yield mock

@pytest.fixture
def process_frame_processor(mock_handle_llm_context_frame):
    """Provides a ContextInjector instance with mocks already set up."""
    processor = ContextInjector()
    processor._handle_llm_context_frame = mock_handle_llm_context_frame
    return processor

@pytest.fixture()
def handle_llm_context_frame_processor(mock_context_handler):
    """Provides a ContextInjector instance with mocks already set up."""
    processor = ContextInjector()
    processor._context_handler = mock_context_handler
    return processor

@pytest.fixture()
def frame():
    """Provides a Frame instance."""
    return Frame()

@pytest.fixture()
def empty_llm_context_frame():
    """Provides a LLMContextFrame instance."""
    return LLMContextFrame(
        context=LLMContext()
    )

@pytest.mark.parametrize("direction", [FrameDirection.UPSTREAM, FrameDirection.DOWNSTREAM])
@pytest.mark.asyncio
async def test_super_process_frame(frame, direction, process_frame_processor, mock_super_process_frame):

    await process_frame_processor.process_frame(frame, direction)

    mock_super_process_frame.assert_awaited_once_with(frame, direction)

@pytest.mark.parametrize("direction", [FrameDirection.UPSTREAM, FrameDirection.DOWNSTREAM])
@pytest.mark.asyncio
async def test_push_frame(direction, process_frame_processor, mock_push_frame):

    await process_frame_processor.process_frame(frame, direction)

    mock_push_frame.assert_awaited_once_with(frame, direction)


@pytest.mark.parametrize("direction", [FrameDirection.UPSTREAM, FrameDirection.DOWNSTREAM])
@pytest.mark.asyncio
async def test_handle_llm_context_frame(empty_llm_context_frame, direction, process_frame_processor, mock_handle_llm_context_frame):
    await process_frame_processor.process_frame(empty_llm_context_frame, direction)
    mock_handle_llm_context_frame.assert_awaited_once_with(empty_llm_context_frame, direction)


@pytest.mark.parametrize("direction", [FrameDirection.UPSTREAM, FrameDirection.DOWNSTREAM])
@pytest.mark.asyncio
async def test_llm_context_not_default_pushed(empty_llm_context_frame, direction, process_frame_processor, mock_push_frame):
    await process_frame_processor.process_frame(empty_llm_context_frame, direction)
    mock_push_frame.assert_not_awaited()

def test_register_current_context_handler():
    processor = ContextInjector()

    @processor.register_context_handler
    def context_handler(context: LLMContext) -> LLMContext:
        return context

    assert processor._context_handler is context_handler

def test_defaults_to_no_context_handler():
    processor = ContextInjector()
    assert processor._context_handler is None

@pytest.mark.parametrize("direction", [FrameDirection.UPSTREAM, FrameDirection.DOWNSTREAM])
@pytest.mark.asyncio
async def test_no_context_handler_pushes_same_frame(empty_llm_context_frame, direction, handle_llm_context_frame_processor, mock_push_frame):
    handle_llm_context_frame_processor._context_handler = None
    await handle_llm_context_frame_processor._handle_llm_context_frame(empty_llm_context_frame, direction)
    args, kwargs = mock_push_frame.call_args
    assert args[0] is empty_llm_context_frame
    assert args[1] == direction

@pytest.mark.parametrize("direction", [FrameDirection.UPSTREAM, FrameDirection.DOWNSTREAM])
@pytest.mark.asyncio
async def test_context_handler_with_mutable_context(empty_llm_context_frame, direction, handle_llm_context_frame_processor, mock_context_handler):
    @handle_llm_context_frame_processor.register_context_handler(mutable=True)
    def context_handler(context: LLMContext) -> LLMContext:
        pass

    handle_llm_context_frame_processor._context_handler = mock_context_handler
    mock_context_handler.return_value = empty_llm_context_frame.context

    await handle_llm_context_frame_processor._handle_llm_context_frame(empty_llm_context_frame, direction)
    args, _ = mock_context_handler.call_args
    assert args[0] is empty_llm_context_frame.context

@pytest.mark.parametrize("direction", [FrameDirection.UPSTREAM, FrameDirection.DOWNSTREAM])
@pytest.mark.asyncio
async def test_context_handler_with_immutable_context(empty_llm_context_frame, direction, handle_llm_context_frame_processor, mock_context_handler):
    @handle_llm_context_frame_processor.register_context_handler(mutable=False)
    def context_handler(context: LLMContext) -> LLMContext:
        pass

    handle_llm_context_frame_processor._context_handler = mock_context_handler
    mock_context_handler.return_value = copy.deepcopy(empty_llm_context_frame.context)

    await handle_llm_context_frame_processor._handle_llm_context_frame(empty_llm_context_frame, direction)
    args, _ = mock_context_handler.call_args
    assert args[0] is not empty_llm_context_frame.context


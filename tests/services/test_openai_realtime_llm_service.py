import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call

from pipecat_extension.services.openai_realtime_llm_service import OpenAIRealtimeLLMServiceExt
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.openai.realtime import events


class TestOpenAIRealtimeLLMServiceExt:
    """Unit tests for OpenAIRealtimeLLMServiceExt."""

    def test_inherits_from_openai_realtime_llm_service(self):
        """Test that OpenAIRealtimeLLMServiceExt inherits from OpenAIRealtimeLLMService."""
        assert issubclass(OpenAIRealtimeLLMServiceExt, OpenAIRealtimeLLMService)

    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    def test_init_calls_parent_init(self, mock_register_event_handler, mock_parent_init):
        """Test that __init__ calls the parent class __init__ with correct arguments."""
        mock_parent_init.return_value = None
        args = (Mock(), Mock())
        kwargs = {"name": "test", "lorem": "ipsum"}

        service = OpenAIRealtimeLLMServiceExt(*args, **kwargs)

        mock_parent_init.assert_called_once_with(*args, **kwargs)
        assert service is not None

    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    def test_init_registers_event_handler(self, mock_register_event_handler, mock_parent_init):
        """Test that __init__ registers the after_function_call_output_sent and on_conversation_item_deleted event handlers."""
        mock_parent_init.return_value = None
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())

        # Verify _register_event_handler was called with both event names
        # after_function_call_output_sent should be registered with sync=True
        # on_conversation_item_deleted should be registered without sync parameter
        mock_register_event_handler.assert_has_calls([
            call("after_function_call_output_sent", sync=True),
            call("on_conversation_item_deleted")
        ], any_order=True)

    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    def test_register_event_handler_inherited(self, mock_register_event_handler, mock_parent_init):
        """Test that _register_event_handler method is inherited from parent class."""
        mock_parent_init.return_value = None
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())

        # Verify the method exists (inherited from parent)
        assert hasattr(service, "_register_event_handler")
        assert callable(service._register_event_handler)

    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    def test_init_with_various_arguments(self, mock_register_event_handler, mock_parent_init):
        """Test that __init__ works with various argument combinations."""
        mock_parent_init.return_value = None

        # Test with only positional arguments
        service1 = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        assert service1 is not None

        # Test with only keyword arguments
        service2 = OpenAIRealtimeLLMServiceExt(
            api_key="test_key",
            model="gpt-4",
            name="test_service"
        )
        assert service2 is not None

        # Test with mixed arguments
        service3 = OpenAIRealtimeLLMServiceExt(
            Mock(),
            api_key="test_key",
            model="gpt-4"
        )
        assert service3 is not None

    @pytest.mark.asyncio
    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    @patch.object(OpenAIRealtimeLLMService, "_handle_function_call_result", new_callable=AsyncMock)
    @patch.object(OpenAIRealtimeLLMService, "_call_event_handler", new_callable=AsyncMock)
    async def test_handle_function_call_result_calls_parent_and_event_handler(
        self,
        mock_call_event_handler,
        mock_handle_function_call_result,
        mock_register_event_handler,
        mock_parent_init,
    ):
        """Test that _handle_function_call_result calls parent method and event handler."""
        mock_parent_init.return_value = None

        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        frame = Mock()

        await service._handle_function_call_result(frame)

        mock_handle_function_call_result.assert_awaited_once_with(frame)
        mock_call_event_handler.assert_awaited_once_with("after_function_call_output_sent", frame)

    @pytest.mark.asyncio
    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    @patch.object(OpenAIRealtimeLLMService, "_call_event_handler", new_callable=AsyncMock)
    async def test_handle_evt_conversation_item_deleted_calls_event_handler(
        self,
        mock_call_event_handler,
        mock_register_event_handler,
        mock_parent_init,
    ):
        """Test that _handle_evt_conversation_item_deleted calls the event handler with item_id."""
        mock_parent_init.return_value = None
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        
        # Create a mock event with item_id
        mock_event = Mock()
        mock_event.item_id = "test_item_id"
        
        await service._handle_evt_conversation_item_deleted(mock_event)
        
        mock_call_event_handler.assert_awaited_once_with("on_conversation_item_deleted", "test_item_id")

    @pytest.mark.asyncio
    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    @patch("pipecat_extension.services.openai_realtime_llm_service.events.parse_server_event")
    @patch.object(OpenAIRealtimeLLMServiceExt, "_handle_evt_conversation_item_deleted", new_callable=AsyncMock)
    async def test_receive_task_handler_conversation_item_deleted(
        self,
        mock_handle_evt_conversation_item_deleted,
        mock_parse_server_event,
        mock_register_event_handler,
        mock_parent_init,
    ):
        """Test that _receive_task_handler handles conversation.item.deleted events."""
        mock_parent_init.return_value = None
        
        delete_event = events.ConversationItemDeleted(
            event_id="test_event_id",
            type="conversation.item.deleted",
            item_id="test_item_id",
        )
        mock_parse_server_event.return_value = delete_event
        
        # Create a mock websocket that yields one message then stops
        async def websocket_iter():
            yield "test_message"
            return  # Natural end of generator
        
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = lambda self: websocket_iter()
        
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        service._websocket = mock_websocket
        
        # Run the handler - it should process one message then stop
        try:
            await asyncio.wait_for(service._receive_task_handler(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        mock_parse_server_event.assert_called_once_with("test_message")
        mock_handle_evt_conversation_item_deleted.assert_awaited_once_with(delete_event)

    @pytest.mark.asyncio
    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    @patch("pipecat_extension.services.openai_realtime_llm_service.events.parse_server_event")
    @patch.object(OpenAIRealtimeLLMService, "_handle_evt_session_created", new_callable=AsyncMock)
    async def test_receive_task_handler_session_created(
        self,
        mock_handle_evt_session_created,
        mock_parse_server_event,
        mock_register_event_handler,
        mock_parent_init,
    ):
        """Test that _receive_task_handler handles session.created events."""
        mock_parent_init.return_value = None
        
        mock_event = Mock()
        mock_event.type = "session.created"
        mock_parse_server_event.return_value = mock_event
        
        async def websocket_iter():
            yield "test_message"
            return  # Natural end of generator
        
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = lambda self: websocket_iter()
        
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        service._websocket = mock_websocket
        
        try:
            await asyncio.wait_for(service._receive_task_handler(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        mock_handle_evt_session_created.assert_awaited_once_with(mock_event)

    @pytest.mark.asyncio
    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    @patch("pipecat_extension.services.openai_realtime_llm_service.events.parse_server_event")
    @patch.object(OpenAIRealtimeLLMService, "_maybe_handle_evt_retrieve_conversation_item_error", new_callable=AsyncMock)
    @patch.object(OpenAIRealtimeLLMService, "_handle_evt_error", new_callable=AsyncMock)
    async def test_receive_task_handler_error_event(
        self,
        mock_handle_evt_error,
        mock_maybe_handle_evt_retrieve_conversation_item_error,
        mock_parse_server_event,
        mock_register_event_handler,
        mock_parent_init,
    ):
        """Test that _receive_task_handler handles error events and exits on fatal errors."""
        mock_parent_init.return_value = None
        mock_maybe_handle_evt_retrieve_conversation_item_error.return_value = False
        
        mock_event = Mock()
        mock_event.type = "error"
        mock_parse_server_event.return_value = mock_event
        
        async def websocket_iter():
            yield "error_message"
            return  # Natural end of generator
        
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = lambda self: websocket_iter()
        
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        service._websocket = mock_websocket
        
        try:
            await asyncio.wait_for(service._receive_task_handler(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        mock_maybe_handle_evt_retrieve_conversation_item_error.assert_awaited_once_with(mock_event)
        mock_handle_evt_error.assert_awaited_once_with(mock_event)

    @pytest.mark.asyncio
    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    @patch("pipecat_extension.services.openai_realtime_llm_service.events.parse_server_event")
    @patch.object(OpenAIRealtimeLLMService, "_handle_evt_response_done", new_callable=AsyncMock)
    async def test_receive_task_handler_response_done(
        self,
        mock_handle_evt_response_done,
        mock_parse_server_event,
        mock_register_event_handler,
        mock_parent_init,
    ):
        """Test that _receive_task_handler handles response.done events."""
        mock_parent_init.return_value = None
        
        mock_event = Mock()
        mock_event.type = "response.done"
        mock_parse_server_event.return_value = mock_event
        
        async def websocket_iter():
            yield "test_message"
            return  # Natural end of generator
        
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = lambda self: websocket_iter()
        
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        service._websocket = mock_websocket
        
        try:
            await asyncio.wait_for(service._receive_task_handler(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        mock_handle_evt_response_done.assert_awaited_once_with(mock_event)

    @pytest.mark.asyncio
    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    @patch("pipecat_extension.services.openai_realtime_llm_service.events.parse_server_event")
    @patch.object(OpenAIRealtimeLLMService, "_handle_evt_session_updated", new_callable=AsyncMock)
    async def test_receive_task_handler_session_updated(
        self,
        mock_handle_evt_session_updated,
        mock_parse_server_event,
        mock_register_event_handler,
        mock_parent_init,
    ):
        """Test that _receive_task_handler handles session.updated events."""
        mock_parent_init.return_value = None
        
        mock_event = Mock()
        mock_event.type = "session.updated"
        mock_parse_server_event.return_value = mock_event
        
        async def websocket_iter():
            yield "test_message"
            return
        
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = lambda self: websocket_iter()
        
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        service._websocket = mock_websocket
        
        try:
            await asyncio.wait_for(service._receive_task_handler(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        mock_handle_evt_session_updated.assert_awaited_once_with(mock_event)

    @pytest.mark.asyncio
    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    @patch("pipecat_extension.services.openai_realtime_llm_service.events.parse_server_event")
    @patch.object(OpenAIRealtimeLLMService, "_handle_evt_audio_delta", new_callable=AsyncMock)
    async def test_receive_task_handler_audio_delta(
        self,
        mock_handle_evt_audio_delta,
        mock_parse_server_event,
        mock_register_event_handler,
        mock_parent_init,
    ):
        """Test that _receive_task_handler handles response.output_audio.delta events."""
        mock_parent_init.return_value = None
        
        mock_event = Mock()
        mock_event.type = "response.output_audio.delta"
        mock_parse_server_event.return_value = mock_event
        
        async def websocket_iter():
            yield "test_message"
            return
        
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = lambda self: websocket_iter()
        
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        service._websocket = mock_websocket
        
        try:
            await asyncio.wait_for(service._receive_task_handler(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        mock_handle_evt_audio_delta.assert_awaited_once_with(mock_event)

    @pytest.mark.asyncio
    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    @patch("pipecat_extension.services.openai_realtime_llm_service.events.parse_server_event")
    @patch.object(OpenAIRealtimeLLMService, "_handle_evt_conversation_item_added", new_callable=AsyncMock)
    async def test_receive_task_handler_conversation_item_added(
        self,
        mock_handle_evt_conversation_item_added,
        mock_parse_server_event,
        mock_register_event_handler,
        mock_parent_init,
    ):
        """Test that _receive_task_handler handles conversation.item.added events."""
        mock_parent_init.return_value = None
        
        mock_event = Mock()
        mock_event.type = "conversation.item.added"
        mock_parse_server_event.return_value = mock_event
        
        async def websocket_iter():
            yield "test_message"
            return
        
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = lambda self: websocket_iter()
        
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        service._websocket = mock_websocket
        
        try:
            await asyncio.wait_for(service._receive_task_handler(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        mock_handle_evt_conversation_item_added.assert_awaited_once_with(mock_event)

    @pytest.mark.asyncio
    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    @patch("pipecat_extension.services.openai_realtime_llm_service.events.parse_server_event")
    @patch.object(OpenAIRealtimeLLMService, "_handle_evt_conversation_item_done", new_callable=AsyncMock)
    async def test_receive_task_handler_conversation_item_done(
        self,
        mock_handle_evt_conversation_item_done,
        mock_parse_server_event,
        mock_register_event_handler,
        mock_parent_init,
    ):
        """Test that _receive_task_handler handles conversation.item.done events."""
        mock_parent_init.return_value = None
        
        mock_event = Mock()
        mock_event.type = "conversation.item.done"
        mock_parse_server_event.return_value = mock_event
        
        async def websocket_iter():
            yield "test_message"
            return
        
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = lambda self: websocket_iter()
        
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        service._websocket = mock_websocket
        
        try:
            await asyncio.wait_for(service._receive_task_handler(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        mock_handle_evt_conversation_item_done.assert_awaited_once_with(mock_event)

    @pytest.mark.asyncio
    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    @patch("pipecat_extension.services.openai_realtime_llm_service.events.parse_server_event")
    @patch.object(OpenAIRealtimeLLMService, "_handle_evt_speech_started", new_callable=AsyncMock)
    async def test_receive_task_handler_speech_started(
        self,
        mock_handle_evt_speech_started,
        mock_parse_server_event,
        mock_register_event_handler,
        mock_parent_init,
    ):
        """Test that _receive_task_handler handles input_audio_buffer.speech_started events."""
        mock_parent_init.return_value = None
        
        mock_event = Mock()
        mock_event.type = "input_audio_buffer.speech_started"
        mock_parse_server_event.return_value = mock_event
        
        async def websocket_iter():
            yield "test_message"
            return
        
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = lambda self: websocket_iter()
        
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        service._websocket = mock_websocket
        
        try:
            await asyncio.wait_for(service._receive_task_handler(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        mock_handle_evt_speech_started.assert_awaited_once_with(mock_event)

    @pytest.mark.asyncio
    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    @patch("pipecat_extension.services.openai_realtime_llm_service.events.parse_server_event")
    @patch.object(OpenAIRealtimeLLMService, "_handle_evt_text_delta", new_callable=AsyncMock)
    async def test_receive_task_handler_text_delta(
        self,
        mock_handle_evt_text_delta,
        mock_parse_server_event,
        mock_register_event_handler,
        mock_parent_init,
    ):
        """Test that _receive_task_handler handles response.output_text.delta events."""
        mock_parent_init.return_value = None
        
        mock_event = Mock()
        mock_event.type = "response.output_text.delta"
        mock_parse_server_event.return_value = mock_event
        
        async def websocket_iter():
            yield "test_message"
            return
        
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = lambda self: websocket_iter()
        
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        service._websocket = mock_websocket
        
        try:
            await asyncio.wait_for(service._receive_task_handler(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        mock_handle_evt_text_delta.assert_awaited_once_with(mock_event)

    @pytest.mark.asyncio
    @patch.object(OpenAIRealtimeLLMService, "__init__")
    @patch.object(OpenAIRealtimeLLMService, "_register_event_handler")
    @patch("pipecat_extension.services.openai_realtime_llm_service.events.parse_server_event")
    @patch.object(OpenAIRealtimeLLMService, "_handle_evt_function_call_arguments_done", new_callable=AsyncMock)
    async def test_receive_task_handler_function_call_arguments_done(
        self,
        mock_handle_evt_function_call_arguments_done,
        mock_parse_server_event,
        mock_register_event_handler,
        mock_parent_init,
    ):
        """Test that _receive_task_handler handles response.function_call_arguments.done events."""
        mock_parent_init.return_value = None
        
        mock_event = Mock()
        mock_event.type = "response.function_call_arguments.done"
        mock_parse_server_event.return_value = mock_event
        
        async def websocket_iter():
            yield "test_message"
            return
        
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = lambda self: websocket_iter()
        
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())
        service._websocket = mock_websocket
        
        try:
            await asyncio.wait_for(service._receive_task_handler(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        mock_handle_evt_function_call_arguments_done.assert_awaited_once_with(mock_event)


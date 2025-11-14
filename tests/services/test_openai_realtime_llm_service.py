import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from pipecat_extension.services.openai_realtime_llm_service import OpenAIRealtimeLLMServiceExt
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


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
        """Test that __init__ registers the after_function_call_output_sent event handler."""
        mock_parent_init.return_value = None
        service = OpenAIRealtimeLLMServiceExt(Mock(), Mock())

        # Verify _register_event_handler was called with the correct event name
        mock_register_event_handler.assert_called_once_with("after_function_call_output_sent")

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
        mock_call_event_handler.assert_awaited_once_with("after_function_call_output_sent")


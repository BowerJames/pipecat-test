import copy
from typing import Optional, Callable
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)

class ContextInjector(FrameProcessor):
    """
    A context injector for programmatically injecting messages into an LLMContext to guide the LLM's response.
    """
    def __init__(self):
        super().__init__()
        self._context_handler = None
        self._handler_is_mutable = False

    def register_context_handler(self, func: Optional[Callable] = None, *, mutable: bool = False):
        """Register a context handler function.
        
        This can be used in two ways:
            1. @processor.register_context_handler
            2. @processor.register_context_handler(mutable=True)
        
        Args:
            func: The function to decorate (when used without parentheses).
            mutable: If True, the handler receives the original context (mutable).
                    If False, the handler receives a deep copy of the context.
        
        Returns:
            A decorator function that stores the handler, or the decorated function.
        """
        def decorator(f):
            self._context_handler = f
            self._handler_is_mutable = mutable
            return f
        
        # Called as: @processor.register_context_handler
        if func is not None:
            self._context_handler = func
            self._handler_is_mutable = False
            return func
        
        # Called as: @processor.register_context_handler(mutable=True)
        return decorator

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMContextFrame):
            await self._handle_llm_context_frame(frame, direction)
        elif isinstance(frame, OpenAILLMContextFrame):
            await self._handle_llm_context_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _handle_llm_context_frame(self, frame: LLMContextFrame, direction: FrameDirection):
        """
        Handle LLMContextFrame processing. This method is called instead of push_frame
        when processing LLMContextFrame instances.
        """
        if self._context_handler is None:
            # No handler registered, push the original frame unchanged
            await self.push_frame(frame, direction)
        else:
            # Get the context to pass to the handler
            if self._handler_is_mutable:
                context = frame.context
            else:
                context = copy.deepcopy(frame.context)
            
            # Call the handler to get the modified context
            modified_context = self._context_handler(context)
            
            frame.context = modified_context
            
            # Push the new frame
            await self.push_frame(frame, direction)

    
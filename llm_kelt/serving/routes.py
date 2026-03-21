"""API routes for the llm-kelt proxy server."""

import time
import uuid
from typing import NoReturn

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from llm_infer.api import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    FinishReason,
    ModelInfo,
    ModelList,
    Role,
)
from llm_infer.client import (
    BackendError,
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
    ChatClient,
)

# Import directly to avoid circular import in llm_infer.serving.api
from llm_infer.serving.api.openai.streaming import stream_chat_completion

from ..inference.context import ContextBuilder


def _raise_http_for_backend_error(e: BackendError) -> NoReturn:
    """Convert backend error to appropriate HTTPException and raise it."""
    if isinstance(e, BackendUnavailableError):
        raise HTTPException(status_code=503, detail=f"Backend unavailable: {e}")
    if isinstance(e, BackendTimeoutError):
        raise HTTPException(status_code=504, detail=f"Backend timeout: {e}")
    if isinstance(e, BackendRequestError):
        raise HTTPException(status_code=502, detail=f"Backend error: {e}")
    raise HTTPException(status_code=500, detail=f"Unexpected backend error: {e}")


def _inject_facts_into_system(
    messages: list[ChatMessage],
    facts_prompt: str | None,
) -> list[ChatMessage]:
    """Inject facts into the system message.

    If no system message exists, creates one with just the facts.
    If a system message exists, prepends facts to it.
    """
    if not facts_prompt:
        return messages

    result = []
    has_system = any(m.role == Role.SYSTEM for m in messages)

    if not has_system:
        # Add system message with facts at the beginning
        result.append(ChatMessage(role=Role.SYSTEM, content=facts_prompt))
        result.extend(messages)
    else:
        # Prepend facts to existing system message
        for msg in messages:
            if msg.role == Role.SYSTEM:
                enhanced_content = f"{facts_prompt}\n\n{msg.content}"
                result.append(ChatMessage(role=Role.SYSTEM, content=enhanced_content))
            else:
                result.append(msg)

    return result


def _convert_to_backend_messages(
    messages: list[ChatMessage],
) -> tuple[list[dict[str, str]], str | None]:
    """Convert ChatMessage list to backend message dicts.

    Extracts system message separately as backends handle it differently.
    Returns (messages without system, system prompt or None).
    """
    system_prompt: str | None = None
    backend_messages: list[dict[str, str]] = []

    for msg in messages:
        # Extract string content (multimodal list content not supported for backend)
        content = msg.content if isinstance(msg.content, str) else ""
        if msg.role == Role.SYSTEM:
            system_prompt = content or None
        else:
            backend_messages.append({"role": msg.role.value, "content": content})

    return backend_messages, system_prompt


def _build_chat_response(
    model_name: str,
    content: str,
    usage: ChatCompletionUsage | None,
) -> ChatCompletionResponse:
    """Build OpenAI-compatible chat completion response.

    Args:
        model_name: Model name to include in response.
        content: Assistant's response content.
        usage: Token usage from backend, or None.

    Returns:
        Formatted ChatCompletionResponse.
    """
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role=Role.ASSISTANT, content=content),
                finish_reason=FinishReason.STOP,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        ),
    )


class _RouteHandlers:
    """Handlers for proxy API routes."""

    def __init__(
        self,
        model_name: str,
        llm_client: ChatClient,
        context_builder: ContextBuilder,
    ) -> None:
        self.model_name = model_name
        self.llm_client = llm_client
        self.context_builder = context_builder

    async def chat_completions(
        self, body: ChatCompletionRequest
    ) -> ChatCompletionResponse | StreamingResponse:
        """Handle chat completion with facts injection."""
        facts_prompt = self.context_builder.build_system_prompt("")
        enhanced_messages = _inject_facts_into_system(body.messages, facts_prompt)

        if body.stream:
            return await self._handle_streaming(enhanced_messages, body)

        backend_messages, system_prompt = _convert_to_backend_messages(enhanced_messages)
        content, usage = await self._call_backend(
            backend_messages, system_prompt, body.temperature, body.max_tokens
        )
        return _build_chat_response(self.model_name, content, usage)

    async def _handle_streaming(
        self,
        messages: list[ChatMessage],
        body: ChatCompletionRequest,
    ) -> StreamingResponse:
        """Handle streaming chat completion."""
        # Extract system prompt from messages for OpenAIClient
        system_prompt = None
        chat_messages: list[dict] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
            else:
                chat_messages.append({"role": msg.role.value, "content": msg.content})

        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        async def token_generator():
            async for token in self.llm_client.chat_stream_async(
                messages=chat_messages,
                system=system_prompt,
                temperature=body.temperature if body.temperature is not None else 0.7,
                max_tokens=body.max_tokens,
            ):
                yield token

        return StreamingResponse(
            stream_chat_completion(
                request_id=request_id,
                model=self.model_name,
                token_iterator=token_generator(),
                get_finish_reason=lambda: FinishReason.STOP,
            ),
            media_type="text/event-stream",
        )

    async def _call_backend(
        self,
        messages: list[dict[str, str]],
        system: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, ChatCompletionUsage | None]:
        """Call backend LLM with error translation."""
        try:
            response = await self.llm_client.chat_async(
                messages=messages,
                system=system,
                temperature=temperature if temperature is not None else 0.7,
                max_tokens=max_tokens,
            )
            return response.content, response.usage
        except BackendError as e:
            _raise_http_for_backend_error(e)

    async def list_models(self) -> ModelList:
        """List available models."""
        return ModelList(
            data=[
                ModelInfo(
                    id=self.model_name,
                    created=int(time.time()),
                    owned_by="llm-kelt",
                )
            ]
        )

    async def health(self) -> dict:
        """Health check endpoint."""
        return {"status": "ok"}


def create_router(
    model_name: str,
    llm_client: ChatClient,
    context_builder: ContextBuilder,
) -> APIRouter:
    """Create API router with chat completions endpoint.

    Args:
        model_name: Name of the backend model for responses.
        llm_client: LLM client for backend calls.
        context_builder: Context builder for facts injection.

    Returns:
        FastAPI router with OpenAI-compatible endpoints.
    """
    router = APIRouter()
    handlers = _RouteHandlers(model_name, llm_client, context_builder)

    router.post("/v1/chat/completions", response_model=None)(handlers.chat_completions)
    router.get("/v1/models", response_model=ModelList)(handlers.list_models)
    router.get("/health")(handlers.health)

    return router

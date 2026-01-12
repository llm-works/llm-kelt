"""API routes for the llm-learn proxy server."""

import time
import uuid

from fastapi import APIRouter, HTTPException
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

from ..inference.backends import (
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from ..inference.backends.base import Message
from ..inference.client import LLMClient
from ..inference.context import ContextBuilder


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
) -> tuple[list[Message], str | None]:
    """Convert ChatMessage list to backend Message list.

    Extracts system message separately as backends handle it differently.
    Returns (messages without system, system prompt or None).
    """
    system_prompt = None
    backend_messages = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            system_prompt = msg.content
        else:
            backend_messages.append(Message(role=msg.role.value, content=msg.content))

    return backend_messages, system_prompt


def create_router(
    model_name: str,
    llm_client: LLMClient,
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

    @router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(body: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion with facts injection."""
        # Build facts prompt from user's profile (empty base = just facts section)
        facts_prompt = context_builder.build_system_prompt("")

        # Inject facts into messages
        enhanced_messages = _inject_facts_into_system(body.messages, facts_prompt)

        # Convert to backend format
        backend_messages, system_prompt = _convert_to_backend_messages(enhanced_messages)

        # Call backend LLM
        try:
            response = await llm_client.chat_full(
                messages=backend_messages,
                system=system_prompt,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
        except BackendUnavailableError as e:
            raise HTTPException(status_code=503, detail=f"Backend unavailable: {e}")
        except BackendTimeoutError as e:
            raise HTTPException(status_code=504, detail=f"Backend timeout: {e}")
        except BackendRequestError as e:
            raise HTTPException(status_code=502, detail=f"Backend error: {e.detail}")

        # Build response
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role=Role.ASSISTANT, content=response.content),
                    finish_reason=FinishReason.STOP,
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=response.usage.get("prompt_tokens", 0) if response.usage else 0,
                completion_tokens=response.usage.get("completion_tokens", 0)
                if response.usage
                else 0,
                total_tokens=response.usage.get("total_tokens", 0) if response.usage else 0,
            ),
        )

    @router.get("/v1/models", response_model=ModelList)
    async def list_models() -> ModelList:
        """List available models."""
        return ModelList(
            data=[
                ModelInfo(
                    id=model_name,
                    created=int(time.time()),
                    owned_by="llm-learn",
                )
            ]
        )

    @router.get("/health")
    async def health() -> dict:
        """Health check endpoint."""
        return {"status": "ok"}

    return router

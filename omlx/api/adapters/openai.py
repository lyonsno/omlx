# SPDX-License-Identifier: Apache-2.0
"""
OpenAI API adapter for oMLX.

This adapter handles conversion between OpenAI API format and the internal
request/response format used by the inference engine.
"""

import json
import time
import uuid
from typing import Any, List, Optional

from .base import (
    BaseAdapter,
    InternalMessage,
    InternalRequest,
    InternalResponse,
    StreamChunk,
)
from ..openai_models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)
from ..thinking import ThinkingParser, extract_thinking, strip_think_tags
from ..utils import clean_special_tokens, extract_text_content
from ..tool_calling import convert_tools_for_template


class OpenAIAdapter(BaseAdapter):
    """
    Adapter for OpenAI API format.

    Handles conversion between OpenAI chat completion requests/responses
    and the internal format used by the inference engine.
    """

    def __init__(self):
        # Track parser state per request so raw <think> markup split across
        # chunks is partitioned consistently throughout the stream.
        self._thinking_parsers: dict[int, ThinkingParser] = {}
        self._request_payload_keys: dict[int, str] = {}
        self._active_request_keys_by_payload: dict[str, set[int]] = {}
        self._stream_request_object_ids: dict[int, int] = {}
        self._ended_request_payload_keys: dict[int, str] = {}
        self._stream_touch_seq = 0
        self._stream_last_touched: dict[int, int] = {}
        self._stream_birth_seq = 0
        self._stream_birth_order: dict[int, int] = {}

    @staticmethod
    def _stream_state_key(request: ChatCompletionRequest) -> int:
        """Build an instance-local key for a request object."""
        return id(request)

    def _stream_payload_key(self, request: ChatCompletionRequest) -> str:
        """Build a deterministic payload key for logical stream continuity."""
        request_key = self._stream_state_key(request)
        request_object_id = id(request)
        if self._stream_request_object_ids.get(request_key) == request_object_id:
            payload_key = self._request_payload_keys.get(request_key)
            if payload_key is not None:
                return payload_key

        payload_key = self._ended_request_payload_keys.get(request_object_id)
        if payload_key is not None:
            return payload_key

        return request.model_dump_json(exclude_none=False)

    def _register_stream_key(
        self,
        request_key: int,
        payload_key: str,
        parser: ThinkingParser,
        *,
        birth_order: int | None = None,
        request_object_id: int | None = None,
    ) -> None:
        self._thinking_parsers[request_key] = parser
        self._request_payload_keys[request_key] = payload_key
        self._active_request_keys_by_payload.setdefault(payload_key, set()).add(request_key)
        if request_object_id is None:
            request_object_id = request_key
        self._stream_request_object_ids[request_key] = request_object_id
        if birth_order is None:
            self._stream_birth_seq += 1
            birth_order = self._stream_birth_seq
        self._stream_birth_order[request_key] = birth_order
        self._touch_stream_key(request_key)

    def _touch_stream_key(self, request_key: int) -> None:
        self._stream_touch_seq += 1
        self._stream_last_touched[request_key] = self._stream_touch_seq

    def _allocate_collision_key(self, request_key: int) -> int:
        candidate = request_key
        while candidate in self._thinking_parsers or candidate in self._request_payload_keys:
            candidate += 1
        return candidate

    def _deregister_stream_key(self, request_key: int) -> None:
        self._thinking_parsers.pop(request_key, None)
        self._stream_last_touched.pop(request_key, None)
        self._stream_birth_order.pop(request_key, None)
        self._stream_request_object_ids.pop(request_key, None)
        payload_key = self._request_payload_keys.pop(request_key, None)
        if not payload_key:
            return
        keys = self._active_request_keys_by_payload.get(payload_key)
        if not keys:
            return

        keys.discard(request_key)
        if not keys:
            self._active_request_keys_by_payload.pop(payload_key, None)

    def _adopt_equivalent_stream(
        self,
        request_key: int,
        payload_key: str,
        request_object_id: int,
    ) -> tuple[ThinkingParser | None, int | None]:
        active_keys = self._active_request_keys_by_payload.get(payload_key)
        if not active_keys or len(active_keys) != 1:
            return None, None

        existing_key = next(iter(active_keys))
        parser = self._thinking_parsers.get(existing_key)
        if parser is None:
            self._deregister_stream_key(existing_key)
            return None, None

        if existing_key == request_key:
            self._stream_request_object_ids[existing_key] = request_object_id
            return parser, existing_key

        # If request_key is already occupied by a different payload, keep the
        # parser on its existing key to avoid cross-stream key-collision clobber.
        occupied_payload = self._request_payload_keys.get(request_key)
        if occupied_payload and occupied_payload != payload_key:
            return parser, existing_key

        # Move ownership so a rehydrated equivalent request can continue the
        # same logical stream without object-identity coupling.
        birth_order = self._stream_birth_order.get(existing_key)
        self._deregister_stream_key(existing_key)
        self._register_stream_key(
            request_key,
            payload_key,
            parser,
            birth_order=birth_order,
            request_object_id=request_object_id,
        )
        return parser, request_key

    @property
    def name(self) -> str:
        return "openai"

    def parse_request(self, request: ChatCompletionRequest) -> InternalRequest:
        """
        Convert an OpenAI ChatCompletionRequest to internal format.

        Args:
            request: OpenAI chat completion request.

        Returns:
            InternalRequest in unified format.
        """
        # Extract text content from messages
        messages = extract_text_content(request.messages)

        # Convert to internal messages
        internal_messages = [
            InternalMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
            )
            for msg in messages
        ]

        # Convert tools if provided
        tools = None
        if request.tools:
            tools = convert_tools_for_template(request.tools)

        return InternalRequest(
            messages=internal_messages,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature if request.temperature is not None else 1.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            min_p=request.min_p if request.min_p is not None else 0.0,
            presence_penalty=request.presence_penalty if request.presence_penalty is not None else 0.0,
            frequency_penalty=request.frequency_penalty if request.frequency_penalty is not None else 0.0,
            stream=request.stream or False,
            stop=request.stop if isinstance(request.stop, list) else (
                [request.stop] if request.stop else None
            ),
            tools=tools,
            tool_choice=request.tool_choice,
            response_format=request.response_format,
            model=request.model,
            request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        )

    def format_response(
        self,
        response: InternalResponse,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """
        Convert an internal response to OpenAI ChatCompletionResponse.

        Args:
            response: Internal response object.
            request: Original OpenAI request.

        Returns:
            ChatCompletionResponse in OpenAI format.
        """
        # Separate thinking from content
        raw_text = clean_special_tokens(response.text) if response.text else ""
        extracted_thinking, regular_content = extract_thinking(raw_text)
        thinking_content = strip_think_tags(
            response.reasoning_content or extracted_thinking,
            trim=True,
        )
        content = regular_content.strip() if regular_content else None

        # Determine finish reason
        finish_reason = (
            "tool_calls" if response.tool_calls else response.finish_reason
        )

        return ChatCompletionResponse(
            id=response.request_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(
                        content=content,
                        reasoning_content=thinking_content or None,
                        tool_calls=response.tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.prompt_tokens + response.completion_tokens,
            ),
        )

    def format_stream_chunk(
        self,
        chunk: StreamChunk,
        request: ChatCompletionRequest,
    ) -> str:
        """
        Format a streaming chunk for SSE output in OpenAI format.

        Args:
            chunk: The stream chunk to format.
            request: Original OpenAI request.

        Returns:
            SSE-formatted string.
        """
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        request_object_id = id(request)
        request_key = self._stream_state_key(request)
        payload_key = self._stream_payload_key(request)

        content = chunk.text or ""
        extracted_thinking = ""
        transient_parser = False
        parser = self._thinking_parsers.get(request_key)
        if parser is not None and self._request_payload_keys.get(request_key) != payload_key:
            parser = None
        if chunk.is_first:
            existing_payload = self._request_payload_keys.get(request_key)
            if existing_payload and existing_payload != payload_key:
                request_key = self._allocate_collision_key(request_key)
            else:
                self._deregister_stream_key(request_key)
            parser = ThinkingParser()
            self._register_stream_key(
                request_key,
                payload_key,
                parser,
                request_object_id=request_object_id,
            )
        elif parser is None:
            parser, adopted_key = self._adopt_equivalent_stream(
                request_key,
                payload_key,
                request_object_id,
            )
            if adopted_key is not None:
                request_key = adopted_key
            if parser is None:
                parser = ThinkingParser()
                if "<" in content or chunk.reasoning_content or chunk.is_last:
                    self._register_stream_key(
                        request_key,
                        payload_key,
                        parser,
                        request_object_id=request_object_id,
                    )
                else:
                    transient_parser = True

        if content:
            extracted_thinking, content = parser.feed(content)
            if request_key in self._thinking_parsers:
                self._touch_stream_key(request_key)

        if chunk.is_last:
            tail_thinking, tail_content = parser.finish()
            extracted_thinking += tail_thinking
            content += tail_content
            if not transient_parser:
                if self._stream_request_object_ids.get(request_key) == request_object_id:
                    payload_key = self._request_payload_keys.get(request_key)
                    if payload_key is not None:
                        self._ended_request_payload_keys[request_object_id] = payload_key
                self._deregister_stream_key(request_key)

        reasoning_content = strip_think_tags(
            chunk.reasoning_content or extracted_thinking,
            trim=False,
        )

        delta = ChatCompletionChunkDelta(
            content=content if content else None,
            reasoning_content=reasoning_content or None,
            tool_calls=chunk.tool_call_delta,
        )

        # Add role on first chunk
        if chunk.is_first:
            delta.role = "assistant"

        response = ChatCompletionChunk(
            id=request_id,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=delta,
                    finish_reason=chunk.finish_reason,
                )
            ],
        )

        # Add usage on last chunk if available
        if chunk.is_last and (chunk.prompt_tokens > 0 or chunk.completion_tokens > 0):
            response.usage = Usage(
                prompt_tokens=chunk.prompt_tokens,
                completion_tokens=chunk.completion_tokens,
                total_tokens=chunk.prompt_tokens + chunk.completion_tokens,
            )

        return f"data: {response.model_dump_json(exclude_none=True)}\n\n"

    def format_stream_end(self, request: ChatCompletionRequest) -> str:
        """
        Format the stream end marker for OpenAI format.

        Args:
            request: Original OpenAI request.

        Returns:
            SSE-formatted end marker.
        """
        request_object_id = id(request)
        request_key = self._stream_state_key(request)
        if (
            request_key in self._thinking_parsers
            and self._stream_request_object_ids.get(request_key) == request_object_id
        ):
            self._deregister_stream_key(request_key)
            self._ended_request_payload_keys.pop(request_object_id, None)
            return "data: [DONE]\n\n"

        # Fallback for equivalent-object end calls. When multiple live streams
        # share the same payload, retire the oldest surviving logical stream so
        # equivalent object lifecycles resolve deterministically.
        payload_key = self._ended_request_payload_keys.pop(request_object_id, None)
        if payload_key is None:
            payload_key = self._stream_payload_key(request)
        active_keys = self._active_request_keys_by_payload.get(payload_key)
        if active_keys:
            if len(active_keys) == 1:
                cleanup_key = next(iter(active_keys))
            else:
                cleanup_key = min(
                    active_keys,
                    key=lambda k: self._stream_birth_order.get(k, float("inf")),
                )
            self._deregister_stream_key(cleanup_key)
        return "data: [DONE]\n\n"

    def create_error_response(
        self,
        error: str,
        error_type: str = "server_error",
        status_code: int = 500,
    ) -> dict:
        """
        Create an error response in OpenAI format.

        Args:
            error: Error message.
            error_type: Type of error.
            status_code: HTTP status code.

        Returns:
            Error response dict.
        """
        return {
            "error": {
                "message": error,
                "type": error_type,
                "param": None,
                "code": status_code,
            }
        }

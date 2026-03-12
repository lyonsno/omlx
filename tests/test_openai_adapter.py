# SPDX-License-Identifier: Apache-2.0
"""
Tests for OpenAI API adapter.

Tests the OpenAIAdapter class and base adapter data structures for converting
between OpenAI API format and internal oMLX format.
"""

import json
import pytest

from omlx.api.adapters.base import (
    BaseAdapter,
    InternalMessage,
    InternalRequest,
    InternalResponse,
    StreamChunk,
)
from omlx.api.adapters.openai import OpenAIAdapter
from omlx.api.openai_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ContentPart,
    Message,
    ToolDefinition,
)


class TestInternalDataClasses:
    """Tests for internal data class structures."""

    # =========================================================================
    # InternalMessage Tests
    # =========================================================================

    def test_internal_message_basic(self):
        """Test InternalMessage with required fields only."""
        msg = InternalMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_internal_message_with_optional_fields(self):
        """Test InternalMessage with all optional fields."""
        tool_calls = [{"id": "call_123", "function": {"name": "test"}}]
        msg = InternalMessage(
            role="assistant",
            content="Response",
            name="assistant_name",
            tool_calls=tool_calls,
            tool_call_id="call_456",
        )
        assert msg.role == "assistant"
        assert msg.content == "Response"
        assert msg.name == "assistant_name"
        assert msg.tool_calls == tool_calls
        assert msg.tool_call_id == "call_456"

    def test_internal_message_roles(self):
        """Test InternalMessage with different roles."""
        for role in ["user", "assistant", "system", "tool"]:
            msg = InternalMessage(role=role, content=f"Content for {role}")
            assert msg.role == role

    # =========================================================================
    # InternalRequest Tests
    # =========================================================================

    def test_internal_request_minimal(self):
        """Test InternalRequest with minimal required fields."""
        messages = [InternalMessage(role="user", content="Hello")]
        req = InternalRequest(messages=messages)

        assert req.messages == messages
        assert req.max_tokens == 2048
        assert req.temperature == 1.0
        assert req.top_p == 1.0
        assert req.top_k == 0
        assert req.stream is False
        assert req.stop is None
        assert req.stop_token_ids is None
        assert req.tools is None
        assert req.tool_choice is None
        assert req.response_format is None
        assert req.model is None
        assert req.request_id is None

    def test_internal_request_with_all_fields(self):
        """Test InternalRequest with all fields set."""
        messages = [
            InternalMessage(role="system", content="Be helpful"),
            InternalMessage(role="user", content="Hello"),
        ]
        tools = [{"type": "function", "function": {"name": "test"}}]

        req = InternalRequest(
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            stream=True,
            stop=["STOP", "END"],
            stop_token_ids=[1, 2],
            tools=tools,
            tool_choice="auto",
            response_format={"type": "json_object"},
            model="test-model",
            request_id="req-123",
        )

        assert len(req.messages) == 2
        assert req.max_tokens == 1024
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.top_k == 50
        assert req.stream is True
        assert req.stop == ["STOP", "END"]
        assert req.stop_token_ids == [1, 2]
        assert req.tools == tools
        assert req.tool_choice == "auto"
        assert req.response_format == {"type": "json_object"}
        assert req.model == "test-model"
        assert req.request_id == "req-123"

    # =========================================================================
    # InternalResponse Tests
    # =========================================================================

    def test_internal_response_minimal(self):
        """Test InternalResponse with minimal fields."""
        resp = InternalResponse(text="Hello!")

        assert resp.text == "Hello!"
        assert resp.finish_reason is None
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0
        assert resp.tool_calls is None
        assert resp.request_id is None
        assert resp.model is None

    def test_internal_response_with_all_fields(self):
        """Test InternalResponse with all fields set."""
        tool_calls = [{"id": "call_123", "function": {"name": "get_weather"}}]

        resp = InternalResponse(
            text="Here is the weather.",
            finish_reason="tool_calls",
            prompt_tokens=100,
            completion_tokens=50,
            tool_calls=tool_calls,
            request_id="chatcmpl-abc123",
            model="test-model",
        )

        assert resp.text == "Here is the weather."
        assert resp.finish_reason == "tool_calls"
        assert resp.prompt_tokens == 100
        assert resp.completion_tokens == 50
        assert resp.tool_calls == tool_calls
        assert resp.request_id == "chatcmpl-abc123"
        assert resp.model == "test-model"

    # =========================================================================
    # StreamChunk Tests
    # =========================================================================

    def test_stream_chunk_default(self):
        """Test StreamChunk with defaults."""
        chunk = StreamChunk()

        assert chunk.text == ""
        assert chunk.finish_reason is None
        assert chunk.tool_call_delta is None
        assert chunk.is_first is False
        assert chunk.is_last is False
        assert chunk.prompt_tokens == 0
        assert chunk.completion_tokens == 0

    def test_stream_chunk_first_chunk(self):
        """Test StreamChunk as first chunk."""
        chunk = StreamChunk(text="Hello", is_first=True)

        assert chunk.text == "Hello"
        assert chunk.is_first is True
        assert chunk.is_last is False

    def test_stream_chunk_last_chunk(self):
        """Test StreamChunk as last chunk."""
        chunk = StreamChunk(
            text="",
            finish_reason="stop",
            is_last=True,
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert chunk.text == ""
        assert chunk.finish_reason == "stop"
        assert chunk.is_last is True
        assert chunk.prompt_tokens == 100
        assert chunk.completion_tokens == 50

    def test_stream_chunk_with_tool_call_delta(self):
        """Test StreamChunk with tool call delta."""
        tool_delta = {"name": "get_weather", "arguments": '{"location":'}
        chunk = StreamChunk(tool_call_delta=tool_delta)

        assert chunk.tool_call_delta == tool_delta


class TestOpenAIAdapter:
    """Tests for OpenAIAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create OpenAIAdapter instance."""
        return OpenAIAdapter()

    # =========================================================================
    # Adapter Name Tests
    # =========================================================================

    def test_adapter_name(self, adapter):
        """Test adapter name property."""
        assert adapter.name == "openai"

    def test_adapter_inherits_base(self, adapter):
        """Test adapter inherits from BaseAdapter."""
        assert isinstance(adapter, BaseAdapter)

    # =========================================================================
    # parse_request Tests
    # =========================================================================

    def test_parse_request_simple_message(self, adapter):
        """Test parsing a simple chat request."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                Message(role="user", content="Hello"),
            ],
        )

        internal = adapter.parse_request(request)

        assert isinstance(internal, InternalRequest)
        assert len(internal.messages) == 1
        assert internal.messages[0].role == "user"
        assert internal.messages[0].content == "Hello"
        assert internal.model == "test-model"

    def test_parse_request_multiple_messages(self, adapter):
        """Test parsing request with multiple messages."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                Message(role="system", content="Be helpful"),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
                Message(role="user", content="How are you?"),
            ],
        )

        internal = adapter.parse_request(request)

        assert len(internal.messages) == 4
        assert internal.messages[0].role == "system"
        assert internal.messages[1].role == "user"
        assert internal.messages[2].role == "assistant"
        assert internal.messages[3].role == "user"

    def test_parse_request_with_temperature(self, adapter):
        """Test parsing request with temperature."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.5,
        )

        internal = adapter.parse_request(request)

        assert internal.temperature == 0.5

    def test_parse_request_with_zero_temperature(self, adapter):
        """Test parsing request with zero temperature."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.0,
        )

        internal = adapter.parse_request(request)

        assert internal.temperature == 0.0

    def test_parse_request_default_temperature(self, adapter):
        """Test parsing request without temperature uses default."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        internal = adapter.parse_request(request)

        assert internal.temperature == 1.0

    def test_parse_request_with_top_p(self, adapter):
        """Test parsing request with top_p."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            top_p=0.9,
        )

        internal = adapter.parse_request(request)

        assert internal.top_p == 0.9

    def test_parse_request_with_min_p(self, adapter):
        """Test parsing request with min_p."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            min_p=0.1,
        )

        internal = adapter.parse_request(request)

        assert internal.min_p == 0.1

    def test_parse_request_with_presence_penalty(self, adapter):
        """Test parsing request with presence_penalty."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            presence_penalty=0.5,
        )

        internal = adapter.parse_request(request)

        assert internal.presence_penalty == 0.5

    def test_parse_request_default_min_p_and_presence_penalty(self, adapter):
        """Test default min_p and presence_penalty values."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        internal = adapter.parse_request(request)

        assert internal.min_p == 0.0
        assert internal.presence_penalty == 0.0

    def test_parse_request_with_max_tokens(self, adapter):
        """Test parsing request with max_tokens."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            max_tokens=500,
        )

        internal = adapter.parse_request(request)

        assert internal.max_tokens == 500

    def test_parse_request_default_max_tokens(self, adapter):
        """Test parsing request without max_tokens uses default."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        internal = adapter.parse_request(request)

        assert internal.max_tokens == 2048

    def test_parse_request_with_stream_true(self, adapter):
        """Test parsing request with stream=True."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            stream=True,
        )

        internal = adapter.parse_request(request)

        assert internal.stream is True

    def test_parse_request_with_stream_false(self, adapter):
        """Test parsing request with stream=False."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            stream=False,
        )

        internal = adapter.parse_request(request)

        assert internal.stream is False

    def test_parse_request_with_stop_list(self, adapter):
        """Test parsing request with stop sequences as list."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            stop=["STOP", "END"],
        )

        internal = adapter.parse_request(request)

        assert internal.stop == ["STOP", "END"]

    def test_parse_request_with_tools(self, adapter):
        """Test parsing request with tools."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            tools=[
                ToolDefinition(
                    type="function",
                    function={
                        "name": "get_weather",
                        "description": "Get weather info",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                        },
                    },
                )
            ],
        )

        internal = adapter.parse_request(request)

        assert internal.tools is not None
        assert len(internal.tools) == 1
        assert internal.tools[0]["function"]["name"] == "get_weather"

    def test_parse_request_with_tool_choice(self, adapter):
        """Test parsing request with tool_choice."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            tool_choice="auto",
        )

        internal = adapter.parse_request(request)

        assert internal.tool_choice == "auto"

    def test_parse_request_with_response_format(self, adapter):
        """Test parsing request with response_format."""
        from omlx.api.openai_models import ResponseFormat

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            response_format=ResponseFormat(type="json_object"),
        )

        internal = adapter.parse_request(request)

        # response_format is passed through (can be dict or ResponseFormat)
        assert internal.response_format is not None

    def test_parse_request_generates_request_id(self, adapter):
        """Test that parse_request generates a request ID."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        internal = adapter.parse_request(request)

        assert internal.request_id is not None
        assert internal.request_id.startswith("chatcmpl-")

    def test_parse_request_with_content_array(self, adapter):
        """Test parsing request with content array."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                Message(
                    role="user",
                    content=[{"type": "text", "text": "Hello world"}],
                ),
            ],
        )

        internal = adapter.parse_request(request)

        assert len(internal.messages) == 1
        # Content should be extracted
        assert "Hello world" in internal.messages[0].content

    # =========================================================================
    # format_response Tests
    # =========================================================================

    def test_format_response_basic(self, adapter):
        """Test formatting a basic response."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        response = InternalResponse(
            text="Hi there!",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            request_id="chatcmpl-abc123",
        )

        result = adapter.format_response(response, request)

        assert isinstance(result, ChatCompletionResponse)
        assert result.model == "test-model"
        assert result.object == "chat.completion"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hi there!"
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_format_response_with_tool_calls(self, adapter):
        """Test formatting response with tool calls."""
        from omlx.api.openai_models import FunctionCall, ToolCall

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        tool_calls = [
            ToolCall(
                id="call_abc123",
                type="function",
                function=FunctionCall(
                    name="get_weather",
                    arguments='{"location": "Tokyo"}',
                ),
            )
        ]
        response = InternalResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=tool_calls,
        )

        result = adapter.format_response(response, request)

        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls == tool_calls

    def test_format_response_empty_text(self, adapter):
        """Test formatting response with empty text."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        response = InternalResponse(text="")

        result = adapter.format_response(response, request)

        # Empty text should result in None content
        assert result.choices[0].message.content is None

    def test_format_response_with_special_tokens(self, adapter):
        """Test formatting response cleans special tokens."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        response = InternalResponse(
            text="Hello<|im_end|>",
            finish_reason="stop",
        )

        result = adapter.format_response(response, request)

        assert result.choices[0].message.content == "Hello"

    def test_format_response_preserves_request_id(self, adapter):
        """Test formatting response preserves request ID."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        response = InternalResponse(
            text="Hi!",
            request_id="chatcmpl-custom123",
        )

        result = adapter.format_response(response, request)

        assert result.id == "chatcmpl-custom123"

    def test_format_response_uses_explicit_reasoning_content(self, adapter):
        """Explicit reasoning_content should be preserved without think tags."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        response = InternalResponse(
            text="Final answer",
            reasoning_content="<think>internal reasoning</think>",
            finish_reason="stop",
        )

        result = adapter.format_response(response, request)

        assert result.choices[0].message.content == "Final answer"
        assert result.choices[0].message.reasoning_content == "internal reasoning"

    def test_format_response_extracts_malformed_thinking_from_text(self, adapter):
        """Malformed raw text should still partition into reasoning and content."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        response = InternalResponse(
            text="<think>\n<think>reasoning details</think>Final answer",
            finish_reason="stop",
        )

        result = adapter.format_response(response, request)

        assert result.choices[0].message.reasoning_content == "reasoning details"
        assert result.choices[0].message.content == "Final answer"

    # =========================================================================
    # format_stream_chunk Tests
    # =========================================================================

    def test_format_stream_chunk_basic(self, adapter):
        """Test formatting a basic stream chunk."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        chunk = StreamChunk(text="Hello")

        result = adapter.format_stream_chunk(chunk, request)

        assert result.startswith("data: ")
        assert result.endswith("\n\n")

        # Parse the JSON
        json_str = result[6:-2]  # Remove "data: " prefix and "\n\n" suffix
        data = json.loads(json_str)

        assert data["object"] == "chat.completion.chunk"
        assert data["model"] == "test-model"
        assert data["choices"][0]["delta"]["content"] == "Hello"

    def test_format_stream_chunk_first(self, adapter):
        """Test formatting first stream chunk includes role."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        chunk = StreamChunk(text="Hi", is_first=True)

        result = adapter.format_stream_chunk(chunk, request)

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["choices"][0]["delta"]["role"] == "assistant"

    def test_format_stream_chunk_last_with_finish_reason(self, adapter):
        """Test formatting last stream chunk includes finish reason."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        chunk = StreamChunk(
            text="",
            finish_reason="stop",
            is_last=True,
            prompt_tokens=10,
            completion_tokens=5,
        )

        result = adapter.format_stream_chunk(chunk, request)

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["choices"][0]["finish_reason"] == "stop"
        # Note: ChatCompletionChunk may not have usage field in all implementations

    def test_format_stream_chunk_with_tool_call_delta(self, adapter):
        """Test formatting stream chunk with tool call delta."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        # tool_call_delta should be a list for OpenAI format
        tool_delta = [{"index": 0, "function": {"name": "get_weather"}}]
        chunk = StreamChunk(tool_call_delta=tool_delta)

        result = adapter.format_stream_chunk(chunk, request)

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["choices"][0]["delta"]["tool_calls"] == tool_delta

    def test_format_stream_chunk_strips_think_tags_from_standard_fields(self, adapter):
        """Streaming chunks should never expose raw think markup."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        chunk = StreamChunk(text="<think>reasoning</think>Answer")

        result = adapter.format_stream_chunk(chunk, request)

        json_str = result[6:-2]
        data = json.loads(json_str)
        delta = data["choices"][0]["delta"]

        assert delta["reasoning_content"] == "reasoning"
        assert delta["content"] == "Answer"
        assert "<think>" not in json.dumps(delta)
        assert "</think>" not in json.dumps(delta)

    def test_format_stream_chunk_preserves_thinking_state_across_chunks(self, adapter):
        """Split think tags across chunks should not leak reasoning into content."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        first = adapter.format_stream_chunk(
            StreamChunk(text="<think>rea", is_first=True),
            request,
        )
        second = adapter.format_stream_chunk(
            StreamChunk(text="soning</think>Answer", is_last=True),
            request,
        )

        first_data = json.loads(first[6:-2])
        second_data = json.loads(second[6:-2])
        first_delta = first_data["choices"][0]["delta"]
        second_delta = second_data["choices"][0]["delta"]

        assert first_delta["reasoning_content"] == "rea"
        assert "content" not in first_delta
        assert second_delta["reasoning_content"] == "soning"
        assert second_delta["content"] == "Answer"
        assert "<think>" not in json.dumps([first_delta, second_delta])
        assert "</think>" not in json.dumps([first_delta, second_delta])

    def test_format_stream_chunk_isolates_equivalent_requests_when_streams_interleave(self, adapter):
        """Equivalent request payloads should not cross-contaminate parser state."""
        request_a = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        request_b = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        # Stream A starts with an incomplete tag so parser state must keep a
        # buffered partial tag specific to A.
        a_first = adapter.format_stream_chunk(
            StreamChunk(text="<thi", is_first=True),
            request_a,
        )
        # Stream B runs to completion while A is still mid-tag.
        b_first = adapter.format_stream_chunk(
            StreamChunk(text="<think>B</think>", is_first=True),
            request_b,
        )
        b_last = adapter.format_stream_chunk(
            StreamChunk(text="B!", is_last=True),
            request_b,
        )
        # Stream A resumes; it should recover its buffered "<thi" and parse
        # "<think>A1</think>A!" correctly.
        a_last = adapter.format_stream_chunk(
            StreamChunk(text="nk>A1</think>A!", is_last=True),
            request_a,
        )

        a_first_delta = json.loads(a_first[6:-2])["choices"][0]["delta"]
        b_first_delta = json.loads(b_first[6:-2])["choices"][0]["delta"]
        b_last_delta = json.loads(b_last[6:-2])["choices"][0]["delta"]
        a_last_delta = json.loads(a_last[6:-2])["choices"][0]["delta"]

        assert a_first_delta["role"] == "assistant"
        assert "reasoning_content" not in a_first_delta
        assert "content" not in a_first_delta
        assert b_first_delta["reasoning_content"] == "B"
        assert "content" not in b_first_delta
        assert b_last_delta["content"] == "B!"
        assert "reasoning_content" not in b_last_delta
        assert a_last_delta["reasoning_content"] == "A1"
        assert a_last_delta["content"] == "A!"

    def test_format_stream_chunk_continues_logical_stream_across_equivalent_request_object(self, adapter):
        """Logical stream continuity should not require request object identity."""
        request1 = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        request2 = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        first = adapter.format_stream_chunk(
            StreamChunk(text="<think>rea", is_first=True),
            request1,
        )
        second = adapter.format_stream_chunk(
            StreamChunk(text="soning</think>Answer", is_last=True),
            request2,
        )

        first_delta = json.loads(first[6:-2])["choices"][0]["delta"]
        second_delta = json.loads(second[6:-2])["choices"][0]["delta"]

        assert first_delta["reasoning_content"] == "rea"
        assert "content" not in first_delta
        assert second_delta["reasoning_content"] == "soning"
        assert second_delta["content"] == "Answer"

    @staticmethod
    def _run_mirrored_end_order_interleaving(adapter):
        """Run mirrored interleaving schedule and return parsed deltas."""
        request_a = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        request_b = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        # A starts with a partial tag (buffered parser state).
        a_first = adapter.format_stream_chunk(
            StreamChunk(text="<thi", is_first=True),
            request_a,
        )
        # B starts in open-thinking mode without closing in first chunk.
        b_first = adapter.format_stream_chunk(
            StreamChunk(text="<think>B", is_first=True),
            request_b,
        )
        # Mirrored end order: A ends before B.
        a_last = adapter.format_stream_chunk(
            StreamChunk(text="nk>A1</think>A!", is_last=True),
            request_a,
        )
        b_last = adapter.format_stream_chunk(
            StreamChunk(text="2</think>B!", is_last=True),
            request_b,
        )

        a_first_delta = json.loads(a_first[6:-2])["choices"][0]["delta"]
        b_first_delta = json.loads(b_first[6:-2])["choices"][0]["delta"]
        a_last_delta = json.loads(a_last[6:-2])["choices"][0]["delta"]
        b_last_delta = json.loads(b_last[6:-2])["choices"][0]["delta"]

        return a_first_delta, b_first_delta, a_last_delta, b_last_delta

    def test_format_stream_chunk_isolates_equivalent_requests_mirrored_end_order_stream_a(self, adapter):
        """Mirrored completion order should preserve stream A state."""
        a_first_delta, _, a_last_delta, _ = self._run_mirrored_end_order_interleaving(adapter)

        assert a_first_delta["role"] == "assistant"
        assert "reasoning_content" not in a_first_delta
        assert "content" not in a_first_delta
        assert a_last_delta["reasoning_content"] == "A1"
        assert a_last_delta["content"] == "A!"

    def test_format_stream_chunk_isolates_equivalent_requests_mirrored_end_order_stream_b(self, adapter):
        """Mirrored completion order should preserve stream B state."""
        _, b_first_delta, _, b_last_delta = self._run_mirrored_end_order_interleaving(adapter)

        assert b_first_delta["reasoning_content"] == "B"
        assert "content" not in b_first_delta
        assert b_last_delta["reasoning_content"] == "2"
        assert b_last_delta["content"] == "B!"

    def test_stream_parser_state_is_cleared_after_last_chunk(self, adapter):
        """Parser state should be fully released after natural completion."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        adapter.format_stream_chunk(
            StreamChunk(text="<think>rea", is_first=True),
            request,
        )
        assert len(adapter._thinking_parsers) == 1

        adapter.format_stream_chunk(
            StreamChunk(text="soning</think>Answer", is_last=True),
            request,
        )

        assert adapter._thinking_parsers == {}
        assert adapter._request_payload_keys == {}
        assert adapter._active_request_keys_by_payload == {}

    def test_format_stream_end_equivalent_request_does_not_leave_stale_state_that_breaks_adoption(self, adapter):
        """Equivalent-object end calls should not strand stale parser state for equal payloads."""
        request_a = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        request_a_end = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        request_b = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        request_b_resume = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        key_a = adapter._stream_state_key(request_a)
        key_b = adapter._stream_state_key(request_b)
        payload_key = adapter._stream_payload_key(request_a)

        # Set up asymmetric parser states so deleting the wrong key is detectable:
        # A is buffering a partial open tag; B is inside an active think span.
        adapter.format_stream_chunk(
            StreamChunk(text="<thi", is_first=True),
            request_a,
        )
        adapter.format_stream_chunk(
            StreamChunk(text="<think>B", is_first=True),
            request_b,
        )

        assert set(adapter._active_request_keys_by_payload[payload_key]) == {key_a, key_b}
        assert adapter._thinking_parsers[key_a]._buffer == "<thi"
        assert adapter._thinking_parsers[key_b]._in_thinking is True

        adapter.format_stream_end(request_a_end)

        # Equivalent-object end for A must retire A specifically, not arbitrary
        # one-of-N payload-matching streams.
        assert key_a not in adapter._thinking_parsers
        assert key_b in adapter._thinking_parsers
        assert set(adapter._active_request_keys_by_payload[payload_key]) == {key_b}

        resumed = adapter.format_stream_chunk(
            StreamChunk(text="2</think>X", is_last=True),
            request_b_resume,
        )
        resumed_delta = json.loads(resumed[6:-2])["choices"][0]["delta"]

        assert resumed_delta["reasoning_content"] == "2"
        assert resumed_delta["content"] == "X"
        assert adapter._thinking_parsers == {}
        assert adapter._request_payload_keys == {}
        assert adapter._active_request_keys_by_payload == {}

    def test_format_stream_end_equivalent_request_does_not_remove_other_stream_when_ending_stream_was_touched_last(
        self, adapter
    ):
        """Equivalent-object end must retire the intended stream regardless of touch order."""
        request_a = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        request_a_end = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        request_b = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        request_b_resume = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        key_a = adapter._stream_state_key(request_a)
        key_b = adapter._stream_state_key(request_b)
        payload_key = adapter._stream_payload_key(request_a)

        adapter.format_stream_chunk(
            StreamChunk(text="<thi", is_first=True),
            request_a,
        )
        adapter.format_stream_chunk(
            StreamChunk(text="<think>B", is_first=True),
            request_b,
        )
        # Touch A most recently so least-recently-touched cleanup would remove B.
        adapter.format_stream_chunk(
            StreamChunk(text="n"),
            request_a,
        )

        assert set(adapter._active_request_keys_by_payload[payload_key]) == {key_a, key_b}

        adapter.format_stream_end(request_a_end)

        assert key_a not in adapter._thinking_parsers
        assert key_b in adapter._thinking_parsers
        assert set(adapter._active_request_keys_by_payload[payload_key]) == {key_b}

        resumed = adapter.format_stream_chunk(
            StreamChunk(text="2</think>X", is_last=True),
            request_b_resume,
        )
        resumed_delta = json.loads(resumed[6:-2])["choices"][0]["delta"]

        assert resumed_delta["reasoning_content"] == "2"
        assert resumed_delta["content"] == "X"
        assert adapter._thinking_parsers == {}
        assert adapter._request_payload_keys == {}
        assert adapter._active_request_keys_by_payload == {}

    def test_format_stream_chunk_state_key_collision_does_not_corrupt_other_stream(self, adapter, monkeypatch):
        """Stream-key collisions should not let one stream wipe another stream's parser state."""
        monkeypatch.setattr(adapter, "_stream_state_key", lambda _: 12345)

        request_a = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="A")],
        )
        request_b = ChatCompletionRequest(
            model="other-model",
            messages=[Message(role="user", content="B")],
        )

        adapter.format_stream_chunk(
            StreamChunk(text="<think>A", is_first=True),
            request_a,
        )
        adapter.format_stream_chunk(
            StreamChunk(text="<think>B", is_first=True),
            request_b,
        )

        a_resumed = adapter.format_stream_chunk(
            StreamChunk(text="2</think>X", is_last=True),
            request_a,
        )
        b_resumed = adapter.format_stream_chunk(
            StreamChunk(text="3</think>Y", is_last=True),
            request_b,
        )
        a_resumed_delta = json.loads(a_resumed[6:-2])["choices"][0]["delta"]
        b_resumed_delta = json.loads(b_resumed[6:-2])["choices"][0]["delta"]

        assert a_resumed_delta["reasoning_content"] == "2"
        assert a_resumed_delta["content"] == "X"
        assert b_resumed_delta["reasoning_content"] == "3"
        assert b_resumed_delta["content"] == "Y"
        assert adapter._thinking_parsers == {}
        assert adapter._request_payload_keys == {}
        assert adapter._active_request_keys_by_payload == {}

    def test_stream_payload_key_is_not_recomputed_for_every_chunk(self, adapter, monkeypatch):
        """Payload-key derivation should be cached per request, not serialized per chunk."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        dump_calls = 0
        original_model_dump_json = ChatCompletionRequest.model_dump_json

        def counted_model_dump_json(self, *args, **kwargs):
            nonlocal dump_calls
            if self is request:
                dump_calls += 1
            return original_model_dump_json(self, *args, **kwargs)

        monkeypatch.setattr(
            ChatCompletionRequest,
            "model_dump_json",
            counted_model_dump_json,
        )

        adapter.format_stream_chunk(
            StreamChunk(text="A", is_first=True),
            request,
        )
        adapter.format_stream_chunk(
            StreamChunk(text="B"),
            request,
        )
        adapter.format_stream_chunk(
            StreamChunk(text="C", is_last=True),
            request,
        )
        adapter.format_stream_end(request)

        assert dump_calls == 1

    def test_format_stream_end_clears_parser_state_without_last_chunk(self, adapter):
        """format_stream_end should drop parser state even without an is_last chunk."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        adapter.format_stream_chunk(
            StreamChunk(text="<think>rea", is_first=True),
            request,
        )
        end_marker = adapter.format_stream_end(request)
        resumed = adapter.format_stream_chunk(StreamChunk(text="Answer"), request)

        resumed_delta = json.loads(resumed[6:-2])["choices"][0]["delta"]

        assert end_marker == "data: [DONE]\n\n"
        assert resumed_delta["content"] == "Answer"
        assert "reasoning_content" not in resumed_delta
        assert adapter._thinking_parsers == {}
        assert adapter._request_payload_keys == {}
        assert adapter._active_request_keys_by_payload == {}

    # =========================================================================
    # format_stream_end Tests
    # =========================================================================

    def test_format_stream_end(self, adapter):
        """Test formatting stream end marker."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        result = adapter.format_stream_end(request)

        assert result == "data: [DONE]\n\n"

    # =========================================================================
    # create_error_response Tests
    # =========================================================================

    def test_create_error_response_default(self, adapter):
        """Test creating error response with defaults."""
        result = adapter.create_error_response("Something went wrong")

        assert result["error"]["message"] == "Something went wrong"
        assert result["error"]["type"] == "server_error"
        assert result["error"]["code"] == 500
        assert result["error"]["param"] is None

    def test_create_error_response_custom_type(self, adapter):
        """Test creating error response with custom type."""
        result = adapter.create_error_response(
            "Invalid request",
            error_type="invalid_request_error",
            status_code=400,
        )

        assert result["error"]["message"] == "Invalid request"
        assert result["error"]["type"] == "invalid_request_error"
        assert result["error"]["code"] == 400

    def test_create_error_response_not_found(self, adapter):
        """Test creating 404 error response."""
        result = adapter.create_error_response(
            "Model not found",
            error_type="not_found_error",
            status_code=404,
        )

        assert result["error"]["code"] == 404
        assert result["error"]["type"] == "not_found_error"

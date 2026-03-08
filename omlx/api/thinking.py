# SPDX-License-Identifier: Apache-2.0
"""
Thinking/reasoning content parser for separating <think>...</think> blocks.

Provides both streaming (ThinkingParser) and non-streaming (extract_thinking)
interfaces for separating reasoning content from regular response content.

Used by reasoning models like DeepSeek R1, Qwen3/3.5, MiniMax that wrap
their chain-of-thought reasoning in <think>...</think> tags.
"""

import re
from typing import List, Tuple


# Tags used for thinking blocks
_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"
_OPEN_LEN = len(_OPEN_TAG)   # 7
_CLOSE_LEN = len(_CLOSE_TAG)  # 8

# Handle case where <think> is missing but </think> is present
# (scheduler prepends <think>\n but the tag may be split)
_THINKING_TAIL_PATTERN = re.compile(r'^(.*?)</think>', re.DOTALL)


def strip_think_tags(text: str | None, *, trim: bool = True) -> str:
    """Remove reserved think tag markers from a standard output field.

    Standard assistant-facing fields should expose reasoning as structured data,
    not raw ``<think>`` / ``</think>`` markup. This helper preserves the payload
    text while removing the control tags.
    """
    if not text:
        return ""

    cleaned = text.replace(_OPEN_TAG, "").replace(_CLOSE_TAG, "")
    return cleaned.strip() if trim else cleaned


def extract_thinking(text: str) -> Tuple[str, str]:
    """Extract thinking and content from complete text.

    Handles:
    - Normal: ``<think>reasoning</think>answer`` → ``("reasoning", "answer")``
    - No thinking: ``just answer`` → ``("", "just answer")``
    - Partial (no open tag): ``reasoning</think>answer`` → ``("reasoning", "answer")``
    - Empty think: ``<think></think>answer`` → ``("", "answer")``
    - Think only: ``<think>reasoning</think>`` → ``("reasoning", "")``

    Args:
        text: Complete model output text.

    Returns:
        Tuple of (thinking_content, regular_content).
    """
    if not text:
        return ("", "")

    # Handle partial: content before </think> without <think> tag
    if '</think>' in text and '<think>' not in text:
        match = _THINKING_TAIL_PATTERN.match(text)
        if match:
            thinking = match.group(1).strip()
            remaining = text[match.end():].strip()
            return (thinking, remaining)

    # Robust state-machine extraction for malformed/mixed tag streams:
    # - duplicate opening tags: "<think><think>..."
    # - extra closing tags: "</think></think>"
    # - unclosed thinking blocks at end of output
    thinking_parts: List[str] = []
    content_parts: List[str] = []
    current_thinking: List[str] = []
    in_thinking = False
    i = 0

    while i < len(text):
        remaining = text[i:]

        if remaining.startswith(_OPEN_TAG):
            # Duplicate opens are treated as idempotent.
            in_thinking = True
            i += _OPEN_LEN
            continue

        if remaining.startswith(_CLOSE_TAG):
            if in_thinking:
                thinking_parts.append("".join(current_thinking).strip())
                current_thinking = []
                in_thinking = False
            i += _CLOSE_LEN
            continue

        if in_thinking:
            current_thinking.append(text[i])
        else:
            content_parts.append(text[i])
        i += 1

    # If stream ends inside thinking, treat the trailing text as thinking.
    if in_thinking:
        thinking_parts.append("".join(current_thinking).strip())

    # Keep block boundaries for multi-think outputs while dropping empty blocks.
    non_empty_thinking = [part for part in thinking_parts if part]
    thinking = "\n".join(non_empty_thinking).strip()
    content = "".join(content_parts).strip()
    return (thinking, content)


class ThinkingParser:
    """Stateful streaming parser for separating <think>...</think> from content.

    Handles streaming chunks where tags may span multiple chunks.
    Returns (thinking_delta, content_delta) tuples for each feed() call.

    Example::

        parser = ThinkingParser()

        # Chunk 1: "<think>Let me"
        t, c = parser.feed("<think>Let me")
        # t = "Let me", c = ""

        # Chunk 2: " think</think>Answer"
        t, c = parser.feed(" think</think>Answer")
        # t = " think", c = "Answer"

        # Flush remaining
        t, c = parser.finish()
    """

    def __init__(self):
        self._in_thinking: bool = False
        self._buffer: str = ""  # Buffer for potential partial tags

    def feed(self, text: str) -> Tuple[str, str]:
        """Feed a text chunk, return (thinking_delta, content_delta).

        Args:
            text: New text chunk from model output.

        Returns:
            Tuple of (thinking_text, content_text) extracted from this chunk.
        """
        if not text:
            return ("", "")

        # Prepend any buffered partial tag content
        text = self._buffer + text
        self._buffer = ""

        thinking_out = []
        content_out = []

        i = 0
        while i < len(text):
            if text[i] == '<':
                # Check if this could be a tag start
                remaining = text[i:]

                # Try to match <think>
                if remaining.startswith(_OPEN_TAG):
                    self._in_thinking = True
                    i += _OPEN_LEN
                    continue

                # Try to match </think>
                if remaining.startswith(_CLOSE_TAG):
                    self._in_thinking = False
                    i += _CLOSE_LEN
                    continue

                # Check if it could be a partial tag (not enough chars yet)
                if self._could_be_tag(remaining):
                    # Buffer the rest and wait for more data
                    self._buffer = remaining
                    break

                # Not a tag, emit the '<' as regular content
                if self._in_thinking:
                    thinking_out.append('<')
                else:
                    content_out.append('<')
                i += 1
            else:
                if self._in_thinking:
                    thinking_out.append(text[i])
                else:
                    content_out.append(text[i])
                i += 1

        return ("".join(thinking_out), "".join(content_out))

    def finish(self) -> Tuple[str, str]:
        """Flush any remaining buffered content.

        Should be called when the stream is complete to emit any
        buffered characters that were waiting for potential tag completion.

        Returns:
            Tuple of (thinking_text, content_text) from remaining buffer.
        """
        if not self._buffer:
            return ("", "")

        # The buffer contains a partial tag that never completed.
        # Emit it as-is in the current mode.
        buf = self._buffer
        self._buffer = ""

        if self._in_thinking:
            return (buf, "")
        else:
            return ("", buf)

    @staticmethod
    def _could_be_tag(text: str) -> bool:
        """Check if text could be the start of a <think> or </think> tag.

        Returns True if text is a proper prefix of either tag but not
        yet a complete match.
        """
        length = len(text)
        if length >= _CLOSE_LEN:
            # Long enough to determine - not a partial tag
            return False

        # Check against both tags
        if _OPEN_TAG[:length] == text:
            return True
        if _CLOSE_TAG[:length] == text:
            return True

        return False

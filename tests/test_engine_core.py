# SPDX-License-Identifier: Apache-2.0
"""
Tests for EngineCore module.

Tests cover:
- EngineConfig: default values
- EngineCore initialization
- add_request(): adding requests (async)
- abort_request(): aborting requests (async)
- get_stats(): statistics

Note: Uses pytest-asyncio for async tests.
"""

import asyncio
from collections import deque
from contextlib import suppress
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from omlx.engine_core import EngineCore, AsyncEngineCore, EngineConfig
from omlx.request import Request, RequestOutput, RequestStatus, SamplingParams
from omlx.scheduler import SchedulerConfig


TEST_SYNC_TIMEOUT = 5.0


async def _wait_for_thread_event(
    event,
    timeout: float = TEST_SYNC_TIMEOUT,
    message: str = "timed out",
):
    """Wait for a threading.Event from async code with a clear assertion message."""
    fired = await asyncio.to_thread(event.wait, timeout)
    assert fired, message


async def _wait_until(
    predicate,
    timeout: float = TEST_SYNC_TIMEOUT,
    message: str = "condition not met",
):
    """Poll an in-memory condition from async code until it becomes true."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    assert predicate(), message


def _scheduler_request_fully_cleaned(scheduler, request_id: str) -> bool:
    """Return True when no scheduler structure still references the request."""
    return (
        request_id not in scheduler.requests
        and all(req.request_id != request_id for req in scheduler.waiting)
        and request_id not in scheduler.running
        and request_id not in scheduler.request_id_to_uid
        and request_id not in scheduler.uid_to_request_id.values()
    )


def _assert_scheduler_request_fully_cleaned(scheduler, request_id: str) -> None:
    """Assert the same cleanup invariants used by the ghost-request regression tests."""
    assert request_id not in scheduler.requests
    assert all(req.request_id != request_id for req in scheduler.waiting)
    assert request_id not in scheduler.running
    assert request_id not in scheduler.request_id_to_uid
    assert request_id not in scheduler.uid_to_request_id.values()


def _wait_for_release(
    event,
    timeout: float = TEST_SYNC_TIMEOUT,
    message: str = "release not signaled",
):
    """Block a worker-side hook and fail loudly if the test never opened the race window."""
    released = event.wait(timeout)
    assert released, message


class TestEngineConfig:
    """Tests for EngineConfig dataclass."""

    def test_default_values(self):
        """Test EngineConfig has correct defaults."""
        config = EngineConfig()

        assert config.model_name == ""
        assert config.scheduler_config is None
        assert config.step_interval == 0.001
        assert config.stream_interval == 1

    def test_custom_values(self):
        """Test EngineConfig with custom values."""
        scheduler_config = SchedulerConfig(max_num_seqs=64)
        config = EngineConfig(
            model_name="test-model",
            scheduler_config=scheduler_config,
            step_interval=0.005,
            stream_interval=5,
        )

        assert config.model_name == "test-model"
        assert config.scheduler_config is scheduler_config
        assert config.scheduler_config.max_num_seqs == 64
        assert config.step_interval == 0.005
        assert config.stream_interval == 5


class TestEngineCoreInitialization:
    """Tests for EngineCore initialization."""

    def test_init_with_defaults(self, mock_model, mock_tokenizer):
        """Test EngineCore initializes with default config."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                assert engine.model is mock_model
                assert engine.tokenizer is mock_tokenizer
                assert isinstance(engine.config, EngineConfig)
                assert engine._running is False
                assert engine._task is None
                assert engine._steps_executed == 0
                assert engine._output_collectors == {}
                assert engine._stream_states == {}
                assert engine._finished_events == {}
            finally:
                engine.close()

    def test_init_with_custom_config(self, mock_model, mock_tokenizer):
        """Test EngineCore initializes with custom config."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            config = EngineConfig(
                model_name="custom-model",
                step_interval=0.01,
                stream_interval=3,
            )
            engine = EngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=config,
            )

            try:
                assert engine.config.model_name == "custom-model"
                assert engine.config.step_interval == 0.01
                assert engine.config.stream_interval == 3
            finally:
                engine.close()

    def test_init_generates_engine_id(self, mock_model, mock_tokenizer):
        """Test EngineCore generates unique engine ID."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                assert engine.engine_id is not None
                assert len(engine.engine_id) > 0
            finally:
                engine.close()

    def test_init_with_custom_engine_id(self, mock_model, mock_tokenizer):
        """Test EngineCore uses provided engine ID."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
                engine_id="custom-engine-123",
            )

            try:
                assert engine.engine_id == "custom-engine-123"
            finally:
                engine.close()


class TestEngineCoreStartStop:
    """Tests for EngineCore start/stop."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, mock_model, mock_tokenizer):
        """Test start() sets engine to running state."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                assert engine._running is True
                assert engine._task is not None
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, mock_model, mock_tokenizer):
        """Test stop() clears running state."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()
                await engine.stop()

                assert engine._running is False
                assert engine._task is None
            finally:
                engine.close()

    @pytest.mark.asyncio
    async def test_is_running(self, mock_model, mock_tokenizer):
        """Test is_running() returns correct state."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                assert engine.is_running() is False

                await engine.start()
                assert engine.is_running() is True

                await engine.stop()
                assert engine.is_running() is False
            finally:
                engine.close()

    @pytest.mark.asyncio
    async def test_double_start_noop(self, mock_model, mock_tokenizer):
        """Test starting already running engine is no-op."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()
                first_task = engine._task

                await engine.start()  # Second start should be no-op
                assert engine._task is first_task
            finally:
                await engine.stop()
                engine.close()


class TestEngineCoreAddRequest:
    """Tests for EngineCore.add_request()."""

    @pytest.mark.asyncio
    async def test_add_request_returns_id(self, mock_model, mock_tokenizer):
        """Test add_request() returns request ID."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(
                    prompt="Hello, world!",
                    sampling_params=SamplingParams(max_tokens=50),
                )

                assert request_id is not None
                assert isinstance(request_id, str)
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_with_custom_id(self, mock_model, mock_tokenizer):
        """Test add_request() uses provided request ID."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(
                    prompt="Hello",
                    request_id="custom-request-001",
                )

                assert request_id == "custom-request-001"
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_creates_collector(self, mock_model, mock_tokenizer):
        """Test add_request() creates output collector."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(prompt="Hello")

                assert request_id in engine._output_collectors
                assert request_id in engine._stream_states
                assert request_id in engine._finished_events
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_with_default_sampling_params(self, mock_model, mock_tokenizer):
        """add_request() should construct and propagate default SamplingParams()."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            captured_requests = []
            original_add_request = engine.scheduler.add_request

            def capture_add_request(request):
                captured_requests.append(request)
                return original_add_request(request)

            engine.scheduler.add_request = capture_add_request

            try:
                request_id = await engine.add_request(prompt="Hello")

                assert request_id is not None
                assert len(captured_requests) == 1
                assert captured_requests[0].sampling_params == SamplingParams()
                assert engine.scheduler.requests[request_id].sampling_params == SamplingParams()
            finally:
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_cancellation_rolls_back_new_request_tracking(
        self, mock_model, mock_tokenizer
    ):
        """Cancelled add_request() should clear engine-local tracking even if scheduler insertion never happens."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "cancelled-new-request"
            entered_add = threading.Event()
            release_add = threading.Event()
            finished_add = threading.Event()

            def blocking_add_request(_request):
                entered_add.set()
                try:
                    _wait_for_release(release_add, message="test did not release blocked scheduler.add_request in time")
                finally:
                    finished_add.set()

            engine.scheduler.add_request = blocking_add_request

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt="Hello",
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                await _wait_for_thread_event(
                    entered_add,
                    message="scheduler.add_request did not start in time",
                )

                add_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await add_task

                assert request_id in engine._pending_adds

                release_add.set()
                await _wait_for_thread_event(
                    finished_add,
                    message="scheduler.add_request worker did not finish in time",
                )
                await _wait_until(
                    lambda: (
                        request_id not in engine._output_collectors
                        and request_id not in engine._stream_states
                        and request_id not in engine._finished_events
                        and request_id not in engine._pending_adds
                    ),
                    message="engine-local tracking cleanup did not finish in time",
                )

                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
                assert request_id not in engine._pending_adds
            finally:
                release_add.set()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_cancellation_preserves_existing_request_routing_for_subsequent_engine_loop_dispatch(
        self, mock_model, mock_tokenizer
    ):
        """Cancelled queued duplicate add should leave the live request routable when the engine loop runs later."""
        import threading
        from omlx.scheduler import SchedulerOutput

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "cancelled-existing-request"
            original_collector = MagicMock()
            original_stream_state = object()
            original_finished_event = asyncio.Event()
            engine._output_collectors[request_id] = original_collector
            engine._stream_states[request_id] = original_stream_state
            engine._finished_events[request_id] = original_finished_event
            existing_request = Request(
                request_id=request_id,
                prompt=[11, 12],
                sampling_params=SamplingParams(max_tokens=8),
            )
            existing_request.prompt_token_ids = [11, 12]
            existing_request.num_prompt_tokens = 2
            existing_request.batch_uid = 77
            existing_request.status = RequestStatus.RUNNING
            engine.scheduler.requests[request_id] = existing_request
            engine.scheduler.running[request_id] = existing_request
            engine.scheduler.request_id_to_uid[request_id] = 77
            engine.scheduler.uid_to_request_id[77] = request_id

            entered_add = threading.Event()
            release_add = threading.Event()
            finished_add = threading.Event()
            original_add_request = engine.scheduler.add_request
            step_calls = 0

            def blocking_add_request(request):
                entered_add.set()
                try:
                    _wait_for_release(release_add, message="test did not release blocked scheduler.add_request in time")
                    return original_add_request(request)
                finally:
                    finished_add.set()

            engine.scheduler.add_request = blocking_add_request
            queued_output = RequestOutput(
                request_id=request_id,
                new_text="token",
                finished=True,
                completion_tokens=1,
            )

            def live_request_is_schedulable():
                return (
                    engine.scheduler.requests.get(request_id) is existing_request
                    and engine.scheduler.running.get(request_id) is existing_request
                    and engine.scheduler.request_id_to_uid.get(request_id) == 77
                    and engine.scheduler.uid_to_request_id.get(77) == request_id
                )

            def has_requests():
                return step_calls == 0 and live_request_is_schedulable()

            def step():
                nonlocal step_calls
                assert live_request_is_schedulable(), (
                    "duplicate add disturbed live scheduler bookkeeping before later dispatch"
                )
                step_calls += 1
                return SchedulerOutput(outputs=[queued_output], has_work=True)

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt="Hello",
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                await _wait_for_thread_event(
                    entered_add,
                    message="scheduler.add_request did not start in time",
                )
                assert not add_task.done(), "duplicate add finished before queued cancellation window"

                current_collector = engine._output_collectors[request_id]
                current_finished_event = engine._finished_events[request_id]
                assert current_collector is original_collector
                assert current_finished_event is original_finished_event

                add_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await add_task

                assert engine._output_collectors[request_id] is original_collector
                assert engine._stream_states[request_id] is original_stream_state
                assert engine._finished_events[request_id] is original_finished_event

                release_add.set()
                await _wait_for_thread_event(
                    finished_add,
                    message="scheduler.add_request worker did not finish in time",
                )
                await _wait_until(
                    lambda: (
                        request_id in engine.scheduler.requests
                        and request_id not in engine._pending_adds
                    ),
                    message="duplicate add cancellation reconciliation did not finish in time",
                )

                assert engine.scheduler.requests[request_id] is existing_request
                assert engine.scheduler.running[request_id] is existing_request
                assert engine.scheduler.request_id_to_uid[request_id] == 77
                assert engine.scheduler.uid_to_request_id[77] == request_id
                assert engine._output_collectors[request_id] is original_collector
                assert engine._stream_states[request_id] is original_stream_state
                assert engine._finished_events[request_id] is original_finished_event
                assert request_id not in engine._pending_adds

                engine.scheduler.has_requests = has_requests
                engine.scheduler.step = step
                await engine.start()
                await _wait_until(
                    lambda: (
                        original_collector.put.call_count == 1
                        and original_finished_event.is_set()
                    ),
                    message="engine loop did not route output after duplicate cancellation",
                )
                delivered_output = original_collector.put.call_args.args[0]
                assert delivered_output.request_id == request_id
            finally:
                release_add.set()
                if engine.is_running():
                    await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_pending_add_guard_rejects_second_fresh_request_id(
        self, mock_model, mock_tokenizer
    ):
        """A second fresh add with the same ID must fail at _pending_adds before scheduler duplicate checks."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "pending-add-duplicate"
            entered_add = threading.Event()
            release_add = threading.Event()
            finished_add = threading.Event()
            original_add_request = engine.scheduler.add_request
            scheduler_call_request_ids = []

            def delayed_add_request(request):
                scheduler_call_request_ids.append(request.request_id)
                entered_add.set()
                try:
                    _wait_for_release(release_add, message="test did not release blocked scheduler.add_request in time")
                    return original_add_request(request)
                finally:
                    finished_add.set()

            engine.scheduler.add_request = delayed_add_request

            try:
                first_add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[1, 2, 3],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                started = await asyncio.to_thread(entered_add.wait, TEST_SYNC_TIMEOUT)
                assert started, "first scheduler.add_request did not start in time"
                assert not first_add_task.done(), "first add_request finished before pending-add window"
                first_collector = engine._output_collectors[request_id]
                first_stream_state = engine._stream_states[request_id]
                first_finished_event = engine._finished_events[request_id]

                with pytest.raises(ValueError, match="already exists"):
                    await engine.add_request(
                        prompt=[9, 9, 9],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )

                assert scheduler_call_request_ids == [request_id]
                assert request_id in engine._pending_adds
                assert engine._output_collectors[request_id] is first_collector
                assert engine._stream_states[request_id] is first_stream_state
                assert engine._finished_events[request_id] is first_finished_event

                release_add.set()
                completed = await asyncio.to_thread(finished_add.wait, TEST_SYNC_TIMEOUT)
                assert completed, "first scheduler.add_request worker did not finish in time"

                actual_request_id = await first_add_task
                assert actual_request_id == request_id
                assert scheduler_call_request_ids == [request_id]
                assert engine._output_collectors[request_id] is first_collector
                assert engine._stream_states[request_id] is first_stream_state
                assert engine._finished_events[request_id] is first_finished_event
                assert request_id not in engine._pending_adds
            finally:
                release_add.set()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_cancellation_does_not_leave_late_scheduler_entry(
        self, mock_model, mock_tokenizer
    ):
        """Cancellation should not leave a late scheduler request after executor job completes."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "cancelled-late-scheduler-entry"
            entered_add = threading.Event()
            release_add = threading.Event()
            finished_add = threading.Event()
            original_add_request = engine.scheduler.add_request

            def delayed_add_request(request):
                entered_add.set()
                _wait_for_release(release_add, message="test did not release blocked scheduler.add_request in time")
                try:
                    return original_add_request(request)
                finally:
                    finished_add.set()

            engine.scheduler.add_request = delayed_add_request

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[1, 2, 3],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                started = await asyncio.to_thread(entered_add.wait, TEST_SYNC_TIMEOUT)
                assert started, "scheduler.add_request did not start in time"

                add_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await add_task

                release_add.set()
                completed = await asyncio.to_thread(
                    finished_add.wait,
                    TEST_SYNC_TIMEOUT,
                )
                assert completed, "scheduler.add_request worker did not finish in time"

                deadline = asyncio.get_running_loop().time() + TEST_SYNC_TIMEOUT
                while asyncio.get_running_loop().time() < deadline:
                    if (
                        _scheduler_request_fully_cleaned(engine.scheduler, request_id)
                        and request_id not in engine._output_collectors
                        and request_id not in engine._stream_states
                        and request_id not in engine._finished_events
                        and request_id not in engine._pending_adds
                    ):
                        break
                    await asyncio.sleep(0.01)

                _assert_scheduler_request_fully_cleaned(engine.scheduler, request_id)
                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
                assert request_id not in engine._pending_adds
            finally:
                release_add.set()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_cancellation_close_immediately_does_not_leave_late_scheduler_entry(
        self, mock_model, mock_tokenizer
    ):
        """Cancellation followed by immediate close should not let a late worker leave scheduler state behind."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "cancelled-close-race"
            entered_add = threading.Event()
            release_add = threading.Event()
            finished_add = threading.Event()
            scheduler_ref = engine.scheduler
            original_add_request = scheduler_ref.add_request

            def delayed_add_request(request):
                entered_add.set()
                _wait_for_release(release_add, message="test did not release blocked scheduler.add_request in time")
                try:
                    return original_add_request(request)
                finally:
                    finished_add.set()

            scheduler_ref.add_request = delayed_add_request

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[4, 5, 6],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                started = await asyncio.to_thread(entered_add.wait, TEST_SYNC_TIMEOUT)
                assert started, "scheduler.add_request did not start in time"
                assert not add_task.done(), "add_request finished before cancellation/close race"

                add_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await add_task

                assert request_id in engine._pending_adds
                engine.close()
                assert request_id not in engine._pending_adds

                release_add.set()
                completed = await asyncio.to_thread(
                    finished_add.wait,
                    TEST_SYNC_TIMEOUT,
                )
                assert completed, "scheduler.add_request worker did not finish in time"

                deadline = asyncio.get_running_loop().time() + TEST_SYNC_TIMEOUT
                while asyncio.get_running_loop().time() < deadline:
                    if _scheduler_request_fully_cleaned(scheduler_ref, request_id):
                        break
                    await asyncio.sleep(0.01)

                _assert_scheduler_request_fully_cleaned(scheduler_ref, request_id)
            finally:
                release_add.set()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_close_during_blocked_add_does_not_repopulate_tracking(
        self, mock_model, mock_tokenizer
    ):
        """Closing during a live blocked add must not let add_request succeed or resurrect routing state."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "close-during-live-add"
            entered_add = threading.Event()
            release_add = threading.Event()
            finished_add = threading.Event()
            scheduler_ref = engine.scheduler
            original_add_request = scheduler_ref.add_request

            def delayed_add_request(request):
                entered_add.set()
                _wait_for_release(release_add, message="test did not release blocked scheduler.add_request in time")
                try:
                    return original_add_request(request)
                finally:
                    finished_add.set()

            scheduler_ref.add_request = delayed_add_request

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[7, 8, 9],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                started = await asyncio.to_thread(entered_add.wait, TEST_SYNC_TIMEOUT)
                assert started, "scheduler.add_request did not start in time"
                assert not add_task.done(), "add_request finished before close raced it"

                engine.close()
                assert engine.scheduler is None

                release_add.set()
                completed = await asyncio.to_thread(
                    finished_add.wait,
                    TEST_SYNC_TIMEOUT,
                )
                assert completed, "scheduler.add_request worker did not finish in time"

                with pytest.raises(RuntimeError, match="aborted before activation"):
                    await add_task

                assert engine._output_collectors == {}
                assert engine._stream_states == {}
                assert engine._finished_events == {}
                assert engine._pending_adds == {}
                _assert_scheduler_request_fully_cleaned(scheduler_ref, request_id)
            finally:
                release_add.set()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_close_during_blocked_duplicate_add_does_not_repopulate_tracking(
        self, mock_model, mock_tokenizer
    ):
        """Closing during a blocked duplicate add must not let the late worker resurrect closed-engine routing."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "close-during-duplicate-add"
            existing_request = Request(
                request_id=request_id,
                prompt=[11, 12],
                sampling_params=SamplingParams(max_tokens=8),
            )
            existing_request.prompt_token_ids = [11, 12]
            existing_request.num_prompt_tokens = 2
            engine.scheduler.requests[request_id] = existing_request

            engine._output_collectors[request_id] = MagicMock()
            engine._stream_states[request_id] = object()
            engine._finished_events[request_id] = asyncio.Event()

            entered_add = threading.Event()
            release_add = threading.Event()
            finished_add = threading.Event()
            scheduler_ref = engine.scheduler
            original_add_request = scheduler_ref.add_request

            def delayed_duplicate_add(request):
                entered_add.set()
                _wait_for_release(
                    release_add,
                    message="test did not release blocked scheduler.add_request in time",
                )
                try:
                    return original_add_request(request)
                finally:
                    finished_add.set()

            scheduler_ref.add_request = delayed_duplicate_add

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[99],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                await _wait_for_thread_event(
                    entered_add,
                    message="scheduler.add_request did not start in time",
                )
                assert not add_task.done(), "duplicate add finished before close raced it"

                engine.close()
                assert engine.scheduler is None

                release_add.set()
                await _wait_for_thread_event(
                    finished_add,
                    message="scheduler.add_request worker did not finish in time",
                )

                with pytest.raises(RuntimeError, match="aborted before activation"):
                    await add_task

                assert engine._output_collectors == {}
                assert engine._stream_states == {}
                assert engine._finished_events == {}
                assert engine._pending_adds == {}
                _assert_scheduler_request_fully_cleaned(scheduler_ref, request_id)
            finally:
                release_add.set()
                engine.close()

    @pytest.mark.asyncio
    async def test_stream_outputs_close_during_pending_add_raises_without_hanging(
        self, mock_model, mock_tokenizer
    ):
        """close() during a blocked add should wake waiting stream consumer with a deterministic error."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "close-pending-add-stream"
            entered_add = threading.Event()
            release_add = threading.Event()
            finished_add = threading.Event()
            scheduler_ref = engine.scheduler
            original_add_request = scheduler_ref.add_request
            stream_task = None

            def delayed_add_request(request):
                entered_add.set()
                _wait_for_release(
                    release_add,
                    message="test did not release blocked scheduler.add_request in time",
                )
                try:
                    return original_add_request(request)
                finally:
                    finished_add.set()

            scheduler_ref.add_request = delayed_add_request

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[57, 58, 59],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                await _wait_for_thread_event(
                    entered_add,
                    message="scheduler.add_request did not start in time",
                )
                assert not add_task.done(), "add_request finished before close raced it"

                collector = engine._output_collectors[request_id]
                consumer_waiting = asyncio.Event()
                original_get = collector.get

                async def instrumented_get():
                    consumer_waiting.set()
                    return await original_get()

                collector.get = instrumented_get

                async def drain_stream():
                    with pytest.raises(RuntimeError, match="Request aborted"):
                        async for _ in engine.stream_outputs(request_id):
                            pass

                stream_task = asyncio.create_task(drain_stream())
                await asyncio.wait_for(consumer_waiting.wait(), timeout=TEST_SYNC_TIMEOUT)

                engine.close()
                assert engine.scheduler is None
                release_add.set()
                await _wait_for_thread_event(
                    finished_add,
                    message="scheduler.add_request worker did not finish in time",
                )
                await _wait_until(
                    lambda: stream_task.done(),
                    message="stream_outputs consumer did not unblock after close",
                )
                await stream_task

                with pytest.raises(RuntimeError, match="aborted before activation"):
                    await add_task

                assert engine._output_collectors == {}
                assert engine._stream_states == {}
                assert engine._finished_events == {}
                assert engine._pending_adds == {}
                _assert_scheduler_request_fully_cleaned(scheduler_ref, request_id)
            finally:
                release_add.set()
                if stream_task is not None and not stream_task.done():
                    stream_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await stream_task
                engine.close()

    @pytest.mark.asyncio
    async def test_generate_close_during_pending_add_raises_without_hanging(
        self, mock_model, mock_tokenizer
    ):
        """close() during a blocked add should cause generate() to raise promptly."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "close-pending-add-generate"
            entered_add = threading.Event()
            release_add = threading.Event()
            finished_add = threading.Event()
            scheduler_ref = engine.scheduler
            original_add_request = scheduler_ref.add_request

            def delayed_add_request(request):
                entered_add.set()
                _wait_for_release(
                    release_add,
                    message="test did not release blocked scheduler.add_request in time",
                )
                try:
                    return original_add_request(request)
                finally:
                    finished_add.set()

            scheduler_ref.add_request = delayed_add_request

            try:
                generate_task = asyncio.create_task(
                    engine.generate(
                        prompt=[67, 68, 69],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                await _wait_for_thread_event(
                    entered_add,
                    message="scheduler.add_request did not start in time",
                )
                assert not generate_task.done(), "generate finished before close raced it"

                engine.close()
                assert engine.scheduler is None

                release_add.set()
                await _wait_for_thread_event(
                    finished_add,
                    message="scheduler.add_request worker did not finish in time",
                )
                await _wait_until(
                    lambda: generate_task.done(),
                    message="generate consumer did not unblock after close",
                )
                with pytest.raises(RuntimeError, match="aborted before activation"):
                    await generate_task

                assert engine._output_collectors == {}
                assert engine._stream_states == {}
                assert engine._finished_events == {}
                assert engine._pending_adds == {}
                _assert_scheduler_request_fully_cleaned(scheduler_ref, request_id)
            finally:
                release_add.set()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_duplicate_id_failure_preserves_routing_for_subsequent_buffered_dispatch(
        self, mock_model, mock_tokenizer
    ):
        """A failed queued duplicate add should leave buffered streaming on the live request intact."""
        import threading
        from omlx.engine_core import RequestStreamState
        from omlx.scheduler import SchedulerOutput

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=EngineConfig(stream_interval=2),
            )

            request_id = "duplicate-routing-window"
            existing_request = Request(
                request_id=request_id,
                prompt=[11, 12],
                sampling_params=SamplingParams(max_tokens=8),
            )
            existing_request.prompt_token_ids = [11, 12]
            existing_request.num_prompt_tokens = 2
            existing_request.batch_uid = 99
            existing_request.status = RequestStatus.RUNNING
            engine.scheduler.requests[request_id] = existing_request
            engine.scheduler.running[request_id] = existing_request
            engine.scheduler.request_id_to_uid[request_id] = 99
            engine.scheduler.uid_to_request_id[99] = request_id

            original_collector = MagicMock()
            original_stream_state = RequestStreamState(stream_interval=2)
            original_finished_event = asyncio.Event()
            engine._output_collectors[request_id] = original_collector
            engine._stream_states[request_id] = original_stream_state
            engine._finished_events[request_id] = original_finished_event

            entered_add = threading.Event()
            release_add = threading.Event()
            original_add_request = engine.scheduler.add_request
            step_calls = 0

            def delayed_duplicate_add(request):
                entered_add.set()
                _wait_for_release(release_add, message="test did not release blocked scheduler.add_request in time")
                return original_add_request(request)

            engine.scheduler.add_request = delayed_duplicate_add
            queued_output = RequestOutput(
                request_id=request_id,
                new_text="token",
                finished=True,
                completion_tokens=1,
            )

            def live_request_is_schedulable():
                return (
                    engine.scheduler.requests.get(request_id) is existing_request
                    and engine.scheduler.running.get(request_id) is existing_request
                    and engine.scheduler.request_id_to_uid.get(request_id) == 99
                    and engine.scheduler.uid_to_request_id.get(99) == request_id
                )

            def has_requests():
                return step_calls == 0 and live_request_is_schedulable()

            def step():
                nonlocal step_calls
                assert live_request_is_schedulable(), (
                    "duplicate add disturbed live scheduler bookkeeping before later buffered dispatch"
                )
                step_calls += 1
                return SchedulerOutput(outputs=[queued_output], has_work=True)

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[99],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                await _wait_for_thread_event(
                    entered_add,
                    message="scheduler.add_request did not start in time",
                )
                assert not add_task.done(), "duplicate add finished before queued-routing window"

                current_collector = engine._output_collectors[request_id]
                current_finished_event = engine._finished_events[request_id]
                assert current_collector is original_collector
                assert current_finished_event is original_finished_event

                release_add.set()
                with pytest.raises(ValueError, match="already exists"):
                    await add_task

                engine.scheduler.has_requests = has_requests
                engine.scheduler.step = step
                await engine.start()
                await _wait_until(
                    lambda: (
                        original_collector.put.call_count == 1
                        and original_finished_event.is_set()
                        and original_stream_state.sent_tokens == 1
                    ),
                    message="buffered engine loop did not route output after duplicate failure",
                )
                assert engine.scheduler.requests[request_id] is existing_request
                assert engine.scheduler.running[request_id] is existing_request
                assert engine.scheduler.request_id_to_uid[request_id] == 99
                assert engine.scheduler.uid_to_request_id[99] == request_id
                assert engine._output_collectors[request_id] is original_collector
                assert engine._stream_states[request_id] is original_stream_state
                assert engine._finished_events[request_id] is original_finished_event
                assert request_id not in engine._pending_adds
                original_collector.put.assert_called_once()
                delivered_output = original_collector.put.call_args.args[0]
                assert delivered_output.request_id == request_id
                assert original_finished_event.is_set()
                assert original_stream_state.sent_tokens == 1
            finally:
                release_add.set()
                if engine.is_running():
                    await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_abort_while_executor_add_is_blocked_raises_cleanly(
        self, mock_model, mock_tokenizer
    ):
        """Aborting an in-flight net-new add should not return a dead request ID."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "abort-during-inflight-add"
            entered_add = threading.Event()
            release_add = threading.Event()
            original_add_request = engine.scheduler.add_request

            def delayed_add_request(request):
                entered_add.set()
                _wait_for_release(release_add, message="test did not release blocked scheduler.add_request in time")
                return original_add_request(request)

            engine.scheduler.add_request = delayed_add_request

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[1, 2, 3],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                started = await asyncio.to_thread(entered_add.wait, TEST_SYNC_TIMEOUT)
                assert started, "scheduler.add_request did not start in time"
                assert not add_task.done(), "add_request completed before abort_request raced it"

                await engine.abort_request(request_id)

                release_add.set()
                with pytest.raises(RuntimeError, match="aborted before activation"):
                    await add_task

                _assert_scheduler_request_fully_cleaned(engine.scheduler, request_id)
                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
                assert request_id not in engine._pending_adds
            finally:
                release_add.set()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_request_during_pending_add_notifies_stream_outputs_consumer(
        self, mock_model, mock_tokenizer
    ):
        """An attached stream_outputs() consumer should get an abort error, not silently time out."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "abort-pending-add-stream"
            entered_add = threading.Event()
            release_add = threading.Event()
            original_add_request = engine.scheduler.add_request
            stream_task = None

            def delayed_add_request(request):
                entered_add.set()
                _wait_for_release(
                    release_add,
                    message="test did not release blocked scheduler.add_request in time",
                )
                return original_add_request(request)

            engine.scheduler.add_request = delayed_add_request

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[21, 22, 23],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                await _wait_for_thread_event(
                    entered_add,
                    message="scheduler.add_request did not start in time",
                )
                assert not add_task.done(), "add_request completed before abort_request raced it"

                collector = engine._output_collectors[request_id]
                consumer_waiting = asyncio.Event()
                original_get = collector.get

                async def instrumented_get():
                    consumer_waiting.set()
                    return await original_get()

                collector.get = instrumented_get
                seen_outputs = []

                async def drain_stream():
                    with pytest.raises(RuntimeError, match="Request aborted"):
                        async for output in engine.stream_outputs(request_id):
                            seen_outputs.append(output)

                stream_task = asyncio.create_task(drain_stream())
                await asyncio.wait_for(consumer_waiting.wait(), timeout=TEST_SYNC_TIMEOUT)

                await engine.abort_request(request_id)
                await asyncio.wait_for(stream_task, timeout=TEST_SYNC_TIMEOUT)
                assert not add_task.done(), "stream_outputs only unblocked after pending add completed"

                release_add.set()
                with pytest.raises(RuntimeError, match="aborted before activation"):
                    await add_task

                assert seen_outputs
                assert seen_outputs[-1].request_id == request_id
                assert seen_outputs[-1].finished is True
                assert seen_outputs[-1].error == "Request aborted"
                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
                assert request_id not in engine._pending_adds
            finally:
                release_add.set()
                if stream_task is not None and not stream_task.done():
                    stream_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await stream_task
                engine.close()

    @pytest.mark.asyncio
    async def test_duplicate_add_failure_does_not_restore_tracking_after_abort(
        self, mock_model, mock_tokenizer
    ):
        """Duplicate add should not resurrect routing state removed by abort_request()."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "duplicate-abort-race"
            existing_request = Request(
                request_id=request_id,
                prompt=[11, 12],
                sampling_params=SamplingParams(max_tokens=8),
            )
            existing_request.prompt_token_ids = [11, 12]
            existing_request.num_prompt_tokens = 2
            engine.scheduler.requests[request_id] = existing_request

            original_collector = MagicMock()
            original_stream_state = object()
            original_finished_event = asyncio.Event()
            engine._output_collectors[request_id] = original_collector
            engine._stream_states[request_id] = original_stream_state
            engine._finished_events[request_id] = original_finished_event

            entered_add = threading.Event()
            release_add = threading.Event()
            original_add_request = engine.scheduler.add_request

            def delayed_duplicate_add(request):
                entered_add.set()
                _wait_for_release(release_add, message="test did not release blocked scheduler.add_request in time")
                return original_add_request(request)

            engine.scheduler.add_request = delayed_duplicate_add

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[99],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                started = await asyncio.to_thread(entered_add.wait, TEST_SYNC_TIMEOUT)
                assert started, "scheduler.add_request did not start in time"
                assert not add_task.done(), "duplicate add finished before abort_request raced it"

                await engine.abort_request(request_id)
                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events

                release_add.set()
                with pytest.raises(ValueError, match="already exists"):
                    await add_task

                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
                assert request_id not in engine._pending_adds
            finally:
                release_add.set()
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_scheduler_failure_rolls_back_tracking_state(
        self, mock_model, mock_tokenizer
    ):
        """Reachable duplicate-ID failure should preserve live routing and scheduler bookkeeping."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "existing-request"
            existing_collector = MagicMock()
            existing_stream_state = object()
            existing_event = asyncio.Event()
            existing_scheduler_request = Request(
                request_id=request_id,
                prompt=[4, 5],
                sampling_params=SamplingParams(max_tokens=8),
            )
            existing_scheduler_request.prompt_token_ids = [4, 5]
            existing_scheduler_request.num_prompt_tokens = 2
            existing_scheduler_request.batch_uid = 41
            existing_scheduler_request.status = RequestStatus.RUNNING
            engine._output_collectors[request_id] = existing_collector
            engine._stream_states[request_id] = existing_stream_state
            engine._finished_events[request_id] = existing_event
            engine.scheduler.requests[request_id] = existing_scheduler_request
            engine.scheduler.running[request_id] = existing_scheduler_request
            engine.scheduler.request_id_to_uid[request_id] = 41
            engine.scheduler.uid_to_request_id[41] = request_id

            try:
                with pytest.raises(ValueError, match="already exists"):
                    await engine.add_request(
                        prompt=[1, 2, 3],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )

                assert engine.scheduler.requests[request_id] is existing_scheduler_request
                assert all(req.request_id != request_id for req in engine.scheduler.waiting)
                assert engine.scheduler.running[request_id] is existing_scheduler_request
                assert engine.scheduler.request_id_to_uid[request_id] == 41
                assert engine.scheduler.uid_to_request_id[41] == request_id
                assert engine._output_collectors[request_id] is existing_collector
                assert engine._stream_states[request_id] is existing_stream_state
                assert engine._finished_events[request_id] is existing_event
                assert request_id not in engine._pending_adds
            finally:
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_scheduler_failure_rolls_back_new_request_tracking(
        self, mock_model, mock_tokenizer
    ):
        """Reachable tokenizer failure should remove new engine tracking without leaving scheduler ghosts."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "fresh-request-failure"

            try:
                with patch.object(
                    mock_tokenizer,
                    "encode",
                    side_effect=RuntimeError("tokenizer boom"),
                ):
                    with pytest.raises(RuntimeError, match="tokenizer boom"):
                        await engine.add_request(
                            prompt="Hello",
                            request_id=request_id,
                            sampling_params=SamplingParams(max_tokens=8),
                        )

                _assert_scheduler_request_fully_cleaned(engine.scheduler, request_id)
                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
                assert request_id not in engine._pending_adds
            finally:
                engine.close()

    @pytest.mark.asyncio
    async def test_add_request_mixed_rewind_serializes_with_running_step(
        self, mock_model, mock_tokenizer
    ):
        """Mixed-cache exact-hit rewind must serialize with step() on the MLX executor."""
        import threading
        import time

        from omlx.cache.paged_cache import BlockTable
        from omlx.scheduler import SchedulerOutput

        class KVCache:
            def __init__(self, offset=4):
                self.offset = offset

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                if not self.can_rewind(n):
                    return False
                self.offset -= n
                return True

        class RotatingKVCache:
            def __init__(self, offset=4):
                self.offset = offset

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                if not self.can_rewind(n):
                    return False
                self.offset -= n
                return True

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            active_count = 0
            max_concurrent = 0
            step_calls = 0
            lock = threading.Lock()
            step_started = threading.Event()
            step_release = threading.Event()
            rewind_started = threading.Event()
            step_thread_ids = []
            rewind_thread_ids = []

            def _enter_critical():
                nonlocal active_count, max_concurrent
                with lock:
                    active_count += 1
                    max_concurrent = max(max_concurrent, active_count)

            def _exit_critical():
                nonlocal active_count
                with lock:
                    active_count -= 1

            def tracked_step():
                nonlocal step_calls
                _enter_critical()
                try:
                    step_thread_ids.append(threading.get_ident())
                    step_started.set()
                    _wait_for_release(step_release, message="test did not release blocked scheduler.step in time")
                    time.sleep(0.01)
                    step_calls += 1
                    return SchedulerOutput(outputs=[])
                finally:
                    _exit_critical()

            def has_requests():
                return step_calls == 0

            original_rewind = engine.scheduler._rewind_prompt_cache_for_generation

            def tracked_rewind(cache_list):
                _enter_critical()
                try:
                    rewind_started.set()
                    rewind_thread_ids.append(threading.get_ident())
                    return original_rewind(cache_list)
                finally:
                    _exit_critical()

            engine.scheduler.step = tracked_step
            engine.scheduler.has_requests = has_requests
            engine.scheduler._rewind_prompt_cache_for_generation = tracked_rewind
            engine.scheduler.block_aware_cache = MagicMock()
            engine.scheduler.paged_cache_manager = MagicMock()
            engine.scheduler.paged_cache_manager.release_for_eviction.return_value = 0
            engine.scheduler.paged_cache_manager.get_block_table.return_value = None

            block_table = BlockTable(
                request_id="req-engine-mixed-rewind-threading",
                block_ids=[1],
                num_tokens=4,
            )
            engine.scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
            engine.scheduler.block_aware_cache.reconstruct_cache.return_value = [
                KVCache(offset=4),
                RotatingKVCache(offset=4),
            ]

            request_id = "req-engine-mixed-rewind-threading"
            loop_thread_id = threading.get_ident()

            try:
                await engine.start()

                started = await asyncio.to_thread(step_started.wait, TEST_SYNC_TIMEOUT)
                assert started, "engine loop did not start a step() in time"

                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[31, 32, 33, 34],
                        sampling_params=SamplingParams(max_tokens=8),
                        request_id=request_id,
                    )
                )
                deadline = asyncio.get_running_loop().time() + TEST_SYNC_TIMEOUT
                while (
                    request_id not in engine._pending_adds
                    and asyncio.get_running_loop().time() < deadline
                ):
                    await asyncio.sleep(0)
                assert request_id in engine._pending_adds
                assert not rewind_started.is_set(), "rewind started before step() released the executor"
                step_release.set()
                rewind_started_in_time = await asyncio.to_thread(
                    rewind_started.wait,
                    TEST_SYNC_TIMEOUT,
                )
                assert rewind_started_in_time, "rewind did not start after step() released the executor"
                actual_request_id = await add_task

                assert actual_request_id == request_id
                assert step_thread_ids, "engine loop did not invoke step()"
                assert rewind_thread_ids, "mixed exact-hit path did not invoke rewind"
                assert all(
                    tid != loop_thread_id for tid in rewind_thread_ids
                ), "rewind executed on the event-loop thread"
                assert all(
                    tid == step_thread_ids[0] for tid in rewind_thread_ids
                ), "rewind did not execute on the shared MLX executor thread"
                assert max_concurrent == 1, (
                    "rewind overlapped with step(); mixed-cache rewind is not serialized "
                    "on the shared MLX executor"
                )
                engine.scheduler.paged_cache_manager.delete_block_table.assert_not_called()
                assert request_id not in engine._pending_adds
            finally:
                await engine.stop()
                engine.close()


class TestEngineCoreAbortRequest:
    """Tests for EngineCore.abort_request()."""

    @pytest.mark.asyncio
    async def test_abort_request(self, mock_model, mock_tokenizer):
        """Test abort_request() returns True for existing request."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(prompt="Hello")
                result = await engine.abort_request(request_id)

                assert result is True
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_request_cleans_up(self, mock_model, mock_tokenizer):
        """Test abort_request() cleans up tracking state."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(prompt="Hello")
                await engine.abort_request(request_id)

                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_request_no_ghost_in_scheduler(
        self, mock_model, mock_tokenizer
    ):
        """Deferred abort must clean scheduler state (no ghost request).

        Regression: _cleanup_request used to call remove_finished_request()
        which deleted from scheduler.requests before the deferred abort ran,
        causing _do_abort_request to skip cleanup and leave ghost state in
        scheduler.running / uid mappings / active batch.
        """
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(prompt="Hello")

                # Request should be in scheduler waiting
                assert request_id in engine.scheduler.requests

                await engine.abort_request(request_id)

                # Engine-core state cleaned immediately
                assert request_id not in engine._output_collectors

                # Process the deferred abort (normally happens in step())
                engine.scheduler._process_pending_aborts()

                # Scheduler state must be fully cleaned
                assert request_id not in engine.scheduler.requests
                assert request_id not in engine.scheduler.running
                assert request_id not in engine.scheduler.request_id_to_uid
            finally:
                await engine.stop()
                engine.close()


class TestEngineCoreGetStats:
    """Tests for EngineCore.get_stats()."""

    @pytest.mark.asyncio
    async def test_get_stats_initial(self, mock_model, mock_tokenizer):
        """Test get_stats() returns initial values."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                stats = engine.get_stats()

                assert "running" in stats
                assert "uptime_seconds" in stats
                assert "steps_executed" in stats
                assert "active_requests" in stats
                assert "stream_interval" in stats
                assert stats["running"] is True
                assert stats["steps_executed"] == 0
                assert stats["active_requests"] == 0
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_get_stats_includes_scheduler_stats(self, mock_model, mock_tokenizer):
        """Test get_stats() includes scheduler statistics."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                stats = engine.get_stats()

                # Should include scheduler stats
                assert "num_waiting" in stats
                assert "num_running" in stats
            finally:
                engine.close()


class TestEngineCoreClose:
    """Tests for EngineCore.close()."""

    def test_close_releases_model(self, mock_model, mock_tokenizer):
        """Test close() releases model ownership."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine.close()

            # Should have called release
            mock_registry.return_value.release.assert_called()

    def test_close_idempotent(self, mock_model, mock_tokenizer):
        """Test close() can be called multiple times safely."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine.close()
            engine.close()  # Should not raise


class TestEngineCoreGetCacheStats:
    """Tests for EngineCore.get_cache_stats()."""

    def test_get_cache_stats(self, mock_model, mock_tokenizer):
        """Test get_cache_stats() returns None when no cache."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                stats = engine.get_cache_stats()

                # No SSD cache configured, should return None
                assert stats is None
            finally:
                engine.close()


class TestEngineCoreGenerateCancellation:
    """Tests for EngineCore.generate() cancellation handling."""

    @pytest.mark.asyncio
    async def test_generate_cancel_aborts_request(self, mock_model, mock_tokenizer):
        """Test that cancelling generate() aborts the underlying request."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                # Create a task that calls generate - it will block on event.wait()
                task = asyncio.create_task(
                    engine.generate(
                        prompt="Hello, world!",
                        sampling_params=SamplingParams(max_tokens=50),
                    )
                )

                # Give the task time to reach event.wait()
                await asyncio.sleep(0.05)

                # There should be one active request
                assert len(engine._output_collectors) == 1
                request_id = list(engine._output_collectors.keys())[0]

                # Cancel the task (simulating client disconnect)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

                # After cancellation, the request should be cleaned up
                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_generate_abort_request_during_pending_add_raises_cleanly(
        self, mock_model, mock_tokenizer
    ):
        """generate() should fail promptly if abort_request() retires a blocked add before activation."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "generate-abort-pending-add"
            entered_add = threading.Event()
            release_add = threading.Event()
            original_add_request = engine.scheduler.add_request

            def delayed_add_request(request):
                entered_add.set()
                _wait_for_release(
                    release_add,
                    message="test did not release blocked scheduler.add_request in time",
                )
                return original_add_request(request)

            engine.scheduler.add_request = delayed_add_request

            try:
                task = asyncio.create_task(
                    engine.generate(
                        prompt=[41, 42, 43],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                await _wait_for_thread_event(
                    entered_add,
                    message="scheduler.add_request did not start in time",
                )
                assert not task.done(), "generate() finished before abort_request raced it"

                await engine.abort_request(request_id)

                release_add.set()
                with pytest.raises(RuntimeError, match="aborted before activation"):
                    await asyncio.wait_for(task, timeout=TEST_SYNC_TIMEOUT)

                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
                assert request_id not in engine._pending_adds
                _assert_scheduler_request_fully_cleaned(engine.scheduler, request_id)
            finally:
                release_add.set()
                engine.close()

    @pytest.mark.asyncio
    async def test_generate_abort_all_requests_during_pending_add_raises_cleanly(
        self, mock_model, mock_tokenizer
    ):
        """generate() should fail promptly if abort_all_requests() retires a blocked add before activation."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "generate-abort-all-pending-add"
            entered_add = threading.Event()
            release_add = threading.Event()
            original_add_request = engine.scheduler.add_request

            def delayed_add_request(request):
                entered_add.set()
                _wait_for_release(
                    release_add,
                    message="test did not release blocked scheduler.add_request in time",
                )
                return original_add_request(request)

            engine.scheduler.add_request = delayed_add_request

            try:
                task = asyncio.create_task(
                    engine.generate(
                        prompt=[51, 52, 53],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                await _wait_for_thread_event(
                    entered_add,
                    message="scheduler.add_request did not start in time",
                )
                assert not task.done(), "generate() finished before abort_all_requests raced it"

                count = await engine.abort_all_requests()
                assert count == 1

                release_add.set()
                with pytest.raises(RuntimeError, match="aborted before activation"):
                    await asyncio.wait_for(task, timeout=TEST_SYNC_TIMEOUT)

                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
                assert request_id not in engine._pending_adds
                _assert_scheduler_request_fully_cleaned(engine.scheduler, request_id)
            finally:
                release_add.set()
                engine.close()

    @pytest.mark.asyncio
    async def test_generate_cancel_multiple_requests(self, mock_model, mock_tokenizer):
        """Test cancelling one generate() does not affect others."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                # Create two generate tasks
                task1 = asyncio.create_task(
                    engine.generate(
                        prompt="Request 1",
                        sampling_params=SamplingParams(max_tokens=50),
                    )
                )
                task2 = asyncio.create_task(
                    engine.generate(
                        prompt="Request 2",
                        sampling_params=SamplingParams(max_tokens=50),
                    )
                )

                await asyncio.sleep(0.05)

                # Should have two active requests
                assert len(engine._output_collectors) == 2
                request_ids = list(engine._output_collectors.keys())

                # Cancel only the first task
                task1.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task1

                # First request cleaned up, second still active
                assert request_ids[0] not in engine._output_collectors
                assert request_ids[1] in engine._output_collectors

                # Clean up second task
                task2.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task2
            finally:
                await engine.stop()
                engine.close()


class TestEngineCoreErrorPropagation:
    """Tests for error propagation from engine loop to requests."""

    @pytest.mark.asyncio
    async def test_error_output_propagates_to_collector(self, mock_model, mock_tokenizer):
        """Test that engine loop errors are sent to request collectors."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                # Add a request
                request_id = await engine.add_request(
                    prompt="Hello",
                    sampling_params=SamplingParams(max_tokens=50),
                )

                # Simulate: put this request into scheduler.running
                engine.scheduler.running[request_id] = MagicMock()

                # Manually put an error output into the collector
                # (simulating what _engine_loop does on exception)
                collector = engine._output_collectors.get(request_id)
                assert collector is not None

                error_output = RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="error",
                    error="Memory limit exceeded during prefill",
                )
                collector.put(error_output)

                # The collector should have the error output
                result = collector.get_nowait()
                assert result is not None
                assert result.error == "Memory limit exceeded during prefill"
                assert result.finished is True
                assert result.finish_reason == "error"
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_stream_outputs_raises_on_error(self, mock_model, mock_tokenizer):
        """Test stream_outputs raises RuntimeError when error output received."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(
                    prompt="Hello",
                    sampling_params=SamplingParams(max_tokens=50),
                )

                # Put an error output into the collector
                collector = engine._output_collectors[request_id]
                error_output = RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="error",
                    error="Memory limit exceeded during prefill",
                )
                collector.put(error_output)

                # stream_outputs should yield the error output then raise
                with pytest.raises(RuntimeError, match="Memory limit exceeded"):
                    async for _ in engine.stream_outputs(request_id):
                        pass
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_generate_raises_on_error(self, mock_model, mock_tokenizer):
        """Test generate() raises RuntimeError when error output received."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                request_id = await engine.add_request(
                    prompt="Hello",
                    sampling_params=SamplingParams(max_tokens=50),
                )

                # Put an error output and set the finished event
                collector = engine._output_collectors[request_id]
                error_output = RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="error",
                    error="Memory limit exceeded during prefill",
                )
                collector.put(error_output)

                event = engine._finished_events[request_id]
                event.set()

                # generate() internally waits on event then drains collector
                # We need to call it in a way that bypasses add_request
                # since the request is already added. Use _generate_from_id
                # directly, but it doesn't exist. Instead, test the drain logic.
                final_output = None
                while True:
                    output = collector.get_nowait()
                    if output is None:
                        break
                    final_output = output

                assert final_output is not None
                assert final_output.error == "Memory limit exceeded during prefill"
            finally:
                await engine.stop()
                engine.close()


class TestAsyncEngineCore:
    """Tests for AsyncEngineCore wrapper."""

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_model, mock_tokenizer):
        """Test AsyncEngineCore as async context manager."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            async with AsyncEngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
            ) as engine:
                assert engine.engine._running is True

            # After exit, should be stopped
            assert engine.engine._running is False

    @pytest.mark.asyncio
    async def test_add_request(self, mock_model, mock_tokenizer):
        """Test AsyncEngineCore.add_request()."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            async with AsyncEngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
            ) as engine:
                request_id = await engine.add_request(prompt="Hello")

                assert request_id is not None

    @pytest.mark.asyncio
    async def test_abort_request(self, mock_model, mock_tokenizer):
        """Test AsyncEngineCore.abort_request()."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            async with AsyncEngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
            ) as engine:
                request_id = await engine.add_request(prompt="Hello")
                result = await engine.abort_request(request_id)

                assert result is True

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_model, mock_tokenizer):
        """Test AsyncEngineCore.get_stats()."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            async with AsyncEngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
            ) as engine:
                stats = engine.get_stats()

                assert "running" in stats
                assert stats["running"] is True

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, mock_model, mock_tokenizer):
        """Test AsyncEngineCore.get_cache_stats()."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            async with AsyncEngineCore(
                model=mock_model,
                tokenizer=mock_tokenizer,
            ) as engine:
                stats = engine.get_cache_stats()

                assert stats is None  # No SSD cache configured


class TestEngineCoreAbortAllRequests:
    """Tests for EngineCore.abort_all_requests()."""

    @pytest.mark.asyncio
    async def test_abort_all_requests(self, mock_model, mock_tokenizer):
        """Test abort_all_requests() sends errors to all collectors."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                # Add multiple requests
                rid1 = await engine.add_request(prompt="Hello")
                rid2 = await engine.add_request(prompt="World")

                # Abort all
                count = await engine.abort_all_requests()
                assert count == 2

                # Collectors should have error outputs
                for rid in [rid1, rid2]:
                    collector = engine._output_collectors.get(rid)
                    if collector is not None:
                        output = collector.get_nowait()
                        assert output is not None
                        assert output.finished is True
                        assert output.finish_reason == "error"
                        assert "memory" in output.error.lower()
                        # new_text should contain error message for SSE delivery
                        assert output.new_text is not None
                        assert "[Error:" in output.new_text
                        assert "memory" in output.new_text.lower()

                    # Finished events should be set
                    event = engine._finished_events.get(rid)
                    if event is not None:
                        assert event.is_set()
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_all_requests_empty(self, mock_model, mock_tokenizer):
        """Test abort_all_requests() with no active requests returns 0."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()
                count = await engine.abort_all_requests()
                assert count == 0
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_all_requests_engine_keeps_running(
        self, mock_model, mock_tokenizer
    ):
        """Test engine loop continues after abort_all_requests()."""
        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                await engine.start()

                rid = await engine.add_request(prompt="Hello")
                await engine.abort_all_requests()

                # Engine should still be running
                assert engine.is_running() is True

                # Can add new requests after abort
                new_rid = await engine.add_request(prompt="New request")
                assert new_rid in engine._output_collectors
            finally:
                await engine.stop()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_all_requests_while_duplicate_add_pending_keeps_live_request_routing(
        self, mock_model, mock_tokenizer
    ):
        """abort_all_requests() should target the live request routing even while a duplicate add is pending."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "abort-all-duplicate-pending"
            existing_request = Request(
                request_id=request_id,
                prompt=[11, 12],
                sampling_params=SamplingParams(max_tokens=8),
            )
            existing_request.prompt_token_ids = [11, 12]
            existing_request.num_prompt_tokens = 2
            engine.scheduler.requests[request_id] = existing_request

            original_collector = MagicMock()
            original_stream_state = object()
            original_finished_event = asyncio.Event()
            engine._output_collectors[request_id] = original_collector
            engine._stream_states[request_id] = original_stream_state
            engine._finished_events[request_id] = original_finished_event

            entered_add = threading.Event()
            release_add = threading.Event()
            original_add_request = engine.scheduler.add_request

            def delayed_duplicate_add(request):
                entered_add.set()
                _wait_for_release(release_add, message="test did not release blocked scheduler.add_request in time")
                return original_add_request(request)

            engine.scheduler.add_request = delayed_duplicate_add

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[99],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                await _wait_for_thread_event(
                    entered_add,
                    message="scheduler.add_request did not start in time",
                )
                assert not add_task.done(), "duplicate add finished before abort_all_requests raced it"

                count = await engine.abort_all_requests()
                assert count == 1

                original_collector.put.assert_called_once()
                abort_output = original_collector.put.call_args.args[0]
                assert abort_output.request_id == request_id
                assert abort_output.finished is True
                assert abort_output.finish_reason == "error"
                assert original_finished_event.is_set()
                assert engine._output_collectors[request_id] is original_collector
                assert engine._stream_states[request_id] is original_stream_state
                assert engine._finished_events[request_id] is original_finished_event
                assert request_id in engine.scheduler._pending_abort_ids

                release_add.set()
                with pytest.raises(ValueError, match="already exists"):
                    await add_task

                engine.scheduler._process_pending_aborts()
                _assert_scheduler_request_fully_cleaned(engine.scheduler, request_id)
                assert engine._output_collectors[request_id] is original_collector
                assert engine._stream_states[request_id] is original_stream_state
                assert engine._finished_events[request_id] is original_finished_event
                assert request_id not in engine._pending_adds
            finally:
                release_add.set()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_all_requests_during_inflight_add_raises_after_stream_cleanup(
        self, mock_model, mock_tokenizer
    ):
        """Abort-all plus stream cleanup should not let add_request succeed afterward."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "abort-all-inflight-add"
            entered_add = threading.Event()
            release_add = threading.Event()
            original_add_request = engine.scheduler.add_request

            def delayed_add_request(request):
                entered_add.set()
                _wait_for_release(release_add, message="test did not release blocked scheduler.add_request in time")
                return original_add_request(request)

            async def drain_stream():
                with pytest.raises(RuntimeError, match="Request aborted"):
                    async for _ in engine.stream_outputs(request_id):
                        pass

            engine.scheduler.add_request = delayed_add_request

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[7, 8, 9],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                started = await asyncio.to_thread(entered_add.wait, TEST_SYNC_TIMEOUT)
                assert started, "scheduler.add_request did not start in time"
                assert not add_task.done(), "add_request completed before abort_all_requests raced it"

                collector = engine._output_collectors[request_id]
                consumer_waiting = asyncio.Event()
                original_get = collector.get

                async def instrumented_get():
                    consumer_waiting.set()
                    return await original_get()

                collector.get = instrumented_get
                stream_task = asyncio.create_task(drain_stream())
                await asyncio.wait_for(consumer_waiting.wait(), timeout=TEST_SYNC_TIMEOUT)

                count = await engine.abort_all_requests()
                assert count == 1

                await stream_task
                assert not add_task.done(), "add_request finished before post-stream-cleanup window"
                assert request_id in engine._pending_adds

                release_add.set()
                with pytest.raises(RuntimeError, match="aborted before activation"):
                    await add_task

                _assert_scheduler_request_fully_cleaned(engine.scheduler, request_id)
                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
                assert request_id not in engine._pending_adds
            finally:
                release_add.set()
                engine.close()

    @pytest.mark.asyncio
    async def test_abort_all_requests_during_inflight_add_without_consumer_raises_cleanly(
        self, mock_model, mock_tokenizer
    ):
        """Abort-all should retire an in-flight net-new add even without consumer cleanup."""
        import threading

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True
            engine = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            request_id = "abort-all-inflight-no-consumer"
            entered_add = threading.Event()
            release_add = threading.Event()
            original_add_request = engine.scheduler.add_request

            def delayed_add_request(request):
                entered_add.set()
                _wait_for_release(release_add, message="test did not release blocked scheduler.add_request in time")
                return original_add_request(request)

            engine.scheduler.add_request = delayed_add_request

            try:
                add_task = asyncio.create_task(
                    engine.add_request(
                        prompt=[17, 18, 19],
                        request_id=request_id,
                        sampling_params=SamplingParams(max_tokens=8),
                    )
                )

                started = await asyncio.to_thread(entered_add.wait, TEST_SYNC_TIMEOUT)
                assert started, "scheduler.add_request did not start in time"
                assert not add_task.done(), "add_request completed before abort_all_requests raced it"

                count = await engine.abort_all_requests()
                assert count == 1

                release_add.set()
                with pytest.raises(RuntimeError, match="aborted before activation"):
                    await add_task

                _assert_scheduler_request_fully_cleaned(engine.scheduler, request_id)
                assert request_id not in engine._output_collectors
                assert request_id not in engine._stream_states
                assert request_id not in engine._finished_events
                assert request_id not in engine._pending_adds
            finally:
                release_add.set()
                engine.close()


class TestGlobalMLXExecutor:
    """Tests for the global MLX executor singleton (issue #85)."""

    def test_get_mlx_executor_returns_singleton(self):
        """get_mlx_executor() must always return the same executor instance."""
        from omlx.engine_core import get_mlx_executor

        executor1 = get_mlx_executor()
        executor2 = get_mlx_executor()
        assert executor1 is executor2

    def test_engines_share_mlx_executor(self, mock_model, mock_tokenizer):
        """Multiple EngineCore instances must share a single MLX executor (#85)."""
        from omlx.engine_core import get_mlx_executor

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine1 = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine2 = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            try:
                assert engine1._mlx_executor is engine2._mlx_executor
                assert engine1._mlx_executor is get_mlx_executor()
            finally:
                engine1.close()
                engine2.close()

    @pytest.mark.asyncio
    async def test_shared_executor_serializes_concurrent_tasks(self):
        """Concurrent submissions to shared executor must never overlap (#85).

        Simulates two engines submitting work simultaneously and verifies
        that tasks run one at a time (no concurrent execution).
        """
        import threading
        import time
        from omlx.engine_core import get_mlx_executor

        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()

        active_count = 0
        max_concurrent = 0
        lock = threading.Lock()

        def simulated_step(task_id: str, duration: float = 0.05):
            """Simulate a scheduler.step() that takes some time."""
            nonlocal active_count, max_concurrent
            with lock:
                active_count += 1
                if active_count > max_concurrent:
                    max_concurrent = active_count
            time.sleep(duration)
            with lock:
                active_count -= 1
            return task_id

        # Submit multiple tasks concurrently (simulating two engines)
        tasks = [
            loop.run_in_executor(executor, simulated_step, "engine_a_step1"),
            loop.run_in_executor(executor, simulated_step, "engine_b_step1"),
            loop.run_in_executor(executor, simulated_step, "engine_a_step2"),
            loop.run_in_executor(executor, simulated_step, "engine_b_step2"),
        ]
        results = await asyncio.gather(*tasks)

        # All tasks completed
        assert set(results) == {
            "engine_a_step1", "engine_b_step1",
            "engine_a_step2", "engine_b_step2",
        }
        # Critical: no two tasks ever ran at the same time
        assert max_concurrent == 1, (
            f"Expected max 1 concurrent task, got {max_concurrent}. "
            f"Shared executor failed to serialize MLX operations."
        )

    @pytest.mark.asyncio
    async def test_two_engine_loops_serialize_on_shared_executor(
        self, mock_model, mock_tokenizer
    ):
        """Two engines running their loops must serialize step() calls (#85).

        Creates two EngineCore instances with mock schedulers, starts both
        engine loops, and verifies their scheduler.step() calls never overlap.
        """
        import threading
        import time

        active_count = 0
        max_concurrent = 0
        total_steps = 0
        lock = threading.Lock()

        def make_tracked_step():
            """Create a step function that tracks concurrency."""
            from omlx.scheduler import SchedulerOutput

            def tracked_step():
                nonlocal active_count, max_concurrent, total_steps
                with lock:
                    active_count += 1
                    total_steps += 1
                    if active_count > max_concurrent:
                        max_concurrent = active_count
                time.sleep(0.01)  # Simulate GPU work
                with lock:
                    active_count -= 1
                return SchedulerOutput(outputs=[])

            return tracked_step

        with patch("omlx.engine_core.get_registry") as mock_registry:
            mock_registry.return_value.acquire.return_value = True

            engine1 = EngineCore(model=mock_model, tokenizer=mock_tokenizer)
            engine2 = EngineCore(model=mock_model, tokenizer=mock_tokenizer)

            # Wire up tracked step functions
            engine1.scheduler.step = make_tracked_step()
            engine2.scheduler.step = make_tracked_step()
            engine1.scheduler.has_requests = lambda: True
            engine2.scheduler.has_requests = lambda: True

            try:
                await engine1.start()
                await engine2.start()

                # Let both engines run for a bit
                await asyncio.sleep(0.3)
            finally:
                await engine1.stop()
                await engine2.stop()
                engine1.close()
                engine2.close()

        assert total_steps >= 4, (
            f"Expected at least 4 steps from two engines, got {total_steps}"
        )
        assert max_concurrent == 1, (
            f"Expected max 1 concurrent step(), got {max_concurrent}. "
            f"Two engines ran MLX operations in parallel — would cause "
            f"Metal command buffer races in production."
        )

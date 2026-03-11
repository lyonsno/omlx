# SPDX-License-Identifier: Apache-2.0
"""
Tests for Scheduler module.

Tests cover:
- SchedulerConfig: default values, custom values
- SchedulerOutput: dataclass behavior
- Scheduler initialization with mock model/tokenizer
- add_request(): adding requests, tokenization
- abort_request(): aborting waiting/running requests
- has_requests(), get_num_waiting(), get_num_running()
- get_request(): request lookup
- get_stats(): statistics

Note: BatchGenerator is mocked; step() is too complex for unit tests.
"""

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from omlx.request import Request, RequestOutput, RequestStatus, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig, SchedulerOutput, SchedulingPolicy


class TestSchedulerConfig:
    """Tests for SchedulerConfig dataclass."""

    def test_default_values(self):
        """Test SchedulerConfig has correct defaults."""
        config = SchedulerConfig()

        assert config.max_num_seqs == 256
        assert config.max_num_batched_tokens == 8192
        assert config.policy == SchedulingPolicy.FCFS
        assert config.completion_batch_size == 32
        assert config.prefill_step_size == 2048
        assert config.paged_cache_block_size == 256
        assert config.max_cache_blocks is None
        assert config.initial_cache_blocks == 256
        assert config.paged_ssd_cache_dir is None
        assert config.paged_ssd_cache_max_size == 100 * 1024 * 1024 * 1024  # 100GB
        assert config.model_name == ""
        assert config.gc_cleanup_interval == 0
        assert config.mlx_cache_cleanup_interval == 32

    def test_custom_values(self):
        """Test SchedulerConfig with custom values."""
        config = SchedulerConfig(
            max_num_seqs=128,
            max_num_batched_tokens=4096,
            policy=SchedulingPolicy.PRIORITY,
            completion_batch_size=16,
            prefill_step_size=1024,
            paged_cache_block_size=128,
            max_cache_blocks=500,
            initial_cache_blocks=100,
            paged_ssd_cache_dir="/tmp/cache",
            paged_ssd_cache_max_size=50 * 1024 * 1024 * 1024,
            model_name="test-model",
            gc_cleanup_interval=5,
            mlx_cache_cleanup_interval=20,
        )

        assert config.max_num_seqs == 128
        assert config.max_num_batched_tokens == 4096
        assert config.policy == SchedulingPolicy.PRIORITY
        assert config.completion_batch_size == 16
        assert config.prefill_step_size == 1024
        assert config.paged_cache_block_size == 128
        assert config.max_cache_blocks == 500
        assert config.initial_cache_blocks == 100
        assert config.paged_ssd_cache_dir == "/tmp/cache"
        assert config.paged_ssd_cache_max_size == 50 * 1024 * 1024 * 1024
        assert config.model_name == "test-model"
        assert config.gc_cleanup_interval == 5
        assert config.mlx_cache_cleanup_interval == 20


class TestSchedulingPolicy:
    """Tests for SchedulingPolicy enum."""

    def test_fcfs_policy(self):
        """Test FCFS policy value."""
        assert SchedulingPolicy.FCFS.value == "fcfs"

    def test_priority_policy(self):
        """Test Priority policy value."""
        assert SchedulingPolicy.PRIORITY.value == "priority"


class TestSchedulerOutput:
    """Tests for SchedulerOutput dataclass."""

    def test_default_values(self):
        """Test SchedulerOutput has correct defaults."""
        output = SchedulerOutput()

        assert output.scheduled_request_ids == []
        assert output.num_scheduled_tokens == 0
        assert output.finished_request_ids == set()
        assert output.outputs == []
        assert output.has_work is False

    def test_custom_values(self):
        """Test SchedulerOutput with custom values."""
        outputs = [
            RequestOutput(
                request_id="req-1",
                new_token_ids=[100],
                new_text="hello",
            )
        ]
        output = SchedulerOutput(
            scheduled_request_ids=["req-1", "req-2"],
            num_scheduled_tokens=100,
            finished_request_ids={"req-1"},
            outputs=outputs,
            has_work=True,
        )

        assert output.scheduled_request_ids == ["req-1", "req-2"]
        assert output.num_scheduled_tokens == 100
        assert output.finished_request_ids == {"req-1"}
        assert len(output.outputs) == 1
        assert output.outputs[0].request_id == "req-1"
        assert output.has_work is True


class TestSchedulerInitialization:
    """Tests for Scheduler initialization."""

    def test_init_with_defaults(self, mock_model, mock_tokenizer):
        """Test Scheduler initializes with default config."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        assert scheduler.model is mock_model
        assert scheduler.tokenizer is mock_tokenizer
        assert isinstance(scheduler.config, SchedulerConfig)
        assert isinstance(scheduler.waiting, deque)
        assert len(scheduler.waiting) == 0
        assert scheduler.running == {}
        assert scheduler.requests == {}
        assert scheduler.finished_req_ids == set()
        assert scheduler.request_id_to_uid == {}
        assert scheduler.uid_to_request_id == {}
        assert scheduler.batch_generator is None

    def test_init_with_custom_config(self, mock_model, mock_tokenizer):
        """Test Scheduler initializes with custom config."""
        config = SchedulerConfig(
            max_num_seqs=64,
        )
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
        )

        assert scheduler.config.max_num_seqs == 64

    def test_init_statistics_zero(self, mock_model, mock_tokenizer):
        """Test Scheduler initializes with zero statistics."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        assert scheduler.num_requests_processed == 0
        assert scheduler.total_prompt_tokens == 0
        assert scheduler.total_completion_tokens == 0


class TestSchedulerAddRequest:
    """Tests for Scheduler.add_request()."""

    def test_add_request_with_string_prompt(self, mock_model, mock_tokenizer):
        """Test adding a request with string prompt."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello, world!",
            sampling_params=SamplingParams(max_tokens=50),
        )
        scheduler.add_request(request)

        assert "test-001" in scheduler.requests
        assert request in scheduler.waiting
        assert request.prompt_token_ids is not None
        assert len(request.prompt_token_ids) > 0
        assert request.num_prompt_tokens == len(request.prompt_token_ids)

    def test_add_request_with_token_ids(self, mock_model, mock_tokenizer):
        """Test adding a request with pre-tokenized prompt."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        token_ids = [1, 100, 200, 300]
        request = Request(
            request_id="test-002",
            prompt=token_ids,
            sampling_params=SamplingParams(max_tokens=50),
        )
        # Pre-set token IDs
        request.prompt_token_ids = token_ids
        request.num_prompt_tokens = len(token_ids)

        scheduler.add_request(request)

        assert "test-002" in scheduler.requests
        assert request.prompt_token_ids == token_ids
        assert request.num_prompt_tokens == 4

    def test_add_duplicate_request_raises(self, mock_model, mock_tokenizer):
        """Test adding duplicate request raises ValueError."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)

        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_request(request)

    def test_add_multiple_requests(self, mock_model, mock_tokenizer):
        """Test adding multiple requests."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        for i in range(5):
            request = Request(
                request_id=f"test-{i:03d}",
                prompt=f"Prompt {i}",
                sampling_params=SamplingParams(),
            )
            scheduler.add_request(request)

        assert len(scheduler.requests) == 5
        assert len(scheduler.waiting) == 5

    def test_add_request_exact_cache_hit_trims_one_token(
        self, mock_model, mock_tokenizer
    ):
        """Exact cache hit should use (N-1) cache + last token for kickoff."""
        from omlx.cache.paged_cache import BlockTable

        class TrimCache:
            def __init__(self):
                self.trim_calls = 0

            def trim(self, n):
                self.trim_calls += 1
                return n

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(request_id="req-exact", block_ids=[1, 2], num_tokens=4)
        trim_cache_a = TrimCache()
        trim_cache_b = TrimCache()

        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [trim_cache_a, trim_cache_b]

        request = Request(
            request_id="req-exact",
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 3
        assert request.remaining_tokens == [14]
        assert request.prompt_cache is not None
        assert trim_cache_a.trim_calls == 1
        assert trim_cache_b.trim_calls == 1

    def test_add_request_exact_cache_hit_falls_back_if_not_trimmable(
        self, mock_model, mock_tokenizer
    ):
        """Exact cache hit should fallback when any layer cannot trim."""
        from omlx.cache.paged_cache import BlockTable

        class NonTrimmableCache:
            pass

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(request_id="req-fallback", block_ids=[3], num_tokens=4)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [NonTrimmableCache()]

        request = Request(
            request_id="req-fallback",
            prompt=[21, 22, 23, 24],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 0
        assert request.remaining_tokens == [21, 22, 23, 24]
        assert request.prompt_cache is None
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with("req-fallback")
        assert request.block_table is None
        assert request.shared_prefix_blocks == 0

    def test_add_request_exact_cache_hit_rotating_only_falls_back_policy(
        self, mock_model, mock_tokenizer
    ):
        """Pure rotating exact-hit policy should fail closed to full prefill."""
        from omlx.cache.paged_cache import BlockTable

        class RotatingKVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                if not self.can_rewind(n):
                    return False
                self.offset -= n
                return True

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(request_id="req-rot-only", block_ids=[8], num_tokens=4)
        rotating_layer = RotatingKVCache(offset=4)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [rotating_layer]

        request = Request(
            request_id="req-rot-only",
            prompt=[51, 52, 53, 54],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 0
        assert request.remaining_tokens == [51, 52, 53, 54]
        assert request.prompt_cache is None
        assert request.block_table is None
        assert request.shared_prefix_blocks == 0
        assert rotating_layer.rewind_calls == []
        assert rotating_layer.offset == 4
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with(
            "req-rot-only"
        )

    def test_add_request_exact_cache_hit_mixed_rotating_rewinds_one_token(
        self, mock_model, mock_tokenizer
    ):
        """Mixed exact hit should rewind both KV and rotating layers by one token."""
        from omlx.cache.paged_cache import BlockTable

        class KVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                if not self.can_rewind(n):
                    return False
                self.offset -= n
                return True

        class RotatingKVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                if not self.can_rewind(n):
                    return False
                self.offset -= n
                return True

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(request_id="req-rotating", block_ids=[9], num_tokens=4)
        kv_layer = KVCache(offset=4)
        rotating_layer = RotatingKVCache(offset=4)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [kv_layer, rotating_layer]

        request = Request(
            request_id="req-rotating",
            prompt=[31, 32, 33, 34],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 3
        assert request.remaining_tokens == [34]
        assert request.prompt_cache is not None
        assert request.block_table is not None
        assert request.block_table.request_id == block_table.request_id
        assert request.block_table.block_ids == block_table.block_ids
        assert request.block_table.num_tokens == block_table.num_tokens
        assert request.shared_prefix_blocks == len(block_table.block_ids)
        assert kv_layer.rewind_calls
        assert rotating_layer.rewind_calls
        assert 1 in kv_layer.rewind_calls
        assert 1 in rotating_layer.rewind_calls
        assert kv_layer.offset == 3
        assert rotating_layer.offset == 3
        scheduler.paged_cache_manager.delete_block_table.assert_not_called()

    def test_step_exact_cache_hit_mixed_rotating_kickoff_uses_last_token_with_rewound_cache(
        self, mock_model, mock_tokenizer
    ):
        """Exact-hit add_request() -> step() must hand off the rewound cache with only the last prompt token."""
        from omlx.cache.paged_cache import BlockTable

        class KVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                if not self.can_rewind(n):
                    return False
                self.offset -= n
                return True

        class RotatingKVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                if not self.can_rewind(n):
                    return False
                self.offset -= n
                return True

        class FakeBatchGenerator:
            def __init__(self):
                self.insert_calls = []
                self.active_batch = None

            def insert(
                self,
                token_batches,
                max_tokens=None,
                caches=None,
                samplers=None,
                logits_processors=None,
            ):
                self.insert_calls.append(
                    {
                        "token_batches": token_batches,
                        "max_tokens": max_tokens,
                        "caches": caches,
                    }
                )
                self.active_batch = MagicMock(uids=[123], cache=caches[0] if caches else None)
                return [123]

            def next(self):
                return []

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(
            request_id="req-step-kickoff",
            block_ids=[10],
            num_tokens=4,
        )
        kv_layer = KVCache(offset=4)
        rotating_layer = RotatingKVCache(offset=4)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [kv_layer, rotating_layer]

        request = Request(
            request_id="req-step-kickoff",
            prompt=[31, 32, 33, 34],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        fake_batch_generator = FakeBatchGenerator()
        scheduler.batch_generator = fake_batch_generator
        scheduler._ensure_batch_generator = MagicMock()

        assert request.remaining_tokens == [34]
        assert request.cached_tokens == 3
        assert request.prompt_cache == [kv_layer, rotating_layer]

        output = scheduler.step()

        assert output.scheduled_request_ids == ["req-step-kickoff"]
        assert len(fake_batch_generator.insert_calls) == 1
        insert_call = fake_batch_generator.insert_calls[0]
        assert insert_call["token_batches"] == [[34]]
        assert insert_call["caches"] == [request.prompt_cache]
        assert request.prompt_cache == [kv_layer, rotating_layer]
        assert kv_layer.offset == 3
        assert rotating_layer.offset == 3
        assert request.status == RequestStatus.RUNNING
        assert scheduler.running[request.request_id] is request
        scheduler.paged_cache_manager.delete_block_table.assert_not_called()

    def test_add_request_exact_cache_hit_mixed_rotating_unrewindable_fails_closed(
        self, mock_model, mock_tokenizer
    ):
        """Mixed exact hit should fail closed without partial mutation."""
        from omlx.cache.paged_cache import BlockTable

        class KVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                if not self.can_rewind(n):
                    return False
                self.offset -= n
                return True

        class RotatingKVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.rewind_calls = []

            def can_rewind(self, n):
                return False

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(request_id="req-rotating-fail", block_ids=[10], num_tokens=4)
        kv_layer = KVCache(offset=4)
        rotating_layer = RotatingKVCache(offset=4)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [kv_layer, rotating_layer]

        request = Request(
            request_id="req-rotating-fail",
            prompt=[41, 42, 43, 44],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 0
        assert request.remaining_tokens == [41, 42, 43, 44]
        assert request.prompt_cache is None
        assert request.block_table is None
        assert request.shared_prefix_blocks == 0
        assert kv_layer.offset == 4
        assert rotating_layer.offset == 4
        assert kv_layer.rewind_calls == []
        assert rotating_layer.rewind_calls == []
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with(
            "req-rotating-fail"
        )

    def test_cache_sliceability_classification_consistency(self, mock_model, mock_tokenizer):
        """Mixed-sliceability and boundary-snapshot classification stay in sync."""

        class KVCache:
            pass

        class RotatingKVCache:
            pass

        class UnknownStatefulCache:
            def __init__(self):
                self.cache = [object()]

        class CacheList:
            def __init__(self, *caches):
                self.caches = tuple(caches)

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        kv_only = [KVCache()]
        rotating_only = [RotatingKVCache()]
        mixed = [KVCache(), RotatingKVCache()]
        nested_mixed = [CacheList(KVCache(), RotatingKVCache())]
        mixed_with_unknown_stateful = [KVCache(), UnknownStatefulCache()]

        assert scheduler._cache_list_needs_boundary_snapshot(kv_only) is False
        assert scheduler._cache_list_has_mixed_sliceability(kv_only) is False

        assert scheduler._cache_list_needs_boundary_snapshot(rotating_only) is True
        assert scheduler._cache_list_has_mixed_sliceability(rotating_only) is False

        assert scheduler._cache_list_needs_boundary_snapshot(mixed) is True
        assert scheduler._cache_list_has_mixed_sliceability(mixed) is True

        assert scheduler._cache_list_needs_boundary_snapshot(nested_mixed) is True
        assert scheduler._cache_list_has_mixed_sliceability(nested_mixed) is True

        assert (
            scheduler._cache_list_needs_boundary_snapshot(mixed_with_unknown_stateful)
            is True
        )
        assert (
            scheduler._cache_list_has_mixed_sliceability(mixed_with_unknown_stateful)
            is False
        )

    def test_add_request_exact_cache_hit_mixed_rewind_mutation_failure_rolls_back(
        self, mock_model, mock_tokenizer
    ):
        """
        Adversarial reproducer: mutation-phase rewind failure must not partially mutate.

        This targets the path where preflight passes for all layers, but a later
        layer returns rewind=False during mutation.
        """
        from omlx.cache.paged_cache import BlockTable

        class KVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                return True

        class RotatingKVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                # Preflight passes.
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                # Mutation-phase failure after earlier layers may have mutated.
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(
            request_id="req-mutation-fail",
            block_ids=[15],
            num_tokens=4,
        )
        kv_layer = KVCache(offset=4)
        rotating_layer = RotatingKVCache(offset=4)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [kv_layer, rotating_layer]

        request = Request(
            request_id="req-mutation-fail",
            prompt=[91, 92, 93, 94],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        # Fail-closed external behavior.
        assert request.cached_tokens == 0
        assert request.remaining_tokens == [91, 92, 93, 94]
        assert request.prompt_cache is None
        assert request.block_table is None
        assert request.shared_prefix_blocks == 0
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with(
            "req-mutation-fail"
        )

        # Path-tight checks: preflight and mutation were both attempted.
        assert kv_layer.can_rewind_calls == [1]
        assert rotating_layer.can_rewind_calls == [1]
        assert kv_layer.rewind_calls == [1]
        assert rotating_layer.rewind_calls == [1]

        # Critical invariant: no partial mutation on failure.
        assert kv_layer.offset == 4
        assert rotating_layer.offset == 4

    def test_add_request_exact_cache_hit_mixed_rewind_mutation_failure_restores_state_payload(
        self, mock_model, mock_tokenizer
    ):
        """
        Rewind rollback must restore mutable cache state payloads, not just offsets.
        """
        from omlx.cache.paged_cache import BlockTable

        class KVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []
                self._state = [101, 102, 103, 104]

            @property
            def state(self):
                return list(self._state)

            @state.setter
            def state(self, v):
                self._state = list(v)

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                # Simulate future rewind implementation that mutates payload.
                if self._state:
                    self._state.pop()
                return True

        class RotatingKVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(
            request_id="req-state-rollback",
            block_ids=[17],
            num_tokens=4,
        )
        kv_layer = KVCache(offset=4)
        rotating_layer = RotatingKVCache(offset=4)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [kv_layer, rotating_layer]

        request = Request(
            request_id="req-state-rollback",
            prompt=[111, 112, 113, 114],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        # Fail-closed external behavior.
        assert request.cached_tokens == 0
        assert request.remaining_tokens == [111, 112, 113, 114]
        assert request.prompt_cache is None
        assert request.block_table is None
        assert request.shared_prefix_blocks == 0
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with(
            "req-state-rollback"
        )

        # Path-tight checks.
        assert kv_layer.can_rewind_calls == [1]
        assert rotating_layer.can_rewind_calls == [1]
        assert kv_layer.rewind_calls == [1]
        assert rotating_layer.rewind_calls == [1]

        # Critical invariant: rollback restores payload as well as metadata.
        assert kv_layer.offset == 4
        assert kv_layer.state == [101, 102, 103, 104]
        assert rotating_layer.offset == 4

    def test_add_request_exact_cache_hit_mixed_rewind_mutation_failure_restores_meta_state_payload(
        self, mock_model, mock_tokenizer
    ):
        """
        Add-time exact-hit rewind rollback must restore meta_state payloads fail-closed.

        This covers the real Scheduler.add_request() rewind path rather than only the
        lower-level helper, and also checks block-table cleanup and prompt-cache reset.
        """
        from omlx.cache.paged_cache import BlockTable

        class KVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                return True

        class RotatingKVCache:
            def __init__(
                self,
                offset=4,
                fail_state_restore=False,
                fail_rewind=False,
            ):
                self.offset = offset
                self.fail_state_restore = fail_state_restore
                self.fail_rewind = fail_rewind
                self.can_rewind_calls = []
                self.rewind_calls = []
                shared_payload = {"tokens": [201, 202, 203, 204]}
                self._meta_state = shared_payload
                self.keys = shared_payload
                self.last_attempted_meta_state = None

            @property
            def meta_state(self):
                return self._meta_state

            @meta_state.setter
            def meta_state(self, v):
                self.last_attempted_meta_state = v
                if self.fail_state_restore and isinstance(v, dict) and isinstance(v.get("tokens"), list):
                    v["tokens"].pop()
                    raise RuntimeError("meta_state setter failed after mutating input")
                self._meta_state = v

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                if self.fail_rewind:
                    return False
                self.offset -= n
                self._meta_state["tokens"].pop()
                return True

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(
            request_id="req-meta-state-rollback",
            block_ids=[19],
            num_tokens=4,
        )
        sliceable_layer = KVCache(offset=4)
        kv_layer = RotatingKVCache(offset=4, fail_state_restore=True)
        failing_layer = RotatingKVCache(offset=4, fail_rewind=True)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [
            sliceable_layer,
            kv_layer,
            failing_layer,
        ]

        request = Request(
            request_id="req-meta-state-rollback",
            prompt=[121, 122, 123, 124],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 0
        assert request.remaining_tokens == [121, 122, 123, 124]
        assert request.prompt_cache is None
        assert request.block_table is None
        assert request.shared_prefix_blocks == 0
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with(
            "req-meta-state-rollback"
        )

        assert scheduler._cache_list_has_mixed_sliceability(
            [sliceable_layer, kv_layer, failing_layer]
        ) is True
        assert sliceable_layer.can_rewind_calls == [1]
        assert sliceable_layer.rewind_calls == [1]
        assert kv_layer.can_rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert kv_layer.rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]

        assert sliceable_layer.offset == 4
        assert kv_layer.offset == 4
        assert failing_layer.offset == 4
        assert kv_layer.keys["tokens"] == [201, 202, 203, 204]
        assert kv_layer.last_attempted_meta_state is not None
        assert kv_layer.last_attempted_meta_state["tokens"] == [201, 202, 203]
        assert kv_layer.last_attempted_meta_state is not kv_layer.keys

    def test_add_request_exact_cache_hit_mixed_rewind_mutation_failure_restores_custom_meta_state_fallback_clone(
        self, mock_model, mock_tokenizer
    ):
        """
        Add-time rewind rollback should restore fallback-cloned custom meta_state payloads.

        This pins the end-to-end add_request() flow, including prompt-cache/block-table
        cleanup, not just the lower-level helper logic.
        """
        from omlx.cache.paged_cache import BlockTable

        class FallbackCloneMeta:
            def __init__(self, tokens):
                self.tokens = list(tokens)

            def __deepcopy__(self, memo):
                raise TypeError("meta payload does not support deepcopy")

        class KVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                return True

        class RotatingKVCache:
            def __init__(self, offset=4, fail_rewind=False):
                self.offset = offset
                self.fail_rewind = fail_rewind
                self.can_rewind_calls = []
                self.rewind_calls = []
                self.meta_state = {"payload": FallbackCloneMeta([1, 2, 3, 4])}

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                if self.fail_rewind:
                    return False
                self.offset -= n
                self.meta_state["payload"].tokens.pop()
                return True

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(
            request_id="req-custom-meta-fallback",
            block_ids=[21],
            num_tokens=4,
        )
        sliceable_layer = KVCache(offset=4)
        custom_layer = RotatingKVCache(offset=4)
        original_meta_payload = custom_layer.meta_state["payload"]
        failing_layer = RotatingKVCache(offset=4, fail_rewind=True)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [
            sliceable_layer,
            custom_layer,
            failing_layer,
        ]

        request = Request(
            request_id="req-custom-meta-fallback",
            prompt=[131, 132, 133, 134],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 0
        assert request.remaining_tokens == [131, 132, 133, 134]
        assert request.prompt_cache is None
        assert request.block_table is None
        assert request.shared_prefix_blocks == 0
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with(
            "req-custom-meta-fallback"
        )

        assert scheduler._cache_list_has_mixed_sliceability(
            [sliceable_layer, custom_layer, failing_layer]
        ) is True
        assert sliceable_layer.can_rewind_calls == [1]
        assert sliceable_layer.rewind_calls == [1]
        assert custom_layer.can_rewind_calls == [1]
        assert custom_layer.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert sliceable_layer.offset == 4
        assert custom_layer.offset == 4
        assert custom_layer.meta_state["payload"].tokens == [1, 2, 3, 4]
        assert isinstance(custom_layer.meta_state["payload"], FallbackCloneMeta)
        assert custom_layer.meta_state["payload"] is not original_meta_payload
        assert failing_layer.offset == 4

    def test_add_request_exact_cache_hit_mixed_rewind_mutation_failure_rolls_back_nested_cache_list(
        self, mock_model, mock_tokenizer
    ):
        """
        Adversarial nested CacheList reproducer for recursive rewind path.

        Ensures recursion branch still enforces no partial mutation on failure.
        """
        from omlx.cache.paged_cache import BlockTable

        class KVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                return True

        class RotatingKVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        class CacheList:
            def __init__(self, *caches):
                self.caches = tuple(caches)

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(
            request_id="req-mutation-fail-nested",
            block_ids=[16],
            num_tokens=4,
        )
        kv_layer = KVCache(offset=4)
        rotating_layer = RotatingKVCache(offset=4)
        nested_cache = CacheList(kv_layer, rotating_layer)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [nested_cache]

        request = Request(
            request_id="req-mutation-fail-nested",
            prompt=[95, 96, 97, 98],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        # Fail-closed external behavior.
        assert request.cached_tokens == 0
        assert request.remaining_tokens == [95, 96, 97, 98]
        assert request.prompt_cache is None
        assert request.block_table is None
        assert request.shared_prefix_blocks == 0
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with(
            "req-mutation-fail-nested"
        )

        # Recursive path-tight checks.
        assert kv_layer.can_rewind_calls == [1]
        assert rotating_layer.can_rewind_calls == [1]
        assert kv_layer.rewind_calls == [1]
        assert rotating_layer.rewind_calls == [1]

        # Critical invariant: no partial mutation on failure.
        assert kv_layer.offset == 4
        assert rotating_layer.offset == 4

    def test_add_request_exact_cache_hit_unknown_cache_falls_back_fail_closed(
        self, mock_model, mock_tokenizer
    ):
        """Unknown cache classes should fail closed on exact hit."""
        from omlx.cache.paged_cache import BlockTable

        class UnknownCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.cache = [object()]
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                if not self.can_rewind(n):
                    return False
                self.offset -= n
                return True

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(request_id="req-unknown", block_ids=[11], num_tokens=4)
        unknown_layer = UnknownCache(offset=4)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [unknown_layer]

        request = Request(
            request_id="req-unknown",
            prompt=[61, 62, 63, 64],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 0
        assert request.remaining_tokens == [61, 62, 63, 64]
        assert request.prompt_cache is None
        assert request.block_table is None
        assert request.shared_prefix_blocks == 0
        assert unknown_layer.rewind_calls == []
        assert unknown_layer.offset == 4
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with("req-unknown")

    def test_add_request_exact_cache_hit_mixed_unknown_cache_fails_closed(self, mock_model, mock_tokenizer):
        """Unknown stateful layer mixed with KV must fail closed (no rewind attempts)."""
        from omlx.cache.paged_cache import BlockTable

        class KVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                if not self.can_rewind(n):
                    return False
                self.offset -= n
                return True

        class UnknownStatefulCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.cache = [object()]
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                if not self.can_rewind(n):
                    return False
                self.offset -= n
                return True

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(
            request_id="req-unknown-mixed",
            block_ids=[111],
            num_tokens=4,
        )
        kv_layer = KVCache(offset=4)
        unknown_layer = UnknownStatefulCache(offset=4)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [kv_layer, unknown_layer]

        request = Request(
            request_id="req-unknown-mixed",
            prompt=[161, 162, 163, 164],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 0
        assert request.remaining_tokens == [161, 162, 163, 164]
        assert request.prompt_cache is None
        assert request.block_table is None
        assert request.shared_prefix_blocks == 0
        assert kv_layer.rewind_calls == []
        assert unknown_layer.rewind_calls == []
        assert kv_layer.offset == 4
        assert unknown_layer.offset == 4
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with(
            "req-unknown-mixed"
        )

    def test_add_request_exact_cache_hit_unknown_wrapper_mixed_fails_closed(
        self, mock_model, mock_tokenizer
    ):
        """Unknown wrapper with .caches must fail closed even if children are known."""
        from omlx.cache.paged_cache import BlockTable

        class KVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                if not self.can_rewind(n):
                    return False
                self.offset -= n
                return True

        class RotatingKVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                if not self.can_rewind(n):
                    return False
                self.offset -= n
                return True

        class UnknownWrapper:
            def __init__(self, *caches):
                self.caches = tuple(caches)
                self.wrapper_offset = 4

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(
            request_id="req-unknown-wrapper",
            block_ids=[41],
            num_tokens=4,
        )
        kv_layer = KVCache(offset=4)
        rotating_layer = RotatingKVCache(offset=4)
        wrapper = UnknownWrapper(kv_layer, rotating_layer)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [wrapper]

        request = Request(
            request_id="req-unknown-wrapper",
            prompt=[31, 32, 33, 34],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        scheduler.block_aware_cache.fetch_cache.assert_called_once_with(
            "req-unknown-wrapper",
            [31, 32, 33, 34],
            extra_keys=None,
        )
        scheduler.block_aware_cache.reconstruct_cache.assert_called_once_with(block_table)
        assert request.cached_tokens == 0
        assert request.remaining_tokens == [31, 32, 33, 34]
        assert request.prompt_cache is None
        assert request.block_table is None
        assert request.shared_prefix_blocks == 0
        assert kv_layer.rewind_calls == []
        assert rotating_layer.rewind_calls == []
        assert kv_layer.offset == 4
        assert rotating_layer.offset == 4
        scheduler.paged_cache_manager.delete_block_table.assert_called_once_with(
            "req-unknown-wrapper"
        )

    def test_add_request_exact_cache_hit_mixed_trim_only_legacy_rewinds_one_token(
        self, mock_model, mock_tokenizer
    ):
        """Mixed trim-only legacy caches should rewind kickoff by one token."""
        from omlx.cache.paged_cache import BlockTable

        class KVCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.trim_calls = 0

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.trim_calls += 1
                n = min(self.offset, n)
                self.offset -= n
                return n

        class RotatingKVCache:
            def __init__(self, offset=4, max_size=16):
                self.offset = offset
                self.max_size = max_size
                self.trim_calls = 0

            def is_trimmable(self):
                # Legacy rotating behavior: trimmable while unsaturated.
                return self.offset < self.max_size

            def trim(self, n):
                self.trim_calls += 1
                n = min(self.offset, n)
                self.offset -= n
                return n

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(request_id="req-mixed-legacy", block_ids=[12], num_tokens=4)
        kv_layer = KVCache(offset=4)
        rotating_layer = RotatingKVCache(offset=4, max_size=16)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [kv_layer, rotating_layer]

        request = Request(
            request_id="req-mixed-legacy",
            prompt=[71, 72, 73, 74],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 3
        assert request.remaining_tokens == [74]
        assert request.prompt_cache is not None
        assert request.block_table is not None
        assert request.shared_prefix_blocks == len(block_table.block_ids)
        assert kv_layer.trim_calls == 1
        assert rotating_layer.trim_calls == 1
        assert kv_layer.offset == 3
        assert rotating_layer.offset == 3
        scheduler.paged_cache_manager.delete_block_table.assert_not_called()

    def test_can_rewind_cache_tree_by_one_batch_kv_like_non_int_offset_uses_trim_capability(
        self, mock_model, mock_tokenizer
    ):
        """
        BatchKV-like trim-only caches should preflight rewind even when
        ``offset`` is not a Python int and ``size()`` reports 0.
        """

        class ScalarOffset:
            def __init__(self, value):
                self.value = value

        class BatchKVCache:
            def __init__(self, tokens=4):
                self.offset = ScalarOffset(tokens)
                self.logical_tokens = tokens

            def size(self):
                # Mirrors current mlx-lm BatchKVCache behavior via _BaseCache.size().
                return 0

            def is_trimmable(self):
                return self.logical_tokens > 0

            def trim(self, n):
                trimmed = min(self.logical_tokens, n)
                self.logical_tokens -= trimmed
                return trimmed

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        batch_kv_like = BatchKVCache(tokens=4)

        assert scheduler._can_rewind_cache_tree_by_one(batch_kv_like) is True

    def test_can_rewind_cache_tree_by_one_batch_kv_like_item_only_offset_uses_trim_capability(
        self, mock_model, mock_tokenizer
    ):
        """
        BatchKV-like trim-only caches should preflight rewind when ``offset``
        is exposed via ``.item()`` and ``size()`` reports 0.
        """

        class ItemOnlyScalar:
            def __init__(self, value):
                self._value = value

            def item(self):
                return self._value

        class BatchKVCache:
            def __init__(self, tokens=4):
                self.offset = ItemOnlyScalar(tokens)
                self.logical_tokens = tokens

            def size(self):
                return 0

            def is_trimmable(self):
                return self.logical_tokens > 0

            def trim(self, n):
                trimmed = min(self.logical_tokens, n)
                self.logical_tokens -= trimmed
                return trimmed

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        batch_kv_like = BatchKVCache(tokens=4)

        assert scheduler._can_rewind_cache_tree_by_one(batch_kv_like) is True

    def test_can_rewind_cache_tree_by_one_rewind_only_cache_api(
        self, mock_model, mock_tokenizer
    ):
        """
        Rewind preflight should accept rewind-only cache APIs.

        Fail-first target: mutation path already supports ``rewind(1)`` directly,
        so preflight must not require ``can_rewind`` or ``trim`` to exist.
        """

        class RewindOnlyCache:
            def __init__(self, offset=4):
                self.offset = offset
                self.rewind_calls = []

            def rewind(self, n):
                self.rewind_calls.append(n)
                if n > self.offset:
                    return False
                self.offset -= n
                return True

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        cache = RewindOnlyCache(offset=4)

        assert scheduler._can_rewind_cache_tree_by_one(cache) is True
        assert scheduler._rewind_cache_tree_by_one(cache) is True
        assert cache.rewind_calls == [1]
        assert cache.offset == 3

    def test_add_request_exact_cache_hit_mixed_batch_kv_like_rewinds_one_token(
        self, mock_model, mock_tokenizer
    ):
        """
        Mixed exact hit with BatchKV-like trim-only layer should rewind kickoff.

        Fail-first target: preflight must not reject rewindable trim-only layers
        solely because ``offset`` is non-int and ``size()`` returns 0.
        """
        from omlx.cache.paged_cache import BlockTable

        class ScalarOffset:
            def __init__(self, value):
                self.value = value

        class BatchKVCache:
            def __init__(self, tokens=4):
                self.offset = ScalarOffset(tokens)
                self.logical_tokens = tokens
                self.trim_calls = 0

            def size(self):
                return 0

            def is_trimmable(self):
                return self.logical_tokens > 0

            def trim(self, n):
                self.trim_calls += 1
                trimmed = min(self.logical_tokens, n)
                self.logical_tokens -= trimmed
                return trimmed

        class BatchRotatingKVCache:
            def __init__(self, tokens=4, max_size=4):
                self.offset = ScalarOffset(tokens)
                self._offset = tokens
                self.max_size = max_size
                self.trim_calls = 0

            def size(self):
                return 0

            def is_trimmable(self):
                # Saturated windows report non-trimmable in some legacy paths.
                return self._offset < self.max_size

            def trim(self, n):
                self.trim_calls += 1
                trimmed = min(self._offset, n)
                self._offset -= trimmed
                return trimmed

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(
            request_id="req-mixed-batch-kv-like",
            block_ids=[121],
            num_tokens=4,
        )
        batch_kv_like = BatchKVCache(tokens=4)
        batch_rotating = BatchRotatingKVCache(tokens=4, max_size=4)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [
            batch_kv_like,
            batch_rotating,
        ]

        request = Request(
            request_id="req-mixed-batch-kv-like",
            prompt=[171, 172, 173, 174],
            sampling_params=SamplingParams(max_tokens=16),
        )
        scheduler.add_request(request)

        assert request.cached_tokens == 3
        assert request.remaining_tokens == [174]
        assert request.prompt_cache is not None
        assert request.block_table is not None
        assert request.shared_prefix_blocks == len(block_table.block_ids)
        assert batch_kv_like.trim_calls == 1
        assert batch_rotating.trim_calls == 1
        assert batch_kv_like.logical_tokens == 3
        assert batch_rotating._offset == 3
        scheduler.paged_cache_manager.delete_block_table.assert_not_called()

    def test_rewind_prompt_cache_mutation_failure_restores_nested_in_place_state(
        self, mock_model, mock_tokenizer
    ):
        """Rollback must restore nested payloads mutated in place by rewind()."""

        class MutableStateKV:
            def __init__(self, offset=4):
                self.offset = offset
                self._state = {"tokens": [201, 202, 203, 204]}

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self._state = v

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.offset -= n
                # In-place nested mutation (common future footgun).
                self._state["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        kv_layer = MutableStateKV(offset=4)
        failing_layer = FailingLayer(offset=4)

        ok = scheduler._rewind_prompt_cache_for_generation([kv_layer, failing_layer])

        assert ok is False
        assert kv_layer.offset == 4
        assert kv_layer.state == {"tokens": [201, 202, 203, 204]}
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_restores_custom_mutable_state_object(
        self, mock_model, mock_tokenizer
    ):
        """Rollback must restore custom mutable objects embedded in state payload."""

        class MutablePayload:
            def __init__(self, tokens):
                self.tokens = list(tokens)

        class MutableStateKV:
            def __init__(self, offset=4):
                self.offset = offset
                self._state = {"payload": MutablePayload([301, 302, 303, 304])}

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self._state = v

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.offset -= n
                # In-place mutation inside a custom mutable object.
                self._state["payload"].tokens.pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        kv_layer = MutableStateKV(offset=4)
        failing_layer = FailingLayer(offset=4)

        ok = scheduler._rewind_prompt_cache_for_generation([kv_layer, failing_layer])

        assert ok is False
        assert kv_layer.offset == 4
        assert kv_layer.state["payload"].tokens == [301, 302, 303, 304]
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_restores_non_deepcopyable_payload(
        self, mock_model, mock_tokenizer
    ):
        """Rollback must remain transactional for non-deepcopyable payload objects."""

        class NonDeepcopyablePayload:
            def __init__(self, tokens):
                self.tokens = list(tokens)

            def __deepcopy__(self, memo):
                raise TypeError("payload does not support deepcopy")

        class MutableStateKV:
            def __init__(self, offset=4):
                self.offset = offset
                self._state = {"payload": NonDeepcopyablePayload([401, 402, 403, 404])}
                self.can_rewind_calls = []
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self._state = v

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["payload"].tokens.pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        kv_layer = MutableStateKV(offset=4)
        failing_layer = FailingLayer(offset=4)
        original_tokens = list(kv_layer.state["payload"].tokens)

        ok = scheduler._rewind_prompt_cache_for_generation([kv_layer, failing_layer])

        assert ok is False
        assert kv_layer.can_rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert kv_layer.rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert kv_layer.offset == 4
        assert kv_layer.state["payload"].tokens == original_tokens
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_preserves_state_aliasing(
        self, mock_model, mock_tokenizer
    ):
        """Rollback should preserve shared-reference aliasing inside state payloads."""

        class AliasedStateKV:
            def __init__(self, offset=4):
                self.offset = offset
                shared_payload = {"tokens": [451, 452, 453, 454]}
                self._state = {"left": shared_payload, "right": shared_payload}
                self.can_rewind_calls = []
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self._state = v

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                # Mutate through one alias branch.
                self._state["left"]["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        kv_layer = AliasedStateKV(offset=4)
        failing_layer = FailingLayer(offset=4)

        ok = scheduler._rewind_prompt_cache_for_generation([kv_layer, failing_layer])

        assert ok is False
        assert kv_layer.can_rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert kv_layer.rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert kv_layer.offset == 4
        assert kv_layer.state["left"]["tokens"] == [451, 452, 453, 454]
        assert kv_layer.state["left"] is kv_layer.state["right"]
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_preserves_keys_values_aliasing(
        self, mock_model, mock_tokenizer
    ):
        """Rollback should preserve aliasing across keys/values fallback payloads."""

        class AliasedKVPayloadLayer:
            def __init__(self, offset=4):
                self.offset = offset
                shared_payload = [461, 462, 463, 464]
                self.keys = shared_payload
                self.values = shared_payload
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                # Mutate through one aliased attribute.
                self.keys.pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        kv_layer = AliasedKVPayloadLayer(offset=4)
        failing_layer = FailingLayer(offset=4)

        ok = scheduler._rewind_prompt_cache_for_generation([kv_layer, failing_layer])

        assert ok is False
        assert kv_layer.can_rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert kv_layer.rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert kv_layer.offset == 4
        assert kv_layer.keys == [461, 462, 463, 464]
        assert kv_layer.keys is kv_layer.values
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_restores_keys_values_when_state_restore_is_partial(
        self, mock_model, mock_tokenizer
    ):
        """Rollback should restore keys/values even when state setter succeeds partially."""

        class PartialStateRestoreLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self._state = {"tokens": [1, 2, 3]}
                self.keys = {"tokens": [10, 20, 30]}
                self.values = {"tokens": [100, 200, 300]}
                self.can_rewind_calls = []
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                # Intentionally partial restore contract: only state is updated.
                self._state = v

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self.keys["tokens"].pop()
                self.values["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        kv_layer = PartialStateRestoreLayer(offset=4)
        failing_layer = FailingLayer(offset=4)
        original_keys_obj = kv_layer.keys
        original_values_obj = kv_layer.values
        assert original_keys_obj is not original_values_obj

        ok = scheduler._rewind_prompt_cache_for_generation([kv_layer, failing_layer])

        assert ok is False
        assert kv_layer.can_rewind_calls
        assert all(call == 1 for call in kv_layer.can_rewind_calls)
        assert failing_layer.can_rewind_calls
        assert all(call == 1 for call in failing_layer.can_rewind_calls)
        assert kv_layer.rewind_calls
        assert all(call == 1 for call in kv_layer.rewind_calls)
        assert failing_layer.rewind_calls
        assert all(call == 1 for call in failing_layer.rewind_calls)
        assert kv_layer.offset == 4
        assert kv_layer.state == {"tokens": [1, 2, 3]}
        assert kv_layer.keys == {"tokens": [10, 20, 30]}
        assert kv_layer.values == {"tokens": [100, 200, 300]}
        assert kv_layer.keys is not kv_layer.values
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_state_setter_failure_does_not_taint_fallback_payload_restore(
        self, mock_model, mock_tokenizer
    ):
        """Fallback keys/values restore should not reuse state-setter-mutated clones."""

        class SetterMutatesThenRaisesLayer:
            def __init__(self, offset=3):
                self.offset = offset
                shared_payload = {"tokens": [1, 2, 3]}
                self._state = shared_payload
                self.keys = shared_payload
                self.values = shared_payload
                self.can_rewind_calls = []
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                # Simulate non-atomic setter behavior: mutate incoming payload
                # then fail, forcing fallback keys/values restoration.
                if isinstance(v, dict) and isinstance(v.get("tokens"), list):
                    v["tokens"].pop()
                raise RuntimeError("state setter failed after mutating input")

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self.keys["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=3):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        kv_layer = SetterMutatesThenRaisesLayer(offset=3)
        failing_layer = FailingLayer(offset=3)

        ok = scheduler._rewind_prompt_cache_for_generation([kv_layer, failing_layer])

        assert ok is False
        assert kv_layer.can_rewind_calls == [1]
        assert kv_layer.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert kv_layer.offset == 3
        assert kv_layer.keys["tokens"] == [1, 2, 3]
        assert kv_layer.values["tokens"] == [1, 2, 3]
        assert kv_layer.keys is kv_layer.values
        assert failing_layer.offset == 3

    def test_rewind_prompt_cache_meta_state_setter_failure_does_not_taint_aliased_payload_restore(
        self, mock_model, mock_tokenizer
    ):
        """A mutating meta_state setter failure must not leave aliased payload attrs truncated."""

        class MetaStateSetterMutatesThenRaisesLayer:
            def __init__(self, offset=3):
                self.offset = offset
                shared_payload = {"tokens": [1, 2, 3]}
                self._meta_state = shared_payload
                self.keys = shared_payload
                self.can_rewind_calls = []
                self.rewind_calls = []
                self.last_attempted_meta_state = None

            @property
            def meta_state(self):
                return self._meta_state

            @meta_state.setter
            def meta_state(self, v):
                self.last_attempted_meta_state = v
                if isinstance(v, dict) and isinstance(v.get("tokens"), list):
                    v["tokens"].pop()
                raise RuntimeError("meta_state setter failed after mutating input")

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._meta_state["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=3):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        kv_layer = MetaStateSetterMutatesThenRaisesLayer(offset=3)
        failing_layer = FailingLayer(offset=3)

        assert kv_layer.meta_state is kv_layer.keys

        ok = scheduler._rewind_prompt_cache_for_generation([kv_layer, failing_layer])

        assert ok is False
        assert kv_layer.can_rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert kv_layer.rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert kv_layer.offset == 3
        assert kv_layer.last_attempted_meta_state is not None
        assert kv_layer.last_attempted_meta_state["tokens"] == [1, 2]
        assert kv_layer.keys["tokens"] == [1, 2, 3]
        assert kv_layer.last_attempted_meta_state is not kv_layer.keys
        assert failing_layer.offset == 3

    def test_rewind_prompt_cache_mutation_failure_preserves_nested_alias_from_deepcopy_branch(
        self, mock_model, mock_tokenizer
    ):
        """Rollback should preserve nested aliasing when cloning custom objects."""

        class ChildHolder:
            def __init__(self):
                self.child = {"tokens": [551, 552, 553, 554]}

        class NestedAliasStateKV:
            def __init__(self, offset=4):
                self.offset = offset
                holder = ChildHolder()
                self._state = {"a": holder, "b": holder.child}
                self.can_rewind_calls = []
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self._state = v

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["a"].child["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        kv_layer = NestedAliasStateKV(offset=4)
        failing_layer = FailingLayer(offset=4)

        assert kv_layer.state["a"].child is kv_layer.state["b"]

        ok = scheduler._rewind_prompt_cache_for_generation([kv_layer, failing_layer])

        assert ok is False
        assert kv_layer.can_rewind_calls == [1]
        assert kv_layer.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert kv_layer.offset == 4
        assert kv_layer.state["a"].child["tokens"] == [551, 552, 553, 554]
        assert kv_layer.state["b"]["tokens"] == [551, 552, 553, 554]
        assert kv_layer.state["a"].child is kv_layer.state["b"]
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_preserves_aliasing_across_cachelist_siblings(
        self, mock_model, mock_tokenizer
    ):
        """Rollback should preserve shared state across sibling CacheList subcaches."""

        class SharedStateLayer:
            def __init__(self, shared_state, offset=4):
                self.offset = offset
                self._state = shared_state
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self._state = v

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["tokens"].pop()
                return True

        class CacheList:
            def __init__(self, *caches):
                self.caches = tuple(caches)

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        shared_state = {"tokens": [601, 602, 603, 604]}
        left = SharedStateLayer(shared_state, offset=4)
        right = SharedStateLayer(shared_state, offset=4)
        cache_tree = CacheList(left, right)
        failing_layer = FailingLayer(offset=4)

        assert left.state is right.state

        ok = scheduler._rewind_prompt_cache_for_generation([cache_tree, failing_layer])

        assert ok is False
        assert left.offset == 4
        assert right.offset == 4
        assert left.state["tokens"] == [601, 602, 603, 604]
        assert left.state is right.state
        assert left.rewind_calls == [1]
        assert right.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_state_setter_does_not_mutate_restored_sibling(
        self, mock_model, mock_tokenizer
    ):
        """Direct restore helper should inherit shared ancestry, then isolate after setter failure."""

        class MarkerStableStateLayer:
            def __init__(self, offset=2):
                self.offset = offset
                self._state = {"tokens": [901]}

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                v["seen_by_left"] = True
                self._state = v

        class MutatingFailingRestoreLayer:
            def __init__(self, offset=2):
                self.offset = offset
                self._state = {"tokens": [902]}
                self.last_attempted_state = None

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self.last_attempted_state = v
                v["tokens"].pop()
                raise RuntimeError("partial restore failure")

        class CacheList:
            def __init__(self, *caches):
                self.caches = tuple(caches)

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        shared_snapshot_state = {"tokens": [611, 612, 613, 614]}
        left = MarkerStableStateLayer(offset=4)
        right = MutatingFailingRestoreLayer(offset=4)
        cache_tree = CacheList(left, right)
        snapshot = (
            "tree",
            [
                ("leaf", {"state": shared_snapshot_state, "offset": 4}),
                ("leaf", {"state": shared_snapshot_state, "offset": 4}),
            ],
        )

        scheduler._restore_cache_tree_rewind_metadata(cache_tree, snapshot)

        assert left.offset == 4
        assert right.offset == 4
        assert right.last_attempted_state is not None
        assert left.state["tokens"] == [611, 612, 613, 614]
        assert left.state["seen_by_left"] is True
        assert right.last_attempted_state["tokens"] == [611, 612, 613]
        assert right.last_attempted_state["seen_by_left"] is True
        assert right.last_attempted_state is not left.state

    def test_rewind_prompt_cache_mutation_failure_cachelist_sibling_state_setter_preserves_snapshot_aliasing(
        self, mock_model, mock_tokenizer
    ):
        """Full snapshot+restore path should preserve sibling shared ancestry inside CacheList."""

        class MarkerStableStateLayer:
            def __init__(self, shared_state, offset=4):
                self.offset = offset
                self._state = shared_state
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                v["seen_by_left"] = True
                self._state = v

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["tokens"].pop()
                return True

        class MutatingFailingRestoreLayer:
            def __init__(self, shared_state, offset=4):
                self.offset = offset
                self._state = shared_state
                self.last_attempted_state = None
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self.last_attempted_state = v
                v["tokens"].pop()
                raise RuntimeError("partial restore failure")

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["tokens"].pop()
                return True

        class CacheList:
            def __init__(self, *caches):
                self.caches = tuple(caches)

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        shared_state = {"tokens": [612, 613, 614, 615]}
        left = MarkerStableStateLayer(shared_state, offset=4)
        right = MutatingFailingRestoreLayer(shared_state, offset=4)
        cache_tree = CacheList(left, right)
        failing_layer = FailingLayer(offset=4)

        ok = scheduler._rewind_prompt_cache_for_generation([cache_tree, failing_layer])

        assert ok is False
        assert left.offset == 4
        assert right.offset == 4
        assert right.last_attempted_state is not None
        assert left.state["tokens"] == [612, 613, 614, 615]
        assert left.state["seen_by_left"] is True
        assert right.last_attempted_state["tokens"] == [612, 613, 614]
        assert right.last_attempted_state["seen_by_left"] is True
        assert right.last_attempted_state is not left.state
        assert left.rewind_calls == [1]
        assert right.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_preserves_aliasing_across_top_level_entries(
        self, mock_model, mock_tokenizer
    ):
        """Rollback should preserve aliasing that spans separate top-level cache entries."""

        class SharedStateLayer:
            def __init__(self, shared_state, offset=4):
                self.offset = offset
                self._state = shared_state
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self._state = v

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        shared_state = {"tokens": [621, 622, 623, 624]}
        left = SharedStateLayer(shared_state, offset=4)
        right = SharedStateLayer(shared_state, offset=4)
        failing_layer = FailingLayer(offset=4)

        assert left.state is right.state

        ok = scheduler._rewind_prompt_cache_for_generation([left, right, failing_layer])

        assert ok is False
        assert left.offset == 4
        assert right.offset == 4
        assert left.state["tokens"] == [621, 622, 623, 624]
        assert left.state is right.state
        assert left.rewind_calls == [1]
        assert right.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_top_level_state_setter_failure_isolates_restored_owner(
        self, mock_model, mock_tokenizer
    ):
        """Later top-level state-setter failure must not corrupt an already-restored owner."""

        class MarkerStableStateLayer:
            def __init__(self, shared_state, offset=4):
                self.offset = offset
                self._state = shared_state
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                v["seen_by_left"] = True
                self._state = v

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["tokens"].pop()
                return True

        class MutatingFailingRestoreLayer:
            def __init__(self, shared_state, offset=4):
                self.offset = offset
                self._state = shared_state
                self.last_attempted_state = None
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self.last_attempted_state = v
                v["tokens"].pop()
                raise RuntimeError("partial restore failure")

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        shared_state = {"tokens": [626, 627, 628, 629]}
        left = MarkerStableStateLayer(shared_state, offset=4)
        right = MutatingFailingRestoreLayer(shared_state, offset=4)
        failing_layer = FailingLayer(offset=4)

        ok = scheduler._rewind_prompt_cache_for_generation([left, right, failing_layer])

        assert ok is False
        assert left.offset == 4
        assert right.offset == 4
        assert right.last_attempted_state is not None
        assert left.state["tokens"] == [626, 627, 628, 629]
        assert left.state["seen_by_left"] is True
        assert right.last_attempted_state["tokens"] == [626, 627, 628]
        assert right.last_attempted_state["seen_by_left"] is True
        assert right.last_attempted_state is not left.state
        assert left.rewind_calls == [1]
        assert right.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_preserves_aliasing_from_cachelist_child_to_top_level_entry(
        self, mock_model, mock_tokenizer
    ):
        """Shared memo should preserve aliasing across both CacheList recursion and top-level iteration."""

        class SharedStateLayer:
            def __init__(self, shared_state, offset=4):
                self.offset = offset
                self._state = shared_state
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self._state = v

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["tokens"].pop()
                return True

        class CacheList:
            def __init__(self, *caches):
                self.caches = tuple(caches)

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        shared_state = {"tokens": [622, 623, 624, 625]}
        nested_shared = SharedStateLayer(shared_state, offset=4)
        nested_independent = SharedStateLayer({"tokens": [700, 701, 702, 703]}, offset=4)
        cache_tree = CacheList(nested_shared, nested_independent)
        top_level_shared = SharedStateLayer(shared_state, offset=4)
        failing_layer = FailingLayer(offset=4)

        assert nested_shared.state is top_level_shared.state

        ok = scheduler._rewind_prompt_cache_for_generation(
            [cache_tree, top_level_shared, failing_layer]
        )

        assert ok is False
        assert nested_shared.offset == 4
        assert nested_independent.offset == 4
        assert top_level_shared.offset == 4
        assert nested_shared.state["tokens"] == [622, 623, 624, 625]
        assert nested_independent.state["tokens"] == [700, 701, 702, 703]
        assert top_level_shared.state["tokens"] == [622, 623, 624, 625]
        assert nested_shared.state is top_level_shared.state
        assert nested_independent.state is not nested_shared.state
        assert nested_shared.rewind_calls == [1]
        assert nested_independent.rewind_calls == [1]
        assert top_level_shared.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_cachelist_boundary_state_setter_failure_repairs_prior_owners(
        self, mock_model, mock_tokenizer
    ):
        """State-owner repair must survive both CacheList recursion and later top-level setter failure."""

        class MarkerStableStateLayer:
            def __init__(self, shared_state, marker_name, offset=4):
                self.offset = offset
                self._state = shared_state
                self.marker_name = marker_name
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                v[self.marker_name] = True
                self._state = v

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["tokens"].pop()
                return True

        class StableIndependentLayer:
            def __init__(self, tokens, offset=4):
                self.offset = offset
                self._state = {"tokens": list(tokens)}
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self._state = v

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["tokens"].pop()
                return True

        class MutatingFailingRestoreLayer:
            def __init__(self, shared_state, offset=4):
                self.offset = offset
                self._state = shared_state
                self.last_attempted_state = None
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self.last_attempted_state = v
                v["tokens"].pop()
                raise RuntimeError("partial restore failure")

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["tokens"].pop()
                return True

        class CacheList:
            def __init__(self, *caches):
                self.caches = tuple(caches)

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        shared_state = {"tokens": [641, 642, 643, 644]}
        nested_owner = MarkerStableStateLayer(
            shared_state,
            marker_name="seen_by_nested_owner",
            offset=4,
        )
        nested_independent = StableIndependentLayer([710, 711, 712, 713], offset=4)
        cache_tree = CacheList(nested_owner, nested_independent)
        top_level_owner = MarkerStableStateLayer(
            shared_state,
            marker_name="seen_by_top_level_owner",
            offset=4,
        )
        failing_owner = MutatingFailingRestoreLayer(shared_state, offset=4)
        failing_layer = FailingLayer(offset=4)

        ok = scheduler._rewind_prompt_cache_for_generation(
            [cache_tree, top_level_owner, failing_owner, failing_layer]
        )

        assert ok is False
        assert nested_owner.offset == 4
        assert nested_independent.offset == 4
        assert top_level_owner.offset == 4
        assert failing_owner.offset == 4
        assert nested_owner.state["tokens"] == [641, 642, 643, 644]
        assert top_level_owner.state["tokens"] == [641, 642, 643, 644]
        assert nested_independent.state["tokens"] == [710, 711, 712, 713]
        assert nested_owner.state["seen_by_nested_owner"] is True
        assert nested_owner.state["seen_by_top_level_owner"] is True
        assert top_level_owner.state["seen_by_nested_owner"] is True
        assert top_level_owner.state["seen_by_top_level_owner"] is True
        assert nested_owner.state is top_level_owner.state
        assert nested_independent.state is not nested_owner.state
        assert failing_owner.last_attempted_state["tokens"] == [641, 642, 643]
        assert failing_owner.last_attempted_state["seen_by_nested_owner"] is True
        assert failing_owner.last_attempted_state["seen_by_top_level_owner"] is True
        assert failing_owner.last_attempted_state is not nested_owner.state
        assert nested_owner.rewind_calls == [1]
        assert nested_independent.rewind_calls == [1]
        assert top_level_owner.rewind_calls == [1]
        assert failing_owner.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_top_level_state_setter_failure_repairs_all_prior_owners(
        self, mock_model, mock_tokenizer
    ):
        """Later state-setter failure must repair every previously restored owner, not just the first."""

        class MarkerStableStateLayer:
            def __init__(self, shared_state, marker_name, offset=4):
                self.offset = offset
                self._state = shared_state
                self.marker_name = marker_name
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                v[self.marker_name] = True
                self._state = v

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["tokens"].pop()
                return True

        class MutatingFailingRestoreLayer:
            def __init__(self, shared_state, offset=4):
                self.offset = offset
                self._state = shared_state
                self.last_attempted_state = None
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self.last_attempted_state = v
                v["tokens"].pop()
                raise RuntimeError("partial restore failure")

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        shared_state = {"tokens": [627, 628, 629, 630]}
        first_owner = MarkerStableStateLayer(
            shared_state,
            marker_name="seen_by_first_owner",
            offset=4,
        )
        second_owner = MarkerStableStateLayer(
            shared_state,
            marker_name="seen_by_second_owner",
            offset=4,
        )
        failing_owner = MutatingFailingRestoreLayer(shared_state, offset=4)
        failing_layer = FailingLayer(offset=4)

        ok = scheduler._rewind_prompt_cache_for_generation(
            [first_owner, second_owner, failing_owner, failing_layer]
        )

        assert ok is False
        assert first_owner.offset == 4
        assert second_owner.offset == 4
        assert failing_owner.offset == 4
        assert failing_owner.last_attempted_state is not None
        assert first_owner.state["tokens"] == [627, 628, 629, 630]
        assert second_owner.state["tokens"] == [627, 628, 629, 630]
        assert first_owner.state["seen_by_first_owner"] is True
        assert first_owner.state["seen_by_second_owner"] is True
        assert second_owner.state["seen_by_first_owner"] is True
        assert second_owner.state["seen_by_second_owner"] is True
        assert first_owner.state is second_owner.state
        assert failing_owner.last_attempted_state["tokens"] == [627, 628, 629]
        assert failing_owner.last_attempted_state["seen_by_first_owner"] is True
        assert failing_owner.last_attempted_state["seen_by_second_owner"] is True
        assert failing_owner.last_attempted_state is not first_owner.state
        assert first_owner.rewind_calls == [1]
        assert second_owner.rewind_calls == [1]
        assert failing_owner.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_preserves_meta_state_aliasing_across_top_level_entries(
        self, mock_model, mock_tokenizer
    ):
        """Rollback should preserve shared meta_state across top-level cache entries."""

        class SharedMetaStateLayer:
            def __init__(self, shared_meta_state, offset=4):
                self.offset = offset
                self.meta_state = shared_meta_state
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self.meta_state["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        shared_meta_state = {"tokens": [631, 632, 633, 634]}
        left = SharedMetaStateLayer(shared_meta_state, offset=4)
        right = SharedMetaStateLayer(shared_meta_state, offset=4)
        failing_layer = FailingLayer(offset=4)

        assert left.meta_state is right.meta_state

        ok = scheduler._rewind_prompt_cache_for_generation([left, right, failing_layer])

        assert ok is False
        assert left.offset == 4
        assert right.offset == 4
        assert left.meta_state["tokens"] == [631, 632, 633, 634]
        assert left.meta_state is right.meta_state
        assert left.rewind_calls == [1]
        assert right.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_top_level_meta_state_setter_failure_repairs_all_prior_owners(
        self, mock_model, mock_tokenizer
    ):
        """Later meta_state-setter failure must repair every previously restored owner."""

        class MarkerStableMetaStateLayer:
            def __init__(self, shared_meta_state, marker_name, offset=4):
                self.offset = offset
                self._meta_state = shared_meta_state
                self.marker_name = marker_name
                self.rewind_calls = []

            @property
            def meta_state(self):
                return self._meta_state

            @meta_state.setter
            def meta_state(self, v):
                v[self.marker_name] = True
                self._meta_state = v

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._meta_state["tokens"].pop()
                return True

        class MutatingFailingMetaStateRestoreLayer:
            def __init__(self, shared_meta_state, offset=4):
                self.offset = offset
                self._meta_state = shared_meta_state
                self.last_attempted_meta_state = None
                self.rewind_calls = []

            @property
            def meta_state(self):
                return self._meta_state

            @meta_state.setter
            def meta_state(self, v):
                self.last_attempted_meta_state = v
                v["tokens"].pop()
                raise RuntimeError("partial meta_state restore failure")

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._meta_state["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        shared_meta_state = {"tokens": [633, 634, 635, 636]}
        first_owner = MarkerStableMetaStateLayer(
            shared_meta_state,
            marker_name="seen_by_first_owner",
            offset=4,
        )
        second_owner = MarkerStableMetaStateLayer(
            shared_meta_state,
            marker_name="seen_by_second_owner",
            offset=4,
        )
        failing_owner = MutatingFailingMetaStateRestoreLayer(shared_meta_state, offset=4)
        failing_layer = FailingLayer(offset=4)

        ok = scheduler._rewind_prompt_cache_for_generation(
            [first_owner, second_owner, failing_owner, failing_layer]
        )

        assert ok is False
        assert first_owner.offset == 4
        assert second_owner.offset == 4
        assert failing_owner.offset == 4
        assert failing_owner.last_attempted_meta_state is not None
        assert first_owner.meta_state["tokens"] == [633, 634, 635, 636]
        assert second_owner.meta_state["tokens"] == [633, 634, 635, 636]
        assert first_owner.meta_state["seen_by_first_owner"] is True
        assert first_owner.meta_state["seen_by_second_owner"] is True
        assert second_owner.meta_state["seen_by_first_owner"] is True
        assert second_owner.meta_state["seen_by_second_owner"] is True
        assert first_owner.meta_state is second_owner.meta_state
        assert failing_owner.last_attempted_meta_state["tokens"] == [633, 634, 635]
        assert failing_owner.last_attempted_meta_state["seen_by_first_owner"] is True
        assert failing_owner.last_attempted_meta_state["seen_by_second_owner"] is True
        assert failing_owner.last_attempted_meta_state is not first_owner.meta_state
        assert first_owner.rewind_calls == [1]
        assert second_owner.rewind_calls == [1]
        assert failing_owner.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    @pytest.mark.parametrize(
        "left_attr,right_attr",
        [
            ("state", "meta_state"),
            ("state", "keys"),
            ("keys", "cache"),
        ],
    )
    def test_rewind_prompt_cache_mutation_failure_preserves_cross_attribute_aliasing_across_top_level_entries(
        self, mock_model, mock_tokenizer, left_attr, right_attr
    ):
        """Rollback should preserve aliasing that spans different payload attribute families."""

        class SharedCrossAttrLayer:
            def __init__(self, shared_payload, offset=4):
                self.offset = offset
                setattr(self, left_attr, shared_payload)
                setattr(self, right_attr, shared_payload)
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                getattr(self, left_attr)["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        shared_payload = {"tokens": [635, 636, 637, 638]}
        left = SharedCrossAttrLayer(shared_payload, offset=4)
        right = SharedCrossAttrLayer(shared_payload, offset=4)
        failing_layer = FailingLayer(offset=4)

        assert getattr(left, left_attr) is getattr(left, right_attr)
        assert getattr(left, left_attr) is getattr(right, left_attr)
        assert getattr(left, left_attr) is getattr(right, right_attr)

        ok = scheduler._rewind_prompt_cache_for_generation([left, right, failing_layer])

        assert ok is False
        restored_left_primary = getattr(left, left_attr)
        restored_left_secondary = getattr(left, right_attr)
        restored_right_primary = getattr(right, left_attr)
        restored_right_secondary = getattr(right, right_attr)
        assert left.offset == 4
        assert right.offset == 4
        assert restored_left_primary["tokens"] == [635, 636, 637, 638]
        assert restored_left_primary is restored_left_secondary
        assert restored_left_primary is restored_right_primary
        assert restored_left_primary is restored_right_secondary
        assert left.rewind_calls == [1]
        assert right.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_preserves_keys_aliasing_across_top_level_entries(
        self, mock_model, mock_tokenizer
    ):
        """Rollback should preserve shared payload attrs across top-level cache entries."""

        class SharedKeysLayer:
            def __init__(self, shared_keys, offset=4):
                self.offset = offset
                self.keys = shared_keys
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self.keys["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        shared_keys = {"tokens": [641, 642, 643, 644]}
        left = SharedKeysLayer(shared_keys, offset=4)
        right = SharedKeysLayer(shared_keys, offset=4)
        failing_layer = FailingLayer(offset=4)

        assert left.keys is right.keys

        ok = scheduler._rewind_prompt_cache_for_generation([left, right, failing_layer])

        assert ok is False
        assert left.offset == 4
        assert right.offset == 4
        assert left.keys["tokens"] == [641, 642, 643, 644]
        assert left.keys is right.keys
        assert left.rewind_calls == [1]
        assert right.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    @pytest.mark.parametrize("payload_attr", ["values", "cache"])
    def test_rewind_prompt_cache_mutation_failure_preserves_other_payload_attr_aliasing_across_top_level_entries(
        self, mock_model, mock_tokenizer, payload_attr
    ):
        """Rollback should preserve shared values/cache attrs across top-level cache entries."""

        class SharedPayloadAttrLayer:
            def __init__(self, shared_payload, offset=4):
                self.offset = offset
                setattr(self, payload_attr, shared_payload)
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                getattr(self, payload_attr)["tokens"].pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        shared_payload = {"tokens": [651, 652, 653, 654]}
        left = SharedPayloadAttrLayer(shared_payload, offset=4)
        right = SharedPayloadAttrLayer(shared_payload, offset=4)
        failing_layer = FailingLayer(offset=4)

        assert getattr(left, payload_attr) is getattr(right, payload_attr)

        ok = scheduler._rewind_prompt_cache_for_generation([left, right, failing_layer])

        assert ok is False
        assert left.offset == 4
        assert right.offset == 4
        assert getattr(left, payload_attr)["tokens"] == [651, 652, 653, 654]
        assert getattr(left, payload_attr) is getattr(right, payload_attr)
        assert left.rewind_calls == [1]
        assert right.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    def test_snapshot_cache_tree_rewind_metadata_cyclic_state_marks_invalid(
        self, mock_model, mock_tokenizer
    ):
        """Cyclic state payloads must fail-closed without over-broad invalidation."""

        class CyclicStateCache:
            def __init__(self):
                self.offset = 4
                state = {"tokens": [501, 502, 503, 504]}
                state["self"] = state
                self.state = state

            def can_rewind(self, n):
                return n <= self.offset

        class NonCyclicStateCache:
            def __init__(self):
                self.offset = 4
                self.state = {"tokens": [601, 602, 603, 604], "meta": {"count": 4}}

            def can_rewind(self, n):
                return n <= self.offset

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        cyclic_snapshot = scheduler._snapshot_cache_tree_rewind_metadata(CyclicStateCache())
        non_cyclic_snapshot = scheduler._snapshot_cache_tree_rewind_metadata(
            NonCyclicStateCache()
        )

        assert scheduler._is_invalid_rewind_snapshot(cyclic_snapshot) is True
        assert scheduler._is_invalid_rewind_snapshot(non_cyclic_snapshot) is False

    def test_rewind_prompt_cache_cyclic_state_fails_closed_before_mutation(
        self, mock_model, mock_tokenizer
    ):
        """Cyclic state should fail closed before any rewind mutation is attempted."""

        class CyclicPayload:
            def __init__(self, tokens):
                self.tokens = list(tokens)
                self.owner = None

        class MutableStateKV:
            def __init__(self, offset=4):
                self.offset = offset
                payload = CyclicPayload([511, 512, 513, 514])
                self._state = {"payload": payload}
                payload.owner = self._state
                self.can_rewind_calls = []
                self.rewind_calls = []

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, v):
                self._state = v

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self._state["payload"].tokens.pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        kv_layer = MutableStateKV(offset=4)
        failing_layer = FailingLayer(offset=4)
        original_state_obj = kv_layer.state
        original_payload_obj = kv_layer.state["payload"]
        original_tokens = list(kv_layer.state["payload"].tokens)

        ok = scheduler._rewind_prompt_cache_for_generation([kv_layer, failing_layer])

        assert ok is False
        assert kv_layer.can_rewind_calls
        assert all(call == 1 for call in kv_layer.can_rewind_calls)
        assert failing_layer.can_rewind_calls
        assert all(call == 1 for call in failing_layer.can_rewind_calls)
        assert kv_layer.rewind_calls == []
        assert failing_layer.rewind_calls == []
        assert kv_layer.offset == 4
        assert kv_layer.state["payload"].tokens == original_tokens
        assert kv_layer.state is original_state_obj
        assert kv_layer.state["payload"] is original_payload_obj
        assert kv_layer.state["payload"].owner is kv_layer.state
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_cyclic_state_cache_list_fails_closed_before_subcache_mutation(
        self, mock_model, mock_tokenizer
    ):
        """Cyclic subcache in CacheList should block rewind on every subcache."""

        class CacheList:
            def __init__(self, caches):
                self.caches = list(caches)

        class CyclicLayer:
            def __init__(self, offset=4):
                self.offset = offset
                state = {"tokens": [701, 702, 703, 704]}
                state["self"] = state
                self.state = state
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self.state["tokens"].pop()
                return True

        class NormalLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.state = {"tokens": [801, 802, 803, 804]}
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self.state["tokens"].pop()
                return True

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        cyclic_layer = CyclicLayer(offset=4)
        normal_layer = NormalLayer(offset=4)
        original_cyclic_tokens = list(cyclic_layer.state["tokens"])
        original_normal_tokens = list(normal_layer.state["tokens"])
        cache_tree = CacheList([cyclic_layer, normal_layer])

        ok = scheduler._rewind_prompt_cache_for_generation([cache_tree])

        assert ok is False
        assert cyclic_layer.can_rewind_calls
        assert all(call == 1 for call in cyclic_layer.can_rewind_calls)
        assert normal_layer.can_rewind_calls
        assert all(call == 1 for call in normal_layer.can_rewind_calls)
        assert cyclic_layer.rewind_calls == []
        assert normal_layer.rewind_calls == []
        assert cyclic_layer.offset == 4
        assert cyclic_layer.state["tokens"] == original_cyclic_tokens
        assert normal_layer.offset == 4
        assert normal_layer.state["tokens"] == original_normal_tokens

    def test_snapshot_cache_tree_rewind_metadata_custom_meta_state_fallback_clone_stays_valid(
        self, mock_model, mock_tokenizer
    ):
        """Custom meta_state payloads that reject deepcopy should still snapshot via __dict__ fallback."""

        class FallbackCloneMeta:
            def __init__(self, tokens):
                self.tokens = list(tokens)

            def __deepcopy__(self, memo):
                raise TypeError("meta payload does not support deepcopy")

        class FallbackCloneMetaStateCache:
            def __init__(self):
                self.offset = 4
                self.state = {"tokens": [901, 902, 903, 904]}
                self.meta_state = {"payload": FallbackCloneMeta([1, 2, 3])}

            def can_rewind(self, n):
                return n <= self.offset

        class CloneableMetaStateCache:
            def __init__(self):
                self.offset = 4
                self.state = {"tokens": [911, 912, 913, 914]}
                self.meta_state = {"payload": {"tokens": [1, 2, 3]}}

            def can_rewind(self, n):
                return n <= self.offset

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        fallback_cache = FallbackCloneMetaStateCache()
        fallback_snapshot = scheduler._snapshot_cache_tree_rewind_metadata(
            fallback_cache
        )
        cloneable_snapshot = scheduler._snapshot_cache_tree_rewind_metadata(
            CloneableMetaStateCache()
        )

        assert scheduler._is_invalid_rewind_snapshot(fallback_snapshot) is False
        assert scheduler._is_invalid_rewind_snapshot(cloneable_snapshot) is False
        assert fallback_snapshot[0] == "leaf"
        fallback_payload = fallback_snapshot[1]["meta_state"]["payload"]
        assert isinstance(fallback_payload, FallbackCloneMeta)
        assert fallback_payload is not fallback_cache.meta_state["payload"]
        assert fallback_payload.tokens is not fallback_cache.meta_state["payload"].tokens
        fallback_cache.meta_state["payload"].tokens.append(99)
        assert fallback_payload.tokens == [1, 2, 3]

    def test_snapshot_cache_tree_rewind_metadata_truly_uncloneable_meta_state_marks_invalid(
        self, mock_model, mock_tokenizer
    ):
        """Unsupported custom meta_state payloads must still fail closed at snapshot time."""
        import threading

        class TrulyUncloneableMeta:
            __slots__ = ("lock",)

            def __init__(self):
                self.lock = threading.Lock()

            def __deepcopy__(self, memo):
                raise TypeError("meta payload does not support deepcopy")

        class UncloneableMetaStateCache:
            def __init__(self):
                self.offset = 4
                self.state = {"tokens": [921, 922, 923, 924]}
                self.meta_state = {"payload": TrulyUncloneableMeta()}

            def can_rewind(self, n):
                return n <= self.offset

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        uncloneable_snapshot = scheduler._snapshot_cache_tree_rewind_metadata(
            UncloneableMetaStateCache()
        )

        assert scheduler._is_invalid_rewind_snapshot(uncloneable_snapshot) is True

    def test_rewind_prompt_cache_mutation_failure_restores_custom_meta_state_fallback_clone(
        self, mock_model, mock_tokenizer
    ):
        """Rollback should restore custom meta_state payloads through the fallback clone path."""

        class FallbackCloneMeta:
            def __init__(self, tokens):
                self.tokens = list(tokens)

            def __deepcopy__(self, memo):
                raise TypeError("meta payload does not support deepcopy")

        class CustomMetaStateLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.state = {"tokens": [941, 942, 943, 944]}
                self.meta_state = {"payload": FallbackCloneMeta([1, 2, 3, 4])}
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self.meta_state["payload"].tokens.pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        custom_layer = CustomMetaStateLayer(offset=4)
        original_meta_payload = custom_layer.meta_state["payload"]
        failing_layer = FailingLayer(offset=4)

        ok = scheduler._rewind_prompt_cache_for_generation([custom_layer, failing_layer])

        assert ok is False
        assert custom_layer.offset == 4
        assert custom_layer.meta_state["payload"].tokens == [1, 2, 3, 4]
        assert isinstance(custom_layer.meta_state["payload"], FallbackCloneMeta)
        assert custom_layer.meta_state["payload"] is not original_meta_payload
        assert custom_layer.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    def test_rewind_prompt_cache_mutation_failure_restores_slots_based_custom_meta_state(
        self, mock_model, mock_tokenizer
    ):
        """Rollback should restore custom meta_state payloads cloned through the __slots__ fallback path."""

        class SlotsCloneMeta:
            __slots__ = ("tokens",)

            def __init__(self, tokens):
                self.tokens = list(tokens)

            def __deepcopy__(self, memo):
                raise TypeError("meta payload does not support deepcopy")

        class SlotsMetaStateLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.state = {"tokens": [951, 952, 953, 954]}
                self.meta_state = {"payload": SlotsCloneMeta([1, 2, 3, 4])}
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                self.meta_state["payload"].tokens.pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        slots_layer = SlotsMetaStateLayer(offset=4)
        original_meta_payload = slots_layer.meta_state["payload"]
        failing_layer = FailingLayer(offset=4)

        ok = scheduler._rewind_prompt_cache_for_generation([slots_layer, failing_layer])

        assert ok is False
        assert slots_layer.offset == 4
        assert slots_layer.meta_state["payload"].tokens == [1, 2, 3, 4]
        assert isinstance(slots_layer.meta_state["payload"], SlotsCloneMeta)
        assert slots_layer.meta_state["payload"] is not original_meta_payload
        assert slots_layer.rewind_calls == [1]
        assert failing_layer.can_rewind_calls == [1]
        assert failing_layer.rewind_calls == [1]
        assert failing_layer.offset == 4

    def test_snapshot_cache_tree_rewind_metadata_cyclic_meta_state_marks_invalid(
        self, mock_model, mock_tokenizer
    ):
        """Cyclic meta_state payloads must still mark snapshots invalid."""

        class CyclicMetaStateCache:
            def __init__(self):
                self.offset = 4
                self.state = {"tokens": [921, 922, 923, 924]}
                meta_state = {"tokens": [1, 2, 3]}
                meta_state["self"] = meta_state
                self.meta_state = meta_state

            def can_rewind(self, n):
                return n <= self.offset

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        cyclic_snapshot = scheduler._snapshot_cache_tree_rewind_metadata(
            CyclicMetaStateCache()
        )

        assert scheduler._is_invalid_rewind_snapshot(cyclic_snapshot) is True

    @pytest.mark.parametrize("payload_attr", ["keys", "values", "cache"])
    def test_rewind_prompt_cache_cyclic_payload_attr_fails_closed_before_mutation(
        self, mock_model, mock_tokenizer, payload_attr
    ):
        """Cyclic keys/values/cache payloads should fail closed pre-mutation."""

        class CyclicPayloadLayer:
            def __init__(self, offset=4):
                self.offset = offset
                payload = [1001, 1002, 1003, 1004]
                payload.append(payload)
                setattr(self, payload_attr, payload)
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                payload = getattr(self, payload_attr)
                payload.pop()
                return True

        class FailingLayer:
            def __init__(self, offset=4):
                self.offset = offset
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                return False

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        cyclic_layer = CyclicPayloadLayer(offset=4)
        failing_layer = FailingLayer(offset=4)
        original_payload_obj = getattr(cyclic_layer, payload_attr)
        expected_prefix = [1001, 1002, 1003, 1004]

        ok = scheduler._rewind_prompt_cache_for_generation([cyclic_layer, failing_layer])

        assert ok is False
        assert cyclic_layer.can_rewind_calls
        assert all(call == 1 for call in cyclic_layer.can_rewind_calls)
        assert failing_layer.can_rewind_calls
        assert all(call == 1 for call in failing_layer.can_rewind_calls)
        assert cyclic_layer.rewind_calls == []
        assert failing_layer.rewind_calls == []
        assert cyclic_layer.offset == 4
        assert failing_layer.offset == 4
        assert getattr(cyclic_layer, payload_attr) is original_payload_obj
        assert original_payload_obj[:-1] == expected_prefix
        assert len(original_payload_obj) == 5
        assert original_payload_obj[-1] is original_payload_obj

    def test_add_request_exact_cache_hit_mixed_trim_only_legacy_saturated_rewinds_one_token(
        self, mock_model, mock_tokenizer
    ):
        """
        Fail-first reproducer: saturated mixed trim-only caches should rewind.

        Expected to fail pre-fix: current preflight requires is_trimmable() for
        trim-only caches and falls back even when trim(1) would succeed.
        """
        from omlx.cache.paged_cache import BlockTable

        class KVCache:
            def __init__(self, offset=8):
                self.offset = offset
                self.trim_calls = 0

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.trim_calls += 1
                n = min(self.offset, n)
                self.offset -= n
                return n

        class RotatingKVCache:
            def __init__(self, offset=8, max_size=4):
                self.offset = offset
                self.max_size = max_size
                self.trim_calls = 0

            def is_trimmable(self):
                # Legacy saturated-window semantics from trim-only path.
                return self.offset < self.max_size

            def trim(self, n):
                self.trim_calls += 1
                n = min(self.offset, n)
                self.offset -= n
                return n

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        block_table = BlockTable(
            request_id="req-mixed-legacy-sat",
            block_ids=[13, 14],
            num_tokens=8,
        )
        kv_layer = KVCache(offset=8)
        rotating_layer = RotatingKVCache(offset=8, max_size=4)
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [kv_layer, rotating_layer]

        request = Request(
            request_id="req-mixed-legacy-sat",
            prompt=[81, 82, 83, 84, 85, 86, 87, 88],
            sampling_params=SamplingParams(max_tokens=16),
        )

        scheduler.add_request(request)

        assert request.cached_tokens == 7
        assert request.remaining_tokens == [88]
        assert request.prompt_cache is not None
        assert request.block_table is not None
        assert request.shared_prefix_blocks == len(block_table.block_ids)
        assert kv_layer.trim_calls == 1
        assert rotating_layer.trim_calls == 1
        assert kv_layer.offset == 7
        assert rotating_layer.offset == 7
        scheduler.paged_cache_manager.delete_block_table.assert_not_called()


class TestSchedulerMixedSwaKickoffRuntimeSmoke:
    """Runtime smoke tests for mixed-SWA exact-hit kickoff after SSD restore."""

    class _DummyTokenizer:
        eos_token_id = 2
        name_or_path = ""

    class _DummyModel:
        def __init__(self, num_layers: int):
            self.layers = [object() for _ in range(num_layers)]
            self.args = SimpleNamespace(num_hidden_layers=num_layers)

    def _make_layer_cache(self, is_rotating: bool, prompt_len: int, window: int):
        """Build a real mlx-lm cache layer prefilled to prompt_len."""
        mx = pytest.importorskip("mlx.core")
        cache_mod = pytest.importorskip("mlx_lm.models.cache")

        layer_cache = (
            cache_mod.RotatingKVCache(max_size=window, keep=0)
            if is_rotating
            else cache_mod.KVCache()
        )
        keys = mx.ones((1, 1, prompt_len, 1), dtype=mx.float16)
        values = mx.ones((1, 1, prompt_len, 1), dtype=mx.float16)
        layer_cache.update_and_fetch(keys, values)
        return layer_cache

    def _run_exact_hit_disk_roundtrip_case(
        self,
        tmp_path: Path,
        profile_name: str,
        total_layers: int,
        rotating_layers: int,
        prompt_len: int,
        rotating_window: int,
    ):
        """Store mixed cache to SSD, restore, then run exact-hit kickoff."""
        from omlx.cache.paged_cache import PagedCacheManager
        from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
        from omlx.cache.prefix_cache import BlockAwarePrefixCache

        prompt_tokens = list(range(1000, 1000 + prompt_len))
        model = self._DummyModel(total_layers)

        config = SchedulerConfig(
            paged_ssd_cache_dir=str(
                tmp_path / f"{profile_name}-w{rotating_window}-p{prompt_len}"
            ),
            paged_cache_block_size=4,
            max_cache_blocks=256,
            initial_cache_blocks=128,
            model_name=profile_name,
        )
        scheduler = Scheduler(model=model, tokenizer=self._DummyTokenizer(), config=config)

        paged_cache_manager = PagedCacheManager(
            block_size=4,
            max_blocks=256,
            model_name=profile_name,
            initial_blocks=128,
        )
        ssd_cache_manager = PagedSSDCacheManager(
            cache_dir=Path(config.paged_ssd_cache_dir),
            max_size_bytes=256 * 1024 * 1024,
            hot_cache_max_bytes=0,
        )

        try:
            paged_cache_manager.set_paged_ssd_cache_manager(ssd_cache_manager)
            prefix_cache = BlockAwarePrefixCache(
                model=model,
                paged_cache_manager=paged_cache_manager,
                paged_ssd_cache_manager=ssd_cache_manager,
            )
            scheduler.block_aware_cache = prefix_cache
            scheduler.paged_cache_manager = paged_cache_manager

            kv_layers = total_layers - rotating_layers
            raw_cache = [
                *[
                    self._make_layer_cache(False, prompt_len, rotating_window)
                    for _ in range(kv_layers)
                ],
                *[
                    self._make_layer_cache(True, prompt_len, rotating_window)
                    for _ in range(rotating_layers)
                ],
            ]

            extracted, model_cache_config = scheduler._extract_cache_states(raw_cache)
            assert len(extracted) == total_layers

            block_table = prefix_cache.store_cache(
                request_id=f"seed-{profile_name}",
                tokens=prompt_tokens,
                cache_data=extracted,
                model_cache_config=model_cache_config,
            )
            assert block_table is not None
            assert block_table.num_tokens == prompt_len
            assert len(block_table.block_ids) == prompt_len // config.paged_cache_block_size

            # Force true disk-backed restore path.
            ssd_cache_manager.close()
            ssd_cache_manager = PagedSSDCacheManager(
                cache_dir=Path(config.paged_ssd_cache_dir),
                max_size_bytes=256 * 1024 * 1024,
                hot_cache_max_bytes=0,
            )
            paged_cache_manager.set_paged_ssd_cache_manager(ssd_cache_manager)
            prefix_cache.set_paged_ssd_cache_manager(ssd_cache_manager)

            request = Request(
                request_id=f"req-{profile_name}",
                prompt=prompt_tokens,
                sampling_params=SamplingParams(max_tokens=8),
            )
            scheduler.add_request(request)

            kv_offsets = []
            rotating_offsets = []
            if request.prompt_cache is not None:
                for layer_cache in request.prompt_cache:
                    class_name = type(layer_cache).__name__
                    if class_name in ("KVCache", "BatchKVCache"):
                        kv_offsets.append(layer_cache.offset)
                    elif class_name in ("RotatingKVCache", "BatchRotatingKVCache"):
                        rotating_offsets.append(layer_cache.offset)

            return {
                "prompt_tokens": prompt_tokens,
                "cached_tokens": request.cached_tokens,
                "remaining_tokens": request.remaining_tokens,
                "prompt_cache_present": request.prompt_cache is not None,
                "kv_offsets": kv_offsets,
                "rotating_offsets": rotating_offsets,
            }
        finally:
            try:
                ssd_cache_manager.close()
            except Exception:
                pass

    @pytest.mark.parametrize(
        ("profile_name", "total_layers", "rotating_layers"),
        [
            ("gpt_oss_like_50pct_swa", 8, 4),
            ("step3p5_like_75pct_swa", 8, 6),
        ],
    )
    def test_mixed_swa_exact_hit_disk_roundtrip_unsaturated_window_rewinds(
        self, tmp_path, profile_name, total_layers, rotating_layers
    ):
        """Control: unsaturated rotating windows should rewind kickoff by one token."""
        outcome = self._run_exact_hit_disk_roundtrip_case(
            tmp_path=tmp_path,
            profile_name=profile_name,
            total_layers=total_layers,
            rotating_layers=rotating_layers,
            prompt_len=8,
            rotating_window=16,
        )

        assert outcome["cached_tokens"] == 7
        assert outcome["remaining_tokens"] == [outcome["prompt_tokens"][-1]]
        assert outcome["prompt_cache_present"] is True
        assert outcome["kv_offsets"]
        assert outcome["rotating_offsets"]
        assert all(offset == 7 for offset in outcome["kv_offsets"])
        assert all(offset == 7 for offset in outcome["rotating_offsets"])

    @pytest.mark.parametrize(
        ("profile_name", "total_layers", "rotating_layers"),
        [
            ("gpt_oss_like_50pct_swa", 8, 4),
            ("step3p5_like_75pct_swa", 8, 6),
        ],
    )
    def test_mixed_swa_exact_hit_disk_roundtrip_saturated_window_rewinds(
        self, tmp_path, profile_name, total_layers, rotating_layers
    ):
        """
        Reproducer: saturated rotating windows should still rewind kickoff by one token.

        Expected to fail pre-fix: current behavior falls back fail-closed to full prefill.
        """
        outcome = self._run_exact_hit_disk_roundtrip_case(
            tmp_path=tmp_path,
            profile_name=profile_name,
            total_layers=total_layers,
            rotating_layers=rotating_layers,
            prompt_len=8,
            rotating_window=4,
        )

        assert outcome["cached_tokens"] == 7
        assert outcome["remaining_tokens"] == [outcome["prompt_tokens"][-1]]
        assert outcome["prompt_cache_present"] is True
        assert outcome["kv_offsets"]
        assert outcome["rotating_offsets"]
        assert all(offset == 7 for offset in outcome["kv_offsets"])
        assert all(offset == 7 for offset in outcome["rotating_offsets"])

    def test_exact_hit_mixed_batch_rotating_retained_prefix_rewinds(self):
        """
        Exact-hit kickoff should rewind mixed caches with BatchRotating state.

        BatchRotatingKVCache does not expose ``keep`` directly; non-zero
        ``left_padding`` is the batched retained-prefix analogue and exercises
        ring-buffer edge behavior beyond keep=0 toy setups.
        """
        mx = pytest.importorskip("mlx.core")
        cache_mod = pytest.importorskip("mlx_lm.models.cache")
        from omlx.cache.paged_cache import BlockTable

        prompt_tokens = list(range(2000, 2008))
        scheduler = Scheduler(model=self._DummyModel(2), tokenizer=self._DummyTokenizer())
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = MagicMock()

        kv_layer = cache_mod.KVCache()
        batch_rotating_layer = cache_mod.BatchRotatingKVCache(
            max_size=4, left_padding=[2]
        )
        for _ in prompt_tokens:
            keys = mx.ones((1, 1, 1, 1), dtype=mx.float16)
            values = mx.ones((1, 1, 1, 1), dtype=mx.float16)
            kv_layer.update_and_fetch(keys, values)
            batch_rotating_layer.update_and_fetch(keys, values)

        # Saturated rotating window with retained-prefix semantics.
        assert batch_rotating_layer._offset == 8
        assert batch_rotating_layer.is_trimmable() is False

        block_table = BlockTable(
            request_id="req-batch-rotating-retained-prefix",
            block_ids=[31, 32],
            num_tokens=len(prompt_tokens),
        )
        scheduler.block_aware_cache.fetch_cache.return_value = (block_table, [])
        scheduler.block_aware_cache.reconstruct_cache.return_value = [
            kv_layer,
            batch_rotating_layer,
        ]

        request = Request(
            request_id="req-batch-rotating-retained-prefix",
            prompt=prompt_tokens,
            sampling_params=SamplingParams(max_tokens=8),
        )
        scheduler.add_request(request)

        assert request.cached_tokens == 7
        assert request.remaining_tokens == [prompt_tokens[-1]]
        assert request.prompt_cache is not None
        assert request.block_table is not None
        assert request.shared_prefix_blocks == len(block_table.block_ids)
        assert kv_layer.offset == 7
        assert batch_rotating_layer._offset == 7
        assert batch_rotating_layer.offset.item() == 5
        scheduler.paged_cache_manager.delete_block_table.assert_not_called()


class TestSchedulerAbortRequest:
    """Tests for Scheduler.abort_request() (deferred abort pattern)."""

    def test_abort_enqueues_request(self, mock_model, mock_tokenizer):
        """Test abort_request() enqueues for deferred processing."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)

        result = scheduler.abort_request("test-001")

        # abort_request always returns True (enqueue is always successful)
        assert result is True
        # Request should still be in waiting (not yet processed)
        assert "test-001" in scheduler._pending_abort_ids

    def test_abort_waiting_request(self, mock_model, mock_tokenizer):
        """Test aborting a waiting request via deferred processing."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)

        scheduler.abort_request("test-001")
        scheduler._process_pending_aborts()

        assert request.status == RequestStatus.FINISHED_ABORTED
        assert request not in scheduler.waiting
        assert "test-001" in scheduler.finished_req_ids

    def test_abort_nonexistent_request(self, mock_model, mock_tokenizer):
        """Test aborting a non-existent request is silently ignored."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        result = scheduler.abort_request("nonexistent")
        # Enqueue always succeeds
        assert result is True
        # Processing a non-existent abort is a no-op
        scheduler._process_pending_aborts()

    def test_abort_sets_finish_reason(self, mock_model, mock_tokenizer):
        """Test aborting sets correct finish reason."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)
        scheduler.abort_request("test-001")
        scheduler._process_pending_aborts()

        assert request.get_finish_reason() == "abort"

    def test_abort_running_request_removes_from_batch(
        self, mock_model, mock_tokenizer
    ):
        """Abort must remove active UID from BatchGenerator."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="req-run",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1]
        request.num_prompt_tokens = 1
        request.status = RequestStatus.RUNNING

        uid = 7
        scheduler.requests["req-run"] = request
        scheduler.running["req-run"] = request
        scheduler.request_id_to_uid["req-run"] = uid
        scheduler.uid_to_request_id[uid] = "req-run"

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = MagicMock(uids=[uid])

        scheduler.abort_request("req-run")
        scheduler._process_pending_aborts()

        scheduler.batch_generator.remove.assert_called_once_with([uid])

    def test_abort_running_request_skips_remove_when_uid_not_in_active_batch(
        self, mock_model, mock_tokenizer
    ):
        """Abort must not call remove() when UID is already absent."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="req-run-missing",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1]
        request.num_prompt_tokens = 1
        request.status = RequestStatus.RUNNING

        uid = 8
        scheduler.requests["req-run-missing"] = request
        scheduler.running["req-run-missing"] = request
        scheduler.request_id_to_uid["req-run-missing"] = uid
        scheduler.uid_to_request_id[uid] = "req-run-missing"

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = MagicMock(uids=[999])

        scheduler.abort_request("req-run-missing")
        scheduler._process_pending_aborts()

        scheduler.batch_generator.remove.assert_not_called()

    def test_abort_cleans_all_scheduler_state(self, mock_model, mock_tokenizer):
        """Abort must clean running, uid mappings, and requests dict.

        Regression test: previously _cleanup_request (engine_core) removed
        the request from self.requests before the deferred abort ran,
        causing _do_abort_request to early-return and leave ghost state
        in running/uid mappings/active batch.
        """
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="req-ghost",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1]
        request.num_prompt_tokens = 1
        request.status = RequestStatus.RUNNING

        uid = 10
        scheduler.requests["req-ghost"] = request
        scheduler.running["req-ghost"] = request
        scheduler.request_id_to_uid["req-ghost"] = uid
        scheduler.uid_to_request_id[uid] = "req-ghost"

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = MagicMock(uids=[uid])

        scheduler.abort_request("req-ghost")
        scheduler._process_pending_aborts()

        # All scheduler state must be cleaned
        assert "req-ghost" not in scheduler.running
        assert "req-ghost" not in scheduler.requests
        assert "req-ghost" not in scheduler.request_id_to_uid
        assert uid not in scheduler.uid_to_request_id


class TestPrefillAbortInterrupt:
    """Tests for prefill abort interrupt via _check_pending_aborts_for_uids."""

    def test_check_pending_aborts_returns_aborted_uids(
        self, mock_model, mock_tokenizer
    ):
        """_check_pending_aborts_for_uids returns UIDs with pending aborts."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        # Set up UID mapping
        scheduler.uid_to_request_id[0] = "req-a"
        scheduler.uid_to_request_id[1] = "req-b"
        scheduler._pending_abort_ids.add("req-a")

        result = scheduler._check_pending_aborts_for_uids([0, 1])
        assert result == [0]

    def test_check_pending_aborts_empty_when_no_aborts(
        self, mock_model, mock_tokenizer
    ):
        """Returns empty list when no pending aborts."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.uid_to_request_id[0] = "req-a"

        result = scheduler._check_pending_aborts_for_uids([0])
        assert result == []

    def test_prefill_aborted_error_resets_batch_generator(
        self, mock_model, mock_tokenizer
    ):
        """_PrefillAbortedError in step() resets batch_generator to None."""
        from omlx.scheduler import _PrefillAbortedError

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        scheduler.batch_generator = MagicMock()

        # Make batch_generator.next() raise _PrefillAbortedError
        scheduler.batch_generator.next.side_effect = _PrefillAbortedError(
            [0], 1024
        )
        # Need running requests for next() to be called
        request = Request(
            request_id="req-prefill",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1]
        request.num_prompt_tokens = 1
        request.status = RequestStatus.RUNNING
        scheduler.running["req-prefill"] = request
        scheduler.requests["req-prefill"] = request

        output = scheduler.step()

        # batch_generator should be reset
        assert scheduler.batch_generator is None
        # Request should be moved back to waiting
        assert "req-prefill" not in scheduler.running
        assert len(scheduler.waiting) > 0


class TestSchedulerQueryMethods:
    """Tests for Scheduler query methods."""

    def test_has_requests_empty(self, mock_model, mock_tokenizer):
        """Test has_requests() returns False when empty."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        assert scheduler.has_requests() is False

    def test_has_requests_with_waiting(self, mock_model, mock_tokenizer):
        """Test has_requests() returns True with waiting requests."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)
        assert scheduler.has_requests() is True

    def test_get_num_waiting(self, mock_model, mock_tokenizer):
        """Test get_num_waiting() returns correct count."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        assert scheduler.get_num_waiting() == 0

        for i in range(3):
            request = Request(
                request_id=f"test-{i}",
                prompt=f"Prompt {i}",
                sampling_params=SamplingParams(),
            )
            scheduler.add_request(request)

        assert scheduler.get_num_waiting() == 3

    def test_get_num_running(self, mock_model, mock_tokenizer):
        """Test get_num_running() returns correct count."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        assert scheduler.get_num_running() == 0

        # Manually add to running for testing
        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.running["test-001"] = request

        assert scheduler.get_num_running() == 1

    def test_get_request(self, mock_model, mock_tokenizer):
        """Test get_request() returns correct request."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)

        retrieved = scheduler.get_request("test-001")
        assert retrieved is request

    def test_get_request_nonexistent(self, mock_model, mock_tokenizer):
        """Test get_request() returns None for nonexistent request."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
        assert scheduler.get_request("nonexistent") is None


class TestSchedulerStatistics:
    """Tests for Scheduler.get_stats()."""

    def test_get_stats_initial(self, mock_model, mock_tokenizer):
        """Test get_stats() returns correct initial values."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        stats = scheduler.get_stats()

        assert stats["num_waiting"] == 0
        assert stats["num_running"] == 0
        assert stats["num_requests_processed"] == 0
        assert stats["total_prompt_tokens"] == 0
        assert stats["total_completion_tokens"] == 0

    def test_get_stats_with_requests(self, mock_model, mock_tokenizer):
        """Test get_stats() reflects added requests."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        for i in range(3):
            request = Request(
                request_id=f"test-{i}",
                prompt=f"Prompt {i}",
                sampling_params=SamplingParams(),
            )
            scheduler.add_request(request)

        stats = scheduler.get_stats()

        assert stats["num_waiting"] == 3
        assert stats["num_running"] == 0


class TestSchedulerReset:
    """Tests for Scheduler reset methods."""

    def test_reset_clears_state(self, mock_model, mock_tokenizer):
        """Test reset() clears all scheduler state."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        # Add some requests
        for i in range(3):
            request = Request(
                request_id=f"test-{i}",
                prompt=f"Prompt {i}",
                sampling_params=SamplingParams(),
            )
            scheduler.add_request(request)

        scheduler.reset()

        assert len(scheduler.waiting) == 0
        assert len(scheduler.running) == 0
        assert len(scheduler.requests) == 0
        assert scheduler.batch_generator is None


class TestSchedulerStopTokens:
    """Tests for stop token handling."""

    def test_get_stop_tokens(self, mock_model, mock_tokenizer):
        """Test _get_stop_tokens() retrieves EOS token."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        stop_tokens = scheduler._get_stop_tokens()

        # MockTokenizer has eos_token_id = 2
        assert mock_tokenizer.eos_token_id in stop_tokens


class TestSchedulerFormatBytes:
    """Tests for Scheduler._format_bytes()."""

    def test_format_bytes_bytes(self):
        """Test formatting bytes."""
        assert Scheduler._format_bytes(100) == "100 B"
        assert Scheduler._format_bytes(1023) == "1023 B"

    def test_format_bytes_kilobytes(self):
        """Test formatting kilobytes."""
        result = Scheduler._format_bytes(1024)
        assert "KB" in result

        result = Scheduler._format_bytes(2048)
        assert "2.00 KB" in result

    def test_format_bytes_megabytes(self):
        """Test formatting megabytes."""
        result = Scheduler._format_bytes(1024 * 1024)
        assert "MB" in result

        result = Scheduler._format_bytes(5 * 1024 * 1024)
        assert "5.00 MB" in result

    def test_format_bytes_gigabytes(self):
        """Test formatting gigabytes."""
        result = Scheduler._format_bytes(1024 * 1024 * 1024)
        assert "GB" in result

        result = Scheduler._format_bytes(2 * 1024 * 1024 * 1024)
        assert "2.00 GB" in result


class TestSchedulerRemoveFinishedRequest:
    """Tests for Scheduler.remove_finished_request()."""

    def test_remove_finished_request(self, mock_model, mock_tokenizer):
        """Test removing a finished request from tracking."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="test-001",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        scheduler.add_request(request)

        removed = scheduler.remove_finished_request("test-001")

        assert removed is request
        assert "test-001" not in scheduler.requests

    def test_remove_nonexistent_request(self, mock_model, mock_tokenizer):
        """Test removing nonexistent request returns None."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        result = scheduler.remove_finished_request("nonexistent")

        assert result is None


class TestSchedulerBoundarySnapshots:
    """Tests for boundary cache snapshots on non-sliceable cache models."""

    def test_capture_boundary_snapshot_at_block_boundary(self, mock_model, mock_tokenizer):
        """Capture snapshot when total tokens land exactly on block boundary."""
        config = SchedulerConfig(paged_cache_block_size=4)
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
        scheduler.block_aware_cache = MagicMock()
        scheduler._boundary_snapshot_required = True

        mock_batch = MagicMock()
        mock_batch.uids = [123]
        # Create a non-sliceable batch cache layer (e.g. ArraysCache)
        # so the snapshot capture extracts it instead of replacing with None.
        mock_layer_cache = MagicMock()
        type(mock_layer_cache).__name__ = "BatchArraysCache"
        extracted_cache = MagicMock()
        mock_layer_cache.extract.return_value = extracted_cache
        mock_batch.cache = [mock_layer_cache]

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = mock_batch

        request = Request(
            request_id="req-boundary",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [10, 11]
        request.num_prompt_tokens = 2
        request.output_token_ids = [12, 13]  # Total = 4 (boundary)

        scheduler._maybe_capture_boundary_snapshot(request, 123)

        assert 4 in scheduler._boundary_cache_snapshots["req-boundary"]
        snapshot = scheduler._boundary_cache_snapshots["req-boundary"][4]
        assert snapshot == [extracted_cache]
        mock_layer_cache.extract.assert_called_once_with(0)

    def test_cleanup_finished_uses_boundary_snapshot_for_partial_trailing_tokens(
        self, mock_model, mock_tokenizer
    ):
        """When final length has trailing partial tokens, store boundary snapshot."""
        config = SchedulerConfig(paged_cache_block_size=4)
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = None

        request = Request(
            request_id="req-partial",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3, 4]
        request.num_prompt_tokens = 4
        request.output_token_ids = [5, 6, 7]  # Total = 7 (partial trailing block)
        request._extracted_cache = [{"state": "final-cache"}]
        request._model_cache_config = "final-config"

        scheduler.running["req-partial"] = request
        scheduler.requests["req-partial"] = request
        scheduler._boundary_cache_snapshots["req-partial"] = {4: [MagicMock()]}

        snapshot_extracted = [{"state": "boundary-cache"}]
        with patch.object(
            scheduler,
            "_extract_cache_states",
            return_value=(snapshot_extracted, "boundary-config"),
        ):
            scheduler._cleanup_finished({"req-partial"})

        scheduler.block_aware_cache.store_cache.assert_called_once()
        args, kwargs = scheduler.block_aware_cache.store_cache.call_args
        assert args[0] == "req-partial"
        assert args[1] == [1, 2, 3, 4]
        assert args[2] == snapshot_extracted
        assert kwargs["model_cache_config"] == "boundary-config"
        assert "req-partial" not in scheduler._boundary_cache_snapshots

    def test_boundary_snapshot_synchronizes_generation_stream(
        self, mock_model, mock_tokenizer
    ):
        """Boundary snapshot extraction must synchronize generation_stream
        before accessing batch cache tensors to prevent Metal command buffer conflicts."""
        config = SchedulerConfig(paged_cache_block_size=4)
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
        scheduler.block_aware_cache = MagicMock()
        scheduler._boundary_snapshot_required = True

        mock_batch = MagicMock()
        mock_batch.uids = [42]
        mock_batch.extract_cache.return_value = [MagicMock()]

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = mock_batch

        request = Request(
            request_id="req-sync",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2]
        request.num_prompt_tokens = 2
        request.output_token_ids = [3, 4]  # Total = 4 (boundary)

        with patch("omlx.scheduler.mx") as mock_mx:
            scheduler._maybe_capture_boundary_snapshot(request, 42)
            mock_mx.synchronize.assert_called()
            mock_mx.stream.assert_called()

    def test_cleanup_finished_synchronizes_before_cache_store(
        self, mock_model, mock_tokenizer
    ):
        """_cleanup_finished must synchronize generation_stream before cache
        storage even when active_batch is None (all requests finished)."""
        config = SchedulerConfig(paged_cache_block_size=4)
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
        scheduler.block_aware_cache = MagicMock()
        scheduler.paged_cache_manager = None

        # Simulate active_batch = None (all requests finished in this step)
        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = None

        request = Request(
            request_id="req-cleanup-sync",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3, 4]
        request.num_prompt_tokens = 4
        request.output_token_ids = [5]
        request._extracted_cache = [{"state": "cache"}]
        request._model_cache_config = None

        scheduler.running["req-cleanup-sync"] = request
        scheduler.requests["req-cleanup-sync"] = request

        with patch("omlx.scheduler.mx") as mock_mx:
            scheduler._cleanup_finished({"req-cleanup-sync"})
            mock_mx.synchronize.assert_called()
            mock_mx.stream.assert_called()

    def test_prefill_boundary_snapshot_records_rotating_cache(
        self, mock_model, mock_tokenizer
    ):
        """Prefill callback should store rotating boundary snapshots."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(paged_cache_block_size=4),
        )
        scheduler.block_aware_cache = MagicMock()

        request = Request(
            request_id="req-prefill-boundary",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        uid = 77
        scheduler.requests[request.request_id] = request
        scheduler.running[request.request_id] = request
        scheduler.request_id_to_uid[request.request_id] = uid
        scheduler.uid_to_request_id[uid] = request.request_id

        RotatingStub = type("RotatingKVCache", (), {})
        snapshot_cache = [RotatingStub()]

        scheduler._on_prefill_boundary_snapshot(uid, snapshot_cache, 4)

        assert 4 in scheduler._boundary_cache_snapshots[request.request_id]
        assert scheduler._boundary_cache_snapshots[request.request_id][4] == snapshot_cache
        assert scheduler._boundary_snapshot_required is True

    def test_prefill_boundary_snapshot_ignores_non_boundary_token_count(
        self, mock_model, mock_tokenizer
    ):
        """Prefill callback should ignore non-boundary token counts."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(paged_cache_block_size=4),
        )
        scheduler.block_aware_cache = MagicMock()

        request = Request(
            request_id="req-prefill-non-boundary",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        uid = 78
        scheduler.requests[request.request_id] = request
        scheduler.running[request.request_id] = request
        scheduler.request_id_to_uid[request.request_id] = uid
        scheduler.uid_to_request_id[uid] = request.request_id

        RotatingStub = type("RotatingKVCache", (), {})
        scheduler._on_prefill_boundary_snapshot(uid, [RotatingStub()], 3)

        assert request.request_id not in scheduler._boundary_cache_snapshots


class TestBoundarySnapshotBatchGeneratorPromptTupleCompatibility:
    """Guard prompt-tuple compatibility with mlx-lm BatchGenerator."""

    def _make_bg_stub(self):
        from omlx.scheduler import _BoundarySnapshotBatchGenerator

        bg = object.__new__(_BoundarySnapshotBatchGenerator)
        bg._vlm_pending = {}
        bg._stats = SimpleNamespace(prompt_tokens=0)
        return bg

    def test_process_prompts_accepts_seven_field_prompt_tuples(self, monkeypatch):
        """mlx-lm 0.31+ provides a 7th prompt_checkpoints field."""
        import omlx.scheduler as scheduler_module

        bg = self._make_bg_stub()

        class _StopAfterUnpack(Exception):
            pass

        def _stop_after_unpack(_tokens):
            raise _StopAfterUnpack

        monkeypatch.setattr(scheduler_module.mx, "array", _stop_after_unpack)
        prompts = [(7, [1, 2], 16, [MagicMock()], None, [], -1)]

        with pytest.raises(_StopAfterUnpack):
            bg._process_prompts(prompts)

    def test_process_prompts_still_accepts_six_field_prompt_tuples(self, monkeypatch):
        """Older prompt tuples without prompt_checkpoints stay supported."""
        import omlx.scheduler as scheduler_module

        bg = self._make_bg_stub()

        class _StopAfterUnpack(Exception):
            pass

        def _stop_after_unpack(_tokens):
            raise _StopAfterUnpack

        monkeypatch.setattr(scheduler_module.mx, "array", _stop_after_unpack)
        prompts = [(8, [3, 4], 32, [MagicMock()], None, [])]

        with pytest.raises(_StopAfterUnpack):
            bg._process_prompts(prompts)


class TestSchedulerRotatingBlockAlignment:
    """Tests for rotating window/block-size alignment."""

    def test_aligns_block_size_to_rotating_window(self, mock_tokenizer):
        RotatingStub = type("RotatingKVCache", (), {})

        class RotatingModel:
            def __init__(self):
                self.config = MagicMock()
                self.config.num_hidden_layers = 1

            def make_cache(self):
                cache = RotatingStub()
                cache.max_size = 128
                return [cache]

        scheduler = Scheduler(
            model=RotatingModel(),
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(paged_cache_block_size=256),
        )
        scheduler.config.paged_ssd_cache_dir = "/tmp/cache"
        scheduler._align_block_size_with_rotating_window()

        assert scheduler.config.paged_cache_block_size == 128

    def test_multiple_rotating_window_sizes_raise(self, mock_tokenizer):
        RotatingStub = type("RotatingKVCache", (), {})

        class MultiRotatingModel:
            def __init__(self):
                self.config = MagicMock()
                self.config.num_hidden_layers = 2

            def make_cache(self):
                c1 = RotatingStub()
                c1.max_size = 128
                c2 = RotatingStub()
                c2.max_size = 256
                return [c1, c2]

        scheduler = Scheduler(
            model=MultiRotatingModel(),
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(paged_cache_block_size=256),
        )
        scheduler.config.paged_ssd_cache_dir = "/tmp/cache"

        with pytest.raises(ValueError):
            scheduler._align_block_size_with_rotating_window()

    def test_cleanup_finished_skips_remove_when_uid_not_in_active_batch(
        self, mock_model, mock_tokenizer
    ):
        """_cleanup_finished should not call remove() for already-filtered UIDs."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="req-skip-remove",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2]
        request.num_prompt_tokens = 2
        request.output_token_ids = [3]

        uid = 55
        scheduler.running["req-skip-remove"] = request
        scheduler.requests["req-skip-remove"] = request
        scheduler.request_id_to_uid["req-skip-remove"] = uid
        scheduler.uid_to_request_id[uid] = "req-skip-remove"

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = MagicMock(uids=[77])

        scheduler._cleanup_finished({"req-skip-remove"})

        scheduler.batch_generator.remove.assert_not_called()

    def test_cleanup_finished_removes_uid_from_active_batch(
        self, mock_model, mock_tokenizer
    ):
        """_cleanup_finished should remove active UID from batch."""
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        request = Request(
            request_id="req-remove-active",
            prompt="hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2]
        request.num_prompt_tokens = 2
        request.output_token_ids = [3]

        uid = 56
        scheduler.running["req-remove-active"] = request
        scheduler.requests["req-remove-active"] = request
        scheduler.request_id_to_uid["req-remove-active"] = uid
        scheduler.uid_to_request_id[uid] = "req-remove-active"

        scheduler.batch_generator = MagicMock()
        scheduler.batch_generator.active_batch = MagicMock(uids=[uid])

        scheduler._cleanup_finished({"req-remove-active"})

        scheduler.batch_generator.remove.assert_called_once_with([uid])


class TestExtractCacheStatesCacheList:
    """Tests for CacheList handling in _extract_cache_states."""

    @pytest.fixture
    def scheduler(self):
        """Create a minimal scheduler mock for testing _extract_cache_states."""
        from omlx.scheduler import Scheduler

        mock_scheduler = MagicMock(spec=Scheduler)
        mock_scheduler.model_name = "test"
        mock_scheduler._extract_cache_states = Scheduler._extract_cache_states.__get__(
            mock_scheduler, Scheduler
        )
        return mock_scheduler

    def test_extract_cache_states_cache_list(self, scheduler):
        """Test CacheList layer extraction."""
        # Create a mock CacheList object
        mock_kv_sub = MagicMock(spec=[])
        mock_kv_sub.__class__ = type("KVCache", (), {})
        mock_kv_sub.state = (MagicMock(), MagicMock())
        mock_kv_sub.meta_state = (32,)

        mock_cache_list = MagicMock(spec=[])
        mock_cache_list.__class__ = type("CacheList", (), {})
        mock_cache_list.caches = (mock_kv_sub,)
        mock_cache_list.state = [(MagicMock(), MagicMock())]  # CacheList.state
        mock_cache_list.meta_state = (["KVCache"], [(32,)])

        # Standard KVCache layer
        mock_kv = MagicMock(spec=[])
        mock_kv.__class__ = type("KVCache", (), {})
        mock_kv.state = (MagicMock(), MagicMock())
        mock_kv.meta_state = (64,)

        raw_cache = [mock_cache_list, mock_kv]

        extracted, config = scheduler._extract_cache_states(raw_cache)

        assert len(extracted) == 2
        assert extracted[0]['class_name'] == 'CacheList'
        assert extracted[0]['cache_type'] == 'CacheList'
        assert isinstance(extracted[0]['state'], list)
        assert isinstance(extracted[0]['meta_state'], tuple)
        assert len(extracted[0]['meta_state']) == 2

    def test_extract_cache_states_cache_list_no_handlers(self, scheduler):
        """Test CacheList extraction when HAS_CACHE_TYPE_HANDLERS=False."""
        # Use real stub classes so type(obj).__name__ returns the correct name
        # (needed because the fallback branch uses type().__name__ for detection)
        KVCacheStub = type("KVCache", (), {
            "state": (MagicMock(), MagicMock()),
            "meta_state": (32,),
        })
        mock_kv_sub = KVCacheStub()

        CacheListStub = type("CacheList", (), {
            "caches": (mock_kv_sub,),
            "state": [(MagicMock(), MagicMock())],
            "meta_state": (["KVCache"], [(32,)]),
        })
        mock_cache_list = CacheListStub()

        raw_cache = [mock_cache_list]

        # Patch HAS_CACHE_TYPE_HANDLERS to False
        with patch('omlx.scheduler.HAS_CACHE_TYPE_HANDLERS', False):
            extracted, config = scheduler._extract_cache_states(raw_cache)

        # Must still have 1 extracted entry (Issue #1: no layer count mismatch)
        assert len(extracted) == 1
        assert extracted[0]['class_name'] == 'CacheList'
        assert isinstance(extracted[0]['state'], list)


class TestExtractCacheStatesRotatingNormalization:
    """Tests for RotatingKVCache snapshot normalization during extraction."""

    def test_extract_cache_states_normalizes_oversized_rotating_snapshot(
        self, mock_model, mock_tokenizer
    ):
        """Oversized rotating snapshot should be canonicalized to max_size."""
        mx = pytest.importorskip("mlx.core")
        cache_mod = pytest.importorskip("mlx_lm.models.cache")
        RotatingKVCache = cache_mod.RotatingKVCache

        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)

        rotating = RotatingKVCache(max_size=128, keep=0)
        rotating.keys = mx.arange(255).reshape(1, 1, 255, 1)
        rotating.values = mx.arange(1000, 1255).reshape(1, 1, 255, 1)
        rotating.offset = 1280
        rotating._idx = 255

        expected_keys = rotating.keys[..., -128:, :]
        expected_values = rotating.values[..., -128:, :]

        extracted, _ = scheduler._extract_cache_states([rotating])

        assert len(extracted) == 1
        normalized_keys, normalized_values = extracted[0]["state"]
        normalized_meta = tuple(extracted[0]["meta_state"])

        assert normalized_keys.shape == (1, 1, 128, 1)
        assert normalized_values.shape == (1, 1, 128, 1)
        assert bool(mx.all(normalized_keys == expected_keys).item())
        assert bool(mx.all(normalized_values == expected_values).item())
        assert normalized_meta == ("0", "128", "1280", "128")

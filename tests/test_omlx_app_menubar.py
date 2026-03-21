"""Targeted tests for menubar monitoring text in the native app."""

import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest


APP_DIR = Path(__file__).resolve().parents[1] / "packaging" / "omlx_app"


def _load_module(name: str, path: Path):
    """Load a module from disk under a specific import name."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def app_module():
    """Import the real app module with only its direct dependencies mocked."""
    saved = {}
    module_names = [
        "AppKit",
        "Foundation",
        "objc",
        "omlx",
        "omlx._version",
        "omlx_app",
        "omlx_app.config",
        "omlx_app.server_manager",
        "omlx_app.app",
    ]
    for name in module_names:
        saved[name] = sys.modules.get(name)

    appkit = types.ModuleType("AppKit")
    appkit.NSApp = MagicMock()
    appkit.NSAppearanceNameDarkAqua = "dark"
    appkit.NSApplication = MagicMock()
    appkit.NSApplicationActivationPolicyAccessory = 1
    appkit.NSApplicationActivationPolicyRegular = 0
    appkit.NSAttributedString = MagicMock()
    appkit.NSBundle = MagicMock()
    appkit.NSColor = MagicMock()
    appkit.NSForegroundColorAttributeName = "foreground"
    appkit.NSImage = MagicMock()
    appkit.NSMenu = MagicMock()
    appkit.NSMenuItem = MagicMock()
    appkit.NSStatusBar = MagicMock()
    appkit.NSVariableStatusItemLength = -1

    foundation = types.ModuleType("Foundation")
    foundation.NSData = MagicMock()
    foundation.NSObject = object
    foundation.NSRunLoop = MagicMock()
    foundation.NSDefaultRunLoopMode = "default"
    foundation.NSTimer = MagicMock()

    objc_mod = types.ModuleType("objc")
    objc_mod.super = super
    objc_mod.IBAction = lambda fn: fn

    fake_omlx = types.ModuleType("omlx")
    fake_omlx.__path__ = []
    fake_version = types.ModuleType("omlx._version")
    fake_version.__version__ = "0.0.test"

    fake_package = types.ModuleType("omlx_app")
    fake_package.__path__ = [str(APP_DIR)]
    fake_config = types.ModuleType("omlx_app.config")
    fake_server_manager = types.ModuleType("omlx_app.server_manager")

    class FakeServerConfig:
        @classmethod
        def load(cls):
            return cls()

    class FakeServerStatus(Enum):
        STOPPED = "stopped"
        STARTING = "starting"
        RUNNING = "running"
        STOPPING = "stopping"
        ERROR = "error"
        UNRESPONSIVE = "unresponsive"

    class FakePortConflict:
        pass

    class FakeServerManager:
        def __init__(self, config):
            self.config = config
            self.status = FakeServerStatus.STOPPED

    fake_config.ServerConfig = FakeServerConfig
    fake_server_manager.PortConflict = FakePortConflict
    fake_server_manager.ServerManager = FakeServerManager
    fake_server_manager.ServerStatus = FakeServerStatus

    try:
        sys.modules["AppKit"] = appkit
        sys.modules["Foundation"] = foundation
        sys.modules["objc"] = objc_mod
        sys.modules["omlx"] = fake_omlx
        sys.modules["omlx._version"] = fake_version
        sys.modules["omlx_app"] = fake_package
        sys.modules["omlx_app.config"] = fake_config
        sys.modules["omlx_app.server_manager"] = fake_server_manager
        yield _load_module("omlx_app.app", APP_DIR / "app.py")
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class TestMenubarMonitoring:
    """Tests for live menubar monitoring text in the native app."""

    @staticmethod
    def _make_delegate(app_module, stats, status):
        button = Mock()
        status_item = Mock()
        status_item.button.return_value = button
        delegate = types.SimpleNamespace(
            status_item=status_item,
            server_manager=types.SimpleNamespace(status=status),
            _icon_outline="outline-icon",
            _icon_filled="filled-icon",
            _cached_stats=stats,
        )
        return delegate, status_item, button

    @staticmethod
    def _last_single_arg(mock_method):
        assert mock_method.call_args_list, "Expected method to be called at least once"
        args, kwargs = mock_method.call_args_list[-1]
        assert not kwargs
        assert len(args) == 1
        return args[0]

    def _assert_final_icon(self, button, expected_icon):
        assert self._last_single_arg(button.setImage_) == expected_icon

    def _assert_title_fragments(
        self,
        status_item,
        *,
        includes=(),
        excludes=(),
        exact=None,
    ):
        title = self._last_single_arg(status_item.setTitle_)
        if exact is not None:
            assert title == exact
            return
        for fragment in includes:
            assert fragment in title
        for fragment in excludes:
            assert fragment not in title

    def test_update_menubar_icon_prefers_live_prefill_progress(self, app_module):
        """Prefill activity should surface compact token/speed/ETA monitoring."""
        stats = {
            "avg_generation_tps": 78.4,
            "active_models": {
                "total_active_requests": 2,
                "total_waiting_requests": 1,
                "models": [
                    {
                        "id": "mlx-community/Qwen3-Coder-30B-A3B",
                        "active_requests": 1,
                        "waiting_requests": 1,
                        "prefilling": [
                            {
                                "request_id": "req-123",
                                "processed": 12345,
                                "total": 67890,
                                "speed": 511.6,
                                "eta": 65.4,
                            }
                        ],
                    }
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module, stats, app_module.ServerStatus.RUNNING
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        self._assert_title_fragments(
            status_item,
            includes=("1 PP", "12.3k/67.9k tok", "512 tok/s", "1m 5s"),
            excludes=("None", "req", "wait", "78.4 tok/s"),
        )

    def test_update_menubar_icon_finds_prefill_beyond_first_model_row(self, app_module):
        """Prefill should win even when the first model row has only generation/backlog state."""
        stats = {
            "avg_generation_tps": 78.4,
            "active_models": {
                "total_active_requests": 2,
                "total_waiting_requests": 3,
                "models": [
                    {
                        "id": "mlx-community/Qwen3-32B",
                        "active_requests": 2,
                        "waiting_requests": 2,
                        "prefilling": [],
                    },
                    {
                        "id": "mlx-community/Qwen3-14B",
                        "active_requests": 0,
                        "waiting_requests": 1,
                        "prefilling": [
                            {
                                "request_id": "req-late",
                                "processed": 4096,
                                "total": 16384,
                                "speed": 256.2,
                                "eta": 48.8,
                            }
                        ],
                    },
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module, stats, app_module.ServerStatus.RUNNING
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        self._assert_title_fragments(
            status_item,
            includes=("1 PP", "4.1k/16.4k tok", "256 tok/s", "49s"),
            excludes=("req", "wait", "78.4 tok/s"),
        )

    def test_update_menubar_icon_omits_speed_and_eta_for_first_prefill_sample(self, app_module):
        """First prefill observation should not render placeholder speed/ETA text."""
        stats = {
            "avg_generation_tps": 78.4,
            "active_models": {
                "total_active_requests": 0,
                "total_waiting_requests": 1,
                "models": [
                    {
                        "id": "mlx-community/Qwen3-Coder-30B-A3B",
                        "active_requests": 0,
                        "waiting_requests": 1,
                        "prefilling": [
                            {
                                "request_id": "req-boot",
                                "processed": 512,
                                "total": 8000,
                                "speed": 0.0,
                                "eta": None,
                            }
                        ],
                    }
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module, stats, app_module.ServerStatus.RUNNING
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        self._assert_title_fragments(
            status_item,
            includes=("1 PP", "512/8.0k tok"),
            excludes=("tok/s", "None", "left", "req", "wait"),
        )

    def test_update_menubar_icon_shows_live_request_counts_when_generating(self, app_module):
        """Generation-only activity should use aggregate load, not the first model row."""
        stats = {
            "avg_generation_tps": 78.4,
            "active_models": {
                "total_active_requests": 3,
                "total_waiting_requests": 4,
                "models": [
                    {
                        "id": "mlx-community/Qwen3-32B",
                        "active_requests": 2,
                        "waiting_requests": 1,
                        "prefilling": [],
                    },
                    {
                        "id": "mlx-community/Qwen3-14B",
                        "active_requests": 1,
                        "waiting_requests": 3,
                        "prefilling": [],
                    }
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module, stats, app_module.ServerStatus.RUNNING
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        self._assert_title_fragments(
            status_item,
            includes=("3 req", "4 wait", "78.4 tok/s"),
            excludes=("PP", "12.3k/67.9k tok", "512/8.0k tok"),
        )

    def test_update_menubar_icon_shows_queue_only_backlog_as_live_signal(self, app_module):
        """A pure waiting backlog should not collapse to idle or display stale speed."""
        stats = {
            "avg_generation_tps": 78.4,
            "active_models": {
                "total_active_requests": 0,
                "total_waiting_requests": 4,
                "models": [
                    {
                        "id": "mlx-community/Qwen3-32B",
                        "active_requests": 0,
                        "waiting_requests": 1,
                        "prefilling": [],
                    },
                    {
                        "id": "mlx-community/Qwen3-14B",
                        "active_requests": 0,
                        "waiting_requests": 3,
                        "prefilling": [],
                    },
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module, stats, app_module.ServerStatus.RUNNING
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        self._assert_title_fragments(
            status_item,
            includes=("4 wait",),
            excludes=("idle", "tok/s", "PP", "req"),
        )

    def test_update_menubar_icon_stays_compact_when_no_live_activity(self, app_module):
        """Idle servers should keep the icon-only menubar presentation."""
        stats = {
            "avg_generation_tps": 78.4,
            "active_models": {
                "total_active_requests": 0,
                "total_waiting_requests": 0,
                "models": [
                    {
                        "id": "mlx-community/Qwen3-32B",
                        "active_requests": 0,
                        "waiting_requests": 0,
                        "prefilling": [],
                    }
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module, stats, app_module.ServerStatus.RUNNING
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        self._assert_title_fragments(status_item, exact="")

    def test_update_menubar_icon_clears_stale_monitoring_while_starting(self, app_module):
        """STARTING should not reuse stale live-monitor text from a previous run."""
        stats = {
            "avg_generation_tps": 78.4,
            "active_models": {
                "total_active_requests": 3,
                "total_waiting_requests": 4,
                "models": [
                    {
                        "id": "mlx-community/Qwen3-32B",
                        "active_requests": 3,
                        "waiting_requests": 4,
                        "prefilling": [],
                    }
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module, stats, app_module.ServerStatus.STARTING
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        self._assert_title_fragments(status_item, exact="")

    @pytest.mark.parametrize(
        "status_name",
        ["STOPPED", "STOPPING", "ERROR", "UNRESPONSIVE"],
    )
    def test_update_menubar_icon_clears_stale_monitoring_when_server_is_not_running(
        self, app_module, status_name
    ):
        """Cached monitoring text must disappear for any non-running server state."""
        stats = {
            "avg_generation_tps": 78.4,
            "active_models": {
                "total_active_requests": 3,
                "total_waiting_requests": 4,
                "models": [
                    {
                        "id": "mlx-community/Qwen3-32B",
                        "active_requests": 3,
                        "waiting_requests": 4,
                        "prefilling": [],
                    }
                ],
            },
        }
        status = getattr(app_module.ServerStatus, status_name)
        delegate, status_item, button = self._make_delegate(
            app_module, stats, status
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "outline-icon")
        self._assert_title_fragments(status_item, exact="")

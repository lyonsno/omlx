"""Targeted tests for compact menubar metrics in the native app."""

import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, Mock

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
        "omlx_app.updater",
        "omlx_app.app",
    ]
    for name in module_names:
        saved[name] = sys.modules.get(name)

    class FakeFont:
        def __init__(self, size, family="system", weight=None):
            self.size = size
            self.family = family
            self.weight = weight

        @classmethod
        def systemFontOfSize_(cls, size):
            return cls(size, family="system")

        @classmethod
        def systemFontOfSize_weight_(cls, size, weight):
            return cls(size, family="system", weight=weight)

        @classmethod
        def monospacedDigitSystemFontOfSize_weight_(cls, size, weight):
            return cls(size, family="monospaced-digit", weight=weight)

    class FakeAttributedString:
        def __init__(self):
            self.string = ""
            self.base_attributes = {}
            self.range_attributes = []

        @classmethod
        def alloc(cls):
            return cls()

        def initWithString_attributes_(self, string, attributes):
            self.string = string
            self.base_attributes = dict(attributes or {})
            return self

    class FakeMutableAttributedString(FakeAttributedString):
        def addAttribute_value_range_(self, name, value, ns_range):
            self.range_attributes.append((name, value, ns_range))
            return self

    appkit = types.ModuleType("AppKit")
    appkit.NSApp = MagicMock()
    appkit.NSAppearanceNameDarkAqua = "dark"
    appkit.NSApplication = MagicMock()
    appkit.NSApplicationActivationPolicyAccessory = 1
    appkit.NSApplicationActivationPolicyRegular = 0
    appkit.NSAttributedString = FakeAttributedString
    appkit.NSBundle = MagicMock()
    appkit.NSColor = MagicMock()
    appkit.NSFont = FakeFont
    appkit.NSFontAttributeName = "font"
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
    foundation.NSMakeRange = lambda start, length: (start, length)
    foundation.NSMutableAttributedString = FakeMutableAttributedString
    foundation.NSTimer = MagicMock()

    objc_mod = types.ModuleType("objc")
    objc_mod.super = super
    objc_mod.IBAction = lambda fn: fn
    objc_mod.selector = lambda fn, signature=None: fn

    fake_omlx = types.ModuleType("omlx")
    fake_omlx.__path__ = []
    fake_version = types.ModuleType("omlx._version")
    fake_version.__version__ = "0.0.test"

    fake_package = types.ModuleType("omlx_app")
    fake_package.__path__ = [str(APP_DIR)]
    fake_config = types.ModuleType("omlx_app.config")
    fake_server_manager = types.ModuleType("omlx_app.server_manager")
    fake_updater = types.ModuleType("omlx_app.updater")

    class FakeServerConfig:
        def __init__(self):
            self.stats_refresh_interval = 5
            self.is_first_run = False
            self.start_server_on_launch = False

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
    fake_updater.AppUpdater = types.SimpleNamespace(cleanup_staged_app=Mock())

    try:
        sys.modules["AppKit"] = appkit
        sys.modules["Foundation"] = foundation
        sys.modules["objc"] = objc_mod
        sys.modules["omlx"] = fake_omlx
        sys.modules["omlx._version"] = fake_version
        sys.modules["omlx_app"] = fake_package
        sys.modules["omlx_app.config"] = fake_config
        sys.modules["omlx_app.server_manager"] = fake_server_manager
        sys.modules["omlx_app.updater"] = fake_updater
        yield _load_module("omlx_app.app", APP_DIR / "app.py")
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


@pytest.fixture
def preferences_module():
    """Import the real preferences module with only its direct dependencies mocked."""
    saved = {}
    module_names = [
        "AppKit",
        "Foundation",
        "objc",
        "omlx_app",
        "omlx_app.widgets",
        "omlx_app.preferences",
    ]
    for name in module_names:
        saved[name] = sys.modules.get(name)

    class FakeFont:
        def __init__(self, size, family="system", weight=None):
            self.size = size
            self.family = family
            self.weight = weight

        @classmethod
        def systemFontOfSize_(cls, size):
            return cls(size, family="system")

        @classmethod
        def systemFontOfSize_weight_(cls, size, weight):
            return cls(size, family="system", weight=weight)

        @classmethod
        def monospacedSystemFontOfSize_weight_(cls, size, weight):
            return cls(size, family="monospaced", weight=weight)

    class FakeColor:
        @staticmethod
        def controlAccentColor():
            return "accent"

        @staticmethod
        def secondaryLabelColor():
            return "secondary"

        @staticmethod
        def labelColor():
            return "label"

        @staticmethod
        def tertiaryLabelColor():
            return "tertiary"

    appkit = types.ModuleType("AppKit")
    appkit.NSAlert = MagicMock()
    appkit.NSAlertFirstButtonReturn = 1000
    appkit.NSAlertStyleCritical = 2
    appkit.NSAlertStyleWarning = 1
    appkit.NSApp = MagicMock()
    appkit.NSBackingStoreBuffered = 2
    appkit.NSBezelStyleRounded = 1
    appkit.NSBox = MagicMock()
    appkit.NSBoxCustom = 1
    appkit.NSBoxSeparator = 2
    appkit.NSButton = MagicMock()
    appkit.NSButtonTypeSwitch = 3
    appkit.NSColor = FakeColor
    appkit.NSControlStateValueOff = 0
    appkit.NSControlStateValueOn = 1
    appkit.NSFont = FakeFont
    appkit.NSImage = MagicMock()
    appkit.NSMakeRect = lambda *args: args
    appkit.NSOpenPanel = MagicMock()
    appkit.NSTextField = MagicMock()
    appkit.NSView = MagicMock()
    appkit.NSVisualEffectBlendingModeBehindWindow = 1
    appkit.NSVisualEffectMaterialSidebar = 1
    appkit.NSVisualEffectView = MagicMock()
    appkit.NSWindow = MagicMock()
    appkit.NSWindowStyleMaskClosable = 1
    appkit.NSWindowStyleMaskTitled = 2

    foundation = types.ModuleType("Foundation")
    foundation.NSObject = object

    objc_mod = types.ModuleType("objc")
    objc_mod.super = super
    objc_mod.IBAction = lambda fn: fn
    objc_mod.selector = lambda fn, signature=None: fn

    fake_package = types.ModuleType("omlx_app")
    fake_package.__path__ = [str(APP_DIR)]
    fake_widgets = types.ModuleType("omlx_app.widgets")

    class FakePastableSecureTextField:
        pass

    fake_widgets.PastableSecureTextField = FakePastableSecureTextField

    try:
        sys.modules["AppKit"] = appkit
        sys.modules["Foundation"] = foundation
        sys.modules["objc"] = objc_mod
        sys.modules["omlx_app"] = fake_package
        sys.modules["omlx_app.widgets"] = fake_widgets
        yield _load_module("omlx_app.preferences", APP_DIR / "preferences.py")
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class TestMenubarMonitoring:
    """Tests for compact live menubar metrics in the native app."""

    @staticmethod
    def _make_delegate(
        app_module,
        stats,
        status,
        *,
        show_live_metrics=False,
        icons_available=True,
    ):
        button = Mock()
        status_item = Mock()
        status_item.button.return_value = button
        delegate = types.SimpleNamespace(
            status_item=status_item,
            server_manager=types.SimpleNamespace(status=status),
            config=types.SimpleNamespace(
                show_live_metrics_in_menu_bar=show_live_metrics
            ),
            _icon_outline="outline-icon" if icons_available else None,
            _icon_filled="filled-icon" if icons_available else None,
            _cached_stats=stats,
        )
        assert (
            delegate.config.show_live_metrics_in_menu_bar == show_live_metrics
        ), "Delegate fixture failed to wire the live-metrics opt-in flag"
        for method_name in (
            "_format_menubar_title",
            "_format_token_count",
            "_format_speed",
            "_format_eta",
        ):
            setattr(
                delegate,
                method_name,
                getattr(app_module.OMLXAppDelegate, method_name).__get__(delegate),
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

    def _assert_displayed_title(self, status_item, button, expected_title):
        if button.setAttributedTitle_.call_args_list:
            attributed = self._last_single_arg(button.setAttributedTitle_)
            assert attributed.string == expected_title
            assert self._last_single_arg(status_item.setTitle_) == ""
            return attributed

        assert self._last_single_arg(status_item.setTitle_) == expected_title
        return None

    def _assert_variable_width(self, status_item, app_module):
        assert (
            self._last_single_arg(status_item.setLength_)
            == app_module.NSVariableStatusItemLength
        )

    def _minimum_live_metric_width(self, title: str) -> int:
        """Conservative floor derived from the widest compact badge string."""
        return 18 + (len(title) * 5)

    def _assert_fixed_width(self, status_item, app_module, minimum_width=None):
        length = self._last_single_arg(status_item.setLength_)
        assert isinstance(length, (int, float))
        assert length > 0
        assert length != app_module.NSVariableStatusItemLength
        if minimum_width is not None:
            assert length >= minimum_width
        return length

    def _assert_generation_unit_is_deemphasized(self, attributed, expected_title):
        unit_start = expected_title.index("tok/s")
        unit_range = (unit_start, len("tok/s"))
        base_font = attributed.base_attributes.get("font")
        assert base_font is not None
        unit_font = None
        for name, value, ns_range in attributed.range_attributes:
            if name == "font" and ns_range == unit_range:
                unit_font = value
                break
        assert unit_font is not None
        assert unit_font.size < base_font.size

    def test_live_metrics_are_opt_in_even_when_activity_exists(self, app_module):
        """The default behavior should remain icon-only even with live server activity."""
        stats = {
            "avg_generation_tps": 78.6,
            "active_models": {
                "total_active_requests": 2,
                "total_waiting_requests": 3,
                "models": [
                    {
                        "id": "mlx-community/Qwen3-32B",
                        "active_requests": 1,
                        "waiting_requests": 2,
                        "prefilling": [
                            {
                                "request_id": "req-1",
                                "processed": 1500,
                                "total": 3000,
                                "speed": 300.0,
                                "eta": 5.0,
                            }
                        ],
                    }
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module,
            stats,
            app_module.ServerStatus.RUNNING,
            show_live_metrics=False,
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        self._assert_variable_width(status_item, app_module)
        self._assert_displayed_title(status_item, button, "")

    def test_prefill_shows_aggregate_percentage_when_enabled(self, app_module):
        """Prefill should surface raw aggregate progress with no prefix."""
        stats = {
            "avg_generation_tps": 78.6,
            "active_models": {
                "total_active_requests": 2,
                "total_waiting_requests": 4,
                "models": [
                    {
                        "id": "model-a",
                        "active_requests": 1,
                        "waiting_requests": 2,
                        "prefilling": [
                            {
                                "request_id": "req-a",
                                "processed": 600,
                                "total": 1000,
                                "speed": 240.0,
                                "eta": 1.7,
                            }
                        ],
                    },
                    {
                        "id": "model-b",
                        "active_requests": 1,
                        "waiting_requests": 2,
                        "prefilling": [
                            {
                                "request_id": "req-b",
                                "processed": 1400,
                                "total": 4000,
                                "speed": 280.0,
                                "eta": 9.3,
                            }
                        ],
                    },
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module,
            stats,
            app_module.ServerStatus.RUNNING,
            show_live_metrics=True,
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        self._assert_displayed_title(status_item, button, "40%")

    def test_generation_shows_compact_k_badge_with_smaller_unit(self, app_module):
        """Generation should use an attributed compact badge so the unit can be deemphasized."""
        stats = {
            "avg_generation_tps": 1024.6,
            "active_models": {
                "total_active_requests": 3,
                "total_waiting_requests": 4,
                "models": [
                    {
                        "id": "model-a",
                        "active_requests": 2,
                        "waiting_requests": 1,
                        "prefilling": [],
                    },
                    {
                        "id": "model-b",
                        "active_requests": 1,
                        "waiting_requests": 3,
                        "prefilling": [],
                    },
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module,
            stats,
            app_module.ServerStatus.RUNNING,
            show_live_metrics=True,
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        attributed = self._assert_displayed_title(status_item, button, "1k tok/s")
        assert attributed is not None
        self._assert_generation_unit_is_deemphasized(attributed, "1k tok/s")

    @pytest.mark.parametrize("bad_tps", [float("nan"), float("inf"), float("-inf")])
    def test_generation_non_finite_tps_degrades_to_zero_badge(
        self, app_module, bad_tps
    ):
        """Malformed generation TPS values should not crash the menubar update."""
        stats = {
            "avg_generation_tps": bad_tps,
            "active_models": {
                "total_active_requests": 1,
                "total_waiting_requests": 0,
                "models": [
                    {
                        "id": "model-a",
                        "active_requests": 1,
                        "waiting_requests": 0,
                        "prefilling": [],
                    }
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module,
            stats,
            app_module.ServerStatus.RUNNING,
            show_live_metrics=True,
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        attributed = self._assert_displayed_title(status_item, button, "0 tok/s")
        assert attributed is not None
        self._assert_generation_unit_is_deemphasized(attributed, "0 tok/s")

    def test_queue_only_shows_compact_queue_badge_when_enabled(self, app_module):
        """Queue depth should only appear when backlog is the only live signal."""
        stats = {
            "avg_generation_tps": 78.6,
            "active_models": {
                "total_active_requests": 0,
                "total_waiting_requests": 4,
                "models": [
                    {
                        "id": "model-a",
                        "active_requests": 0,
                        "waiting_requests": 1,
                        "prefilling": [],
                    },
                    {
                        "id": "model-b",
                        "active_requests": 0,
                        "waiting_requests": 3,
                        "prefilling": [],
                    },
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module,
            stats,
            app_module.ServerStatus.RUNNING,
            show_live_metrics=True,
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        self._assert_displayed_title(status_item, button, "Q4")

    def test_enabled_metrics_stay_compact_when_idle(self, app_module):
        """Even when enabled, idle servers should keep the icon-only presentation."""
        stats = {
            "avg_generation_tps": 78.6,
            "active_models": {
                "total_active_requests": 0,
                "total_waiting_requests": 0,
                "models": [
                    {
                        "id": "model-a",
                        "active_requests": 0,
                        "waiting_requests": 0,
                        "prefilling": [],
                    }
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module,
            stats,
            app_module.ServerStatus.RUNNING,
            show_live_metrics=True,
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        self._assert_variable_width(status_item, app_module)
        self._assert_displayed_title(status_item, button, "")

    def test_starting_clears_stale_metrics_even_when_enabled(self, app_module):
        """STARTING should not reuse stale metrics from a previous run."""
        stats = {
            "avg_generation_tps": 78.6,
            "active_models": {
                "total_active_requests": 3,
                "total_waiting_requests": 4,
                "models": [
                    {
                        "id": "model-a",
                        "active_requests": 3,
                        "waiting_requests": 4,
                        "prefilling": [],
                    }
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module,
            stats,
            app_module.ServerStatus.STARTING,
            show_live_metrics=True,
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "filled-icon")
        self._assert_variable_width(status_item, app_module)
        self._assert_displayed_title(status_item, button, "")

    @pytest.mark.parametrize(
        "status_name",
        ["STOPPED", "STOPPING", "ERROR", "UNRESPONSIVE"],
    )
    def test_non_running_states_clear_stale_metrics(self, app_module, status_name):
        """Non-running states should stay metric-free even with cached stats."""
        stats = {
            "avg_generation_tps": 78.6,
            "active_models": {
                "total_active_requests": 3,
                "total_waiting_requests": 4,
                "models": [
                    {
                        "id": "model-a",
                        "active_requests": 3,
                        "waiting_requests": 4,
                        "prefilling": [],
                    }
                ],
            },
        }
        status = getattr(app_module.ServerStatus, status_name)
        delegate, status_item, button = self._make_delegate(
            app_module,
            stats,
            status,
            show_live_metrics=True,
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        self._assert_final_icon(button, "outline-icon")
        self._assert_variable_width(status_item, app_module)
        self._assert_displayed_title(status_item, button, "")

    def test_missing_icons_fall_back_to_omlx_when_metrics_disabled(self, app_module):
        """Missing icons should still leave a visible fallback when metrics are off."""
        stats = {
            "avg_generation_tps": 0.0,
            "active_models": {
                "total_active_requests": 0,
                "total_waiting_requests": 0,
                "models": [],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module,
            stats,
            app_module.ServerStatus.STOPPED,
            show_live_metrics=False,
            icons_available=False,
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        button.setImage_.assert_not_called()
        self._assert_variable_width(status_item, app_module)
        self._assert_displayed_title(status_item, button, "oMLX")

    def test_missing_icons_fall_back_to_compact_metric_when_enabled(self, app_module):
        """Missing icons should still use the compact generation badge text."""
        stats = {
            "avg_generation_tps": 1024.6,
            "active_models": {
                "total_active_requests": 3,
                "total_waiting_requests": 4,
                "models": [
                    {
                        "id": "model-a",
                        "active_requests": 2,
                        "waiting_requests": 1,
                        "prefilling": [],
                    },
                    {
                        "id": "model-b",
                        "active_requests": 1,
                        "waiting_requests": 3,
                        "prefilling": [],
                    },
                ],
            },
        }
        delegate, status_item, button = self._make_delegate(
            app_module,
            stats,
            app_module.ServerStatus.RUNNING,
            show_live_metrics=True,
            icons_available=False,
        )

        app_module.OMLXAppDelegate._update_menubar_icon(delegate)

        button.setImage_.assert_not_called()
        self._assert_displayed_title(status_item, button, "1k tok/s")

    def test_visible_live_badges_keep_a_shared_fixed_width_slot(self, app_module):
        """Visible live metrics should reserve the same width across badge states."""
        state_cases = [
            (
                {
                    "avg_generation_tps": 0.0,
                    "active_models": {
                        "total_active_requests": 1,
                        "total_waiting_requests": 0,
                        "models": [
                            {
                                "id": "prefill-model",
                                "active_requests": 1,
                                "waiting_requests": 0,
                                "prefilling": [
                                    {
                                        "request_id": "req-pp",
                                        "processed": 600,
                                        "total": 1000,
                                        "speed": 300.0,
                                        "eta": 1.3,
                                    }
                                ],
                            }
                        ],
                    },
                },
                "60%",
            ),
            (
                {
                    "avg_generation_tps": 1024.6,
                    "active_models": {
                        "total_active_requests": 2,
                        "total_waiting_requests": 1,
                        "models": [
                            {
                                "id": "gen-model",
                                "active_requests": 2,
                                "waiting_requests": 1,
                                "prefilling": [],
                            }
                        ],
                    },
                },
                "1k tok/s",
            ),
            (
                {
                    "avg_generation_tps": 0.0,
                    "active_models": {
                        "total_active_requests": 0,
                        "total_waiting_requests": 4,
                        "models": [
                            {
                                "id": "queue-model",
                                "active_requests": 0,
                                "waiting_requests": 4,
                                "prefilling": [],
                            }
                        ],
                    },
                },
                "Q4",
            ),
        ]

        widths = []
        minimum_width = self._minimum_live_metric_width("1k tok/s")
        for stats, expected_title in state_cases:
            delegate, status_item, button = self._make_delegate(
                app_module,
                stats,
                app_module.ServerStatus.RUNNING,
                show_live_metrics=True,
            )

            app_module.OMLXAppDelegate._update_menubar_icon(delegate)

            self._assert_final_icon(button, "filled-icon")
            self._assert_displayed_title(status_item, button, expected_title)
            widths.append(
                self._assert_fixed_width(
                    status_item,
                    app_module,
                    minimum_width=minimum_width,
                )
            )

        assert len(set(widths)) == 1


class TestRefreshIntervalControls:
    """Tests for the refresh interval controls in Preferences."""

    @staticmethod
    def _last_single_arg(mock_method):
        assert mock_method.call_args_list, "Expected method to be called at least once"
        args, kwargs = mock_method.call_args_list[-1]
        assert not kwargs
        assert len(args) == 1
        return args[0]

    def test_behavior_card_has_room_for_refresh_controls(self, preferences_module):
        """The refresh controls should remain inside the Behavior card."""
        label_y, buttons_y = preferences_module._behavior_refresh_row_positions()

        assert label_y >= 16
        assert buttons_y >= 14

    def test_refresh_interval_selection_stages_button_state_without_mutating_config(
        self, preferences_module
    ):
        """Selecting a refresh button should update the staged selection, not the live config."""
        button_1 = Mock()
        button_2 = Mock()
        button_5 = Mock()
        controller = types.SimpleNamespace(
            config=types.SimpleNamespace(stats_refresh_interval=5),
            refresh_interval_buttons=[
                (1, button_1),
                (2, button_2),
                (5, button_5),
            ],
        )
        controller._update_refresh_interval_button_styles = (
            preferences_module.PreferencesWindowController._update_refresh_interval_button_styles.__get__(
                controller
            )
        )
        handler = (
            preferences_module.PreferencesWindowController._on_refresh_interval_changed.__get__(
                controller
            )
        )

        handler(button_2)

        assert controller.config.stats_refresh_interval == 5
        button_1.setState_.assert_called_with(
            preferences_module.NSControlStateValueOff
        )
        button_2.setState_.assert_called_with(
            preferences_module.NSControlStateValueOn
        )
        button_5.setState_.assert_called_with(
            preferences_module.NSControlStateValueOff
        )
        assert getattr(controller, "_pending_stats_refresh_interval", 2) == 2

    def test_close_discards_unsaved_refresh_interval_selection(self, preferences_module):
        """Closing preferences should discard staged refresh-rate changes."""
        button_1 = Mock()
        button_2 = Mock()
        button_5 = Mock()
        window = Mock()
        controller = types.SimpleNamespace(
            config=types.SimpleNamespace(stats_refresh_interval=5),
            refresh_interval_buttons=[
                (1, button_1),
                (2, button_2),
                (5, button_5),
            ],
            window=window,
        )
        controller._update_refresh_interval_button_styles = (
            preferences_module.PreferencesWindowController._update_refresh_interval_button_styles.__get__(
                controller
            )
        )
        handler = (
            preferences_module.PreferencesWindowController._on_refresh_interval_changed.__get__(
                controller
            )
        )
        close_prefs = (
            preferences_module.PreferencesWindowController.closePrefs_.__get__(controller)
        )

        handler(button_2)
        close_prefs(None)

        assert controller.config.stats_refresh_interval == 5
        window.close.assert_called_once_with()

    def test_save_prefs_commits_staged_refresh_interval(self, preferences_module):
        """Saving preferences should persist the staged refresh interval."""
        base_path_label = Mock()
        base_path_label.stringValue.return_value = "/tmp/.omlx"
        model_dir_label = Mock()
        model_dir_label.stringValue.return_value = "/tmp/.omlx/models"
        port_field = Mock()
        port_field.stringValue.return_value = "8000"
        api_key_secure = Mock()
        api_key_secure.stringValue.return_value = ""
        api_key_plain = Mock()
        api_key_plain.stringValue.return_value = ""
        launch_at_login_checkbox = Mock()
        launch_at_login_checkbox.state.return_value = 0
        auto_start_checkbox = Mock()
        auto_start_checkbox.state.return_value = 0
        live_metrics_checkbox = Mock()
        live_metrics_checkbox.state.return_value = 0

        config = types.SimpleNamespace(
            base_path="/tmp/.omlx",
            port=8000,
            model_dir="",
            launch_at_login=False,
            start_server_on_launch=False,
            show_live_metrics_in_menu_bar=False,
            stats_refresh_interval=5,
            save=Mock(),
        )
        controller = types.SimpleNamespace(
            config=config,
            refresh_interval_buttons=[(2, Mock())],
            _pending_stats_refresh_interval=2,
            base_path_label=base_path_label,
            model_dir_label=model_dir_label,
            port_field=port_field,
            _api_key_visible=False,
            api_key_secure=api_key_secure,
            api_key_plain=api_key_plain,
            launch_at_login_checkbox=launch_at_login_checkbox,
            auto_start_checkbox=auto_start_checkbox,
            live_metrics_checkbox=live_metrics_checkbox,
            server_manager=types.SimpleNamespace(status=object()),
            _apply_launch_at_login=Mock(),
            on_save=Mock(),
            window=Mock(),
        )
        save_prefs = (
            preferences_module.PreferencesWindowController.savePrefs_.__get__(controller)
        )

        save_prefs(None)

        assert controller.config.stats_refresh_interval == 2
        controller.config.save.assert_called_once_with()
        controller.on_save.assert_called_once_with()
        controller.window.close.assert_called_once_with()

    def test_refresh_interval_styles_follow_selected_config(self, preferences_module):
        """The selected refresh interval should be the only active button style."""
        button_1 = Mock()
        button_2 = Mock()
        button_5 = Mock()
        controller = types.SimpleNamespace(
            config=types.SimpleNamespace(stats_refresh_interval=1),
            refresh_interval_buttons=[
                (1, button_1),
                (2, button_2),
                (5, button_5),
            ],
        )
        updater = (
            preferences_module.PreferencesWindowController._update_refresh_interval_button_styles.__get__(
                controller
            )
        )

        updater()

        button_1.setState_.assert_called_with(
            preferences_module.NSControlStateValueOn
        )
        button_2.setState_.assert_called_with(
            preferences_module.NSControlStateValueOff
        )
        button_5.setState_.assert_called_with(
            preferences_module.NSControlStateValueOff
        )

    def test_refresh_interval_styles_visibly_emphasize_selected_button(
        self, preferences_module
    ):
        """The selected refresh interval should get a distinct visual treatment."""
        button_1 = Mock()
        button_2 = Mock()
        button_5 = Mock()
        controller = types.SimpleNamespace(
            config=types.SimpleNamespace(stats_refresh_interval=2),
            refresh_interval_buttons=[
                (1, button_1),
                (2, button_2),
                (5, button_5),
            ],
        )
        updater = (
            preferences_module.PreferencesWindowController._update_refresh_interval_button_styles.__get__(
                controller
            )
        )

        updater()

        selected_font = self._last_single_arg(button_2.setFont_)
        unselected_font = self._last_single_arg(button_1.setFont_)
        assert selected_font.weight is not None and selected_font.weight > 0
        assert unselected_font.weight in (None, 0)
        assert (
            self._last_single_arg(button_2.setContentTintColor_)
            == preferences_module.NSColor.controlAccentColor()
        )
        assert (
            self._last_single_arg(button_1.setContentTintColor_)
            == preferences_module.NSColor.secondaryLabelColor()
        )


class TestRefreshIntervalLifecycle:
    """Tests for launch/save wiring of the refresh interval."""

    def test_launch_schedules_health_timer_with_configured_interval(self, app_module):
        """App launch should honor the configured refresh interval immediately."""
        timer = object()
        app_module.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.return_value = (
            timer
        )
        delegate = types.SimpleNamespace(
            config=types.SimpleNamespace(
                stats_refresh_interval=2,
                is_first_run=False,
                start_server_on_launch=False,
            ),
            server_manager=Mock(),
            status_item=None,
            health_timer=None,
            _icon_outline=None,
            _icon_filled=None,
            _load_menubar_icon=Mock(
                side_effect=["outline-icon", "filled-icon"]
            ),
            _update_menubar_icon=Mock(),
            _build_menu=Mock(),
            _check_for_updates=Mock(),
        )
        delegate._on_server_status_changed = (
            app_module.OMLXAppDelegate._on_server_status_changed.__get__(delegate)
        )

        app_module.OMLXAppDelegate._doFinishLaunching(delegate)

        assert delegate.health_timer is timer
        assert delegate._last_refresh_interval == 2
        app_module.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.assert_called_once_with(
            2.0, delegate, "healthCheck:", None, True
        )

    def test_launch_registers_status_callback_for_immediate_ui_refresh(
        self, app_module
    ):
        """Background status changes should not wait on the stats refresh timer."""
        timer = object()
        app_module.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_.return_value = (
            timer
        )
        server_manager = Mock()
        delegate = types.SimpleNamespace(
            config=types.SimpleNamespace(
                stats_refresh_interval=10,
                is_first_run=False,
                start_server_on_launch=False,
            ),
            server_manager=server_manager,
            status_item=None,
            health_timer=None,
            _icon_outline=None,
            _icon_filled=None,
            _load_menubar_icon=Mock(
                side_effect=["outline-icon", "filled-icon"]
            ),
            _update_menubar_icon=Mock(),
            _build_menu=Mock(),
            _check_for_updates=Mock(),
            _update_status_display=Mock(),
            performSelectorOnMainThread_withObject_waitUntilDone_=Mock(
                side_effect=lambda *args: delegate._update_status_display()
            ),
        )
        delegate._on_server_status_changed = (
            app_module.OMLXAppDelegate._on_server_status_changed.__get__(delegate)
        )

        app_module.OMLXAppDelegate._doFinishLaunching(delegate)

        server_manager.set_status_callback.assert_called_once()
        callback = server_manager.set_status_callback.call_args[0][0]
        assert callable(callback)

        callback(app_module.ServerStatus.RUNNING)

        delegate._update_status_display.assert_called()

    def test_prefs_save_restarts_timer_when_interval_changes(self, app_module):
        """Saving prefs should restart the timer when the refresh interval changes."""
        server_manager = Mock()
        delegate = types.SimpleNamespace(
            config=types.SimpleNamespace(stats_refresh_interval=2),
            server_manager=server_manager,
            _last_refresh_interval=5,
            _restart_health_timer=Mock(),
            _update_status_display=Mock(),
        )

        app_module.OMLXAppDelegate._on_prefs_saved(delegate)

        server_manager.update_config.assert_called_once_with(delegate.config)
        delegate._restart_health_timer.assert_called_once_with(2)
        assert delegate._last_refresh_interval == 2
        delegate._update_status_display.assert_called_once_with()

    def test_prefs_save_skips_timer_restart_when_interval_is_unchanged(
        self, app_module
    ):
        """Saving prefs should not restart the timer when the interval is unchanged."""
        server_manager = Mock()
        delegate = types.SimpleNamespace(
            config=types.SimpleNamespace(stats_refresh_interval=5),
            server_manager=server_manager,
            _last_refresh_interval=5,
            _restart_health_timer=Mock(),
            _update_status_display=Mock(),
        )

        app_module.OMLXAppDelegate._on_prefs_saved(delegate)

        server_manager.update_config.assert_called_once_with(delegate.config)
        delegate._restart_health_timer.assert_not_called()
        assert delegate._last_refresh_interval == 5
        delegate._update_status_display.assert_called_once_with()

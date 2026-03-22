"""oMLX Menubar App - macOS menubar application for oMLX server management."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_version() -> str:
    """Load oMLX version without importing the heavy `omlx` package."""
    version_file = Path(__file__).resolve().parents[2] / "omlx" / "_version.py"
    spec = spec_from_file_location("omlx_version_for_menubar", version_file)
    if spec is None or spec.loader is None:
        return "0.0.0"
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "__version__", "0.0.0")


__version__ = _load_version()

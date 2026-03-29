"""oMLX Menubar App - macOS menubar application for oMLX server management."""

import logging
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_version_from_file(version_file: Path) -> str | None:
    spec = spec_from_file_location("omlx_version_for_menubar", version_file)
    if spec is None or spec.loader is None:
        return None
    try:
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _load_version() -> str:
    """Load oMLX version without importing the heavy `omlx` package."""
    module_path = Path(__file__).resolve()
    candidates = []

    # Bundled app layout:
    #   .../Contents/Resources/omlx_app/__init__.py -> .../Contents/Resources/omlx/_version.py
    if len(module_path.parents) > 1:
        candidates.append(module_path.parents[1] / "omlx" / "_version.py")

    # Repo/dev layout:
    #   .../packaging/omlx_app/__init__.py -> .../omlx/_version.py
    if len(module_path.parents) > 2:
        candidates.append(module_path.parents[2] / "omlx" / "_version.py")

    for version_file in candidates:
        if not version_file.exists():
            continue
        version = _load_version_from_file(version_file)
        if isinstance(version, str) and version:
            return version

    logger.warning(
        "Falling back to default oMLX version 0.0.0; could not load any of: %s",
        ", ".join(str(path) for path in candidates) if candidates else "<no candidates>",
    )
    return "0.0.0"


__version__ = _load_version()

"""
Machine-specific settings in one place, so no path or personal value is hardcoded.

Lookup order for setting("name"):  env var BUDDY_NAME  >  config.yaml (repo root)  >  default.
Private files (profiles, tokens, candidate profile) live in ./config/ and are git-ignored;
copy the *.example.* files there and edit.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("BUDDY_CONFIG_DIR", REPO_ROOT / "config"))
_CONFIG_FILE = Path(os.environ.get("BUDDY_CONFIG_FILE", REPO_ROOT / "config.yaml"))
_cache: dict | None = None


def _file_config() -> dict:
    global _cache
    if _cache is None:
        _cache = {}
        if _CONFIG_FILE.exists():
            try:
                import yaml  # optional dependency
                _cache = yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8")) or {}
            except Exception:
                _cache = {}
    return _cache


def setting(name: str, default: str = "") -> str:
    env = os.environ.get("BUDDY_" + name.upper())
    if env not in (None, ""):
        return env
    value = _file_config().get(name.lower())
    return default if value in (None, "") else str(value)


def load_text(filename: str, default: str = "") -> str:
    """Private file from ./config/, else its *.example.* sibling, else `default`."""
    for candidate in (CONFIG_DIR / filename, CONFIG_DIR / filename.replace(".", ".example.", 1)):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return default


def load_json(filename: str, default):
    text = load_text(filename, "")
    try:
        return json.loads(text) if text else default
    except ValueError:
        return default

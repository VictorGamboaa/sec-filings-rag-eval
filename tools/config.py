"""Configuration loading -- Rule 3: tunables are configuration, never constants.

Concurrency, batch size, rate limit, chunk size and overlap must be able to vary
between runs *without editing code*. A YAML file alone only gets halfway there:
a scaling sweep would still mean editing the file between every run, which makes
the runs hard to reproduce and easy to mis-record. So overrides are applied at
invocation::

    python -m tools.config --print --set embed.batch_size=128 --set fetch.concurrency=8

The resolved values -- post-override -- are what ``RunLog`` writes into the run
log, so every measurement is traceable to the exact tunables that produced it.

Secrets are never stored here. The YAML holds the *name* of an environment
variable (e.g. ``fetch.user_agent_env: EDGAR_USER_AGENT``) and the value is read
from the environment via ``Config.secret()``, after python-dotenv loads ``.env``.

CLI::

    python -m tools.config --print
    python -m tools.config --print --format json
    python -m tools.config --check-env
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

__all__ = ["Config", "load_config", "ConfigError", "add_config_args"]

DEFAULT_CONFIG_PATH = "inputs/config.yaml"
_MISSING = object()


class ConfigError(RuntimeError):
    """Configuration is absent, malformed, or missing a required value."""


def _load_dotenv_once() -> None:
    """Load .env if python-dotenv is installed.

    Deliberately soft: the config module is useful (and self-testable) before
    dependencies are installed, and most keys are only needed by network stages.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _parse_scalar(raw: str) -> Any:
    """Parse a CLI override value into a typed scalar.

    Mirrors YAML scalar rules closely enough for override use: ``true``/``false``
    /``null``, ints, floats, JSON lists and objects, otherwise a string.
    """
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("null", "none", "~", ""):
        return None
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text[0] in "[{\"":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


class Config:
    """Dotted-path access over the nested config tree."""

    def __init__(self, data: dict[str, Any], *, source: Path | None = None) -> None:
        self._data = data
        self.source = source
        self.overrides: dict[str, Any] = {}

    def get(self, path: str, default: Any = _MISSING) -> Any:
        """Read a dotted path, e.g. ``config.get("embed.batch_size")``."""
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise ConfigError(
                        f"missing config key {path!r}"
                        + (f" in {self.source}" if self.source else "")
                    )
                return default
            node = node[part]
        return node

    def __getitem__(self, path: str) -> Any:
        return self.get(path)

    def __contains__(self, path: str) -> bool:
        return self.get(path, None) is not None

    def set(self, path: str, value: Any) -> None:
        """Write a dotted path, creating intermediate dicts as needed."""
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def apply_overrides(self, pairs: Iterable[str]) -> None:
        """Apply ``key.path=value`` strings from the command line."""
        for pair in pairs:
            if "=" not in pair:
                raise ConfigError(
                    f"malformed override {pair!r}; expected key.path=value"
                )
            key, _, raw = pair.partition("=")
            key = key.strip()
            value = _parse_scalar(raw)
            self.set(key, value)
            self.overrides[key] = value

    def secret(self, env_var_key: str, *, required: bool = True) -> str | None:
        """Resolve a secret by indirection.

        ``env_var_key`` is a dotted config path whose *value* is the name of an
        environment variable. Keeps credentials out of the config file entirely.
        """
        var_name = self.get(env_var_key)
        value = os.environ.get(var_name)
        if not value and required:
            raise ConfigError(
                f"environment variable {var_name!r} is not set "
                f"(required by config key {env_var_key!r}). "
                f"Copy .env.example to .env and fill it in."
            )
        return value or None

    def stage_dir(self, path: str, *, create: bool = False) -> Path:
        """Resolve a stage output directory.

        Rule 1: each stage persists its output to disk so it can be re-run
        without repeating the previous one. Stage directories come from config
        precisely so a re-run can be pointed at existing intermediates.
        """
        directory = Path(self.get(path))
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def as_dict(self) -> dict[str, Any]:
        """Deep copy of the resolved tree, for embedding in the run log."""
        import copy

        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config(source={self.source!r}, keys={sorted(self._data)})"


def load_config(
    path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
    *,
    overrides: Iterable[str] | None = None,
    load_env: bool = True,
) -> Config:
    """Load the YAML config and apply any command-line overrides."""
    if load_env:
        _load_dotenv_once()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ConfigError(
            "pyyaml is not installed. Run: uv sync  (or pip install pyyaml)"
        ) from exc

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")

    config = Config(data, source=config_path)
    if overrides:
        config.apply_overrides(overrides)
    return config


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the standard --config / --set flags to a stage's parser.

    Every stage should use this so the override syntax is identical everywhere.
    """
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"path to config YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set embed.batch_size=128. Repeatable.",
    )
    return parser


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.config",
        description="Inspect the resolved configuration.",
    )
    add_config_args(parser)
    parser.add_argument("--print", action="store_true", help="print resolved config")
    parser.add_argument(
        "--format", choices=("yaml", "json"), default="yaml", help="output format"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="report which *_env secrets are resolvable, without printing values",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, overrides=args.overrides)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not (args.print or args.check_env):
        parser.print_help()
        return 0

    if args.print:
        data = config.as_dict()
        if args.format == "json":
            print(json.dumps(data, indent=2, default=str))
        else:
            import yaml

            print(yaml.safe_dump(data, sort_keys=False, default_flow_style=False).rstrip())
        if config.overrides:
            print("\n--- overrides applied ---")
            for key, value in config.overrides.items():
                print(f"  {key} = {value!r}  ({type(value).__name__})")

    if args.check_env:
        print("\n--- secret resolution (values never printed) ---")
        missing = 0
        for key in _find_env_keys(config.as_dict()):
            var_name = config.get(key)
            present = bool(os.environ.get(var_name))
            missing += 0 if present else 1
            status = "set" if present else "MISSING"
            print(f"  [{status:>7}] {key} -> ${var_name}")
        if missing:
            print(f"\n{missing} secret(s) unset. Copy .env.example to .env.")
            return 1
        print("\nAll declared secrets resolve.")

    return 0


def _find_env_keys(node: Any, prefix: str = "") -> list[str]:
    """Collect dotted paths of keys ending in ``_env`` (secret indirections)."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, str) and str(key).endswith("_env"):
                found.append(path)
            else:
                found.extend(_find_env_keys(value, path))
    return found


if __name__ == "__main__":
    raise SystemExit(_main())

from __future__ import annotations

import difflib
import tomllib
from pathlib import Path
from typing import Any

_PATH_FIELDS = {"file", "output", "cookies", "kev_dir", "log_file"}
_BOOLEAN_FIELDS = {
    "musicbrainz",
    "skip_fingerprint",
    "force_fingerprint",
    "no_silence_check",
    "skip_existing",
    "update_json",
    "dry_run",
    "interactive",
    "select",
    "preview",
    "video_preview",
    "jev",
    "kev",
    "kev_skip_update",
    "retry",
    "review",
    "review_only_suspects",
    "verify",
    "repair",
}
_INTEGER_FIELDS = {
    "limit",
    "max_results",
    "max_duration",
    "min_duration",
    "fuzzy_threshold",
    "workers",
    "score_threshold",
    "kev_port",
    "kev_startup_timeout",
    "kev_runs",
    "jev_runs",
    "preview_seconds",
    "review_clip_seconds",
}
_FLOAT_FIELDS = {"kev_threshold", "jev_threshold"}
_STRING_FIELDS = {
    "data",
    "url",
    "format",
    "quality",
    "cookies_browser",
    "proxy",
    "fingerprint_mode",
    "kev_url",
    "kev_model",
    "kev_run",
    "match_title",
    "reject_title",
}


class ProfileError(ValueError):
    pass


def resolve_profiles_path(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return explicit_path
    local = Path.cwd() / "profiles.toml"
    if local.is_file():
        return local
    return Path.home() / ".config" / "ytdl-core" / "profiles.toml"


def load_profile(
    name: str,
    path: Path | None,
    allowed_keys: set[str],
) -> tuple[Path, dict[str, Any]]:
    profiles_path = resolve_profiles_path(path)
    try:
        with profiles_path.open("rb") as file:
            document = tomllib.load(file)
    except FileNotFoundError as error:
        searched: str | Path = profiles_path
        if path is None:
            searched = (
                f"{Path.cwd() / 'profiles.toml'} or "
                f"{Path.home() / '.config' / 'ytdl-core' / 'profiles.toml'}"
            )
        raise ProfileError(f"Profiles file not found: {searched}") from error
    except OSError as error:
        raise ProfileError(f"Could not read profiles file: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ProfileError(f"Invalid TOML in {profiles_path}: {error}") from error

    profiles = document.get("profiles")
    if not isinstance(profiles, dict):
        raise ProfileError(f"No [profiles] section found in {profiles_path}")

    raw_profile = profiles.get(name)
    if not isinstance(raw_profile, dict):
        available = ", ".join(sorted(str(key) for key in profiles)) or "none"
        raise ProfileError(f"Unknown profile '{name}'. Available profiles: {available}")

    settings: dict[str, Any] = {}
    for raw_key, value in raw_profile.items():
        key = str(raw_key).replace("-", "_")
        if key not in allowed_keys:
            suggestion = difflib.get_close_matches(key, sorted(allowed_keys), n=1)
            hint = f". Did you mean '{suggestion[0]}'?" if suggestion else ""
            raise ProfileError(f"Unknown option '{raw_key}' in profile '{name}'{hint}")
        settings[key] = _normalize_value(key, value, name)

    return profiles_path, settings


def _normalize_value(key: str, value: Any, profile_name: str) -> Any:
    if key in _PATH_FIELDS:
        if not isinstance(value, str) or not value.strip():
            raise ProfileError(f"Option '{key}' in profile '{profile_name}' must be a path")
        return Path(value)
    if key == "delay":
        if not isinstance(value, list) or len(value) != 2:
            raise ProfileError(f"Option 'delay' in profile '{profile_name}' must have two values")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            raise ProfileError(f"Option 'delay' in profile '{profile_name}' must be numeric")
        delay = [float(value[0]), float(value[1])]
        if delay[0] < 0 or delay[1] < delay[0]:
            raise ProfileError(f"Option 'delay' in profile '{profile_name}' is out of order")
        return delay
    if key == "report":
        formats = [value] if isinstance(value, str) else value
        if not isinstance(formats, list) or any(not isinstance(item, str) for item in formats):
            raise ProfileError(f"Option 'report' in profile '{profile_name}' must be a list")
        return formats
    if key == "sources":
        sources = value.split(",") if isinstance(value, str) else value
        if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
            raise ProfileError(f"Option 'sources' in profile '{profile_name}' must be a list")
        return [source.strip() for source in sources if source.strip()]
    if key in _BOOLEAN_FIELDS and not isinstance(value, bool):
        raise ProfileError(f"Option '{key}' in profile '{profile_name}' must be true or false")
    if key in _INTEGER_FIELDS and (isinstance(value, bool) or not isinstance(value, int)):
        raise ProfileError(f"Option '{key}' in profile '{profile_name}' must be an integer")
    if key in _FLOAT_FIELDS and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ProfileError(f"Option '{key}' in profile '{profile_name}' must be numeric")
    if key in _STRING_FIELDS and not isinstance(value, str):
        raise ProfileError(f"Option '{key}' in profile '{profile_name}' must be text")
    return value

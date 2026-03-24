from typing import Any


def resolve_named_value(config_value: Any, key: str, default_key: str = "main") -> float:
    if isinstance(config_value, dict):
        if key in config_value:
            return float(config_value[key])
        if default_key in config_value:
            return float(config_value[default_key])
        if len(config_value) == 1:
            return float(next(iter(config_value.values())))
        raise KeyError(f"missing key '{key}' in config value {config_value}")
    return float(config_value)

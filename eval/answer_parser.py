import re
from decimal import Decimal, InvalidOperation
from typing import Optional


_BOX_PATTERNS = (
    re.compile(r"\\box\{([^{}]+)\}"),
    re.compile(r"\\boxed\{([^{}]+)\}"),
)
_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_CHOICE_PATTERN_CANDIDATES = (
    re.compile(r"\\boxed\{\s*([A-Ja-j])\s*\}"),
    re.compile(r"\\box\{\s*([A-Ja-j])\s*\}"),
    re.compile(r"answer\s+is\s*[:\-]?\s*\(?\s*([A-Ja-j])\s*\)?", re.IGNORECASE),
    re.compile(r"correct\s+answer\s*(?:is|:)\s*\(?\s*([A-Ja-j])\s*\)?", re.IGNORECASE),
    re.compile(r"option\s*([A-Ja-j])\b", re.IGNORECASE),
    re.compile(r"choice\s*([A-Ja-j])\b", re.IGNORECASE),
    re.compile(r"\(([A-Ja-j])\)"),
)


def extract_last_boxed_value(text: str) -> Optional[str]:
    if not text:
        return None
    matches: list[str] = []
    for pattern in _BOX_PATTERNS:
        matches.extend(match.group(1) for match in pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].strip()


def _extract_last_number(text: str) -> Optional[str]:
    matches = _NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1]


def normalize_numeric_string(text: str) -> Optional[str]:
    if text is None:
        return None
    raw = text.strip()
    if not raw:
        return None
    raw = raw.replace("$", "").replace(" ", "")
    number = raw if _NUMBER_PATTERN.fullmatch(raw) else _extract_last_number(raw)
    if number is None:
        return None
    number = number.replace(",", "")
    try:
        decimal = Decimal(number)
    except InvalidOperation:
        return None
    normalized = format(decimal.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized == "-0":
        normalized = "0"
    return normalized


def normalize_answer_string(text: str) -> Optional[str]:
    if text is None:
        return None
    raw = text.strip()
    if not raw:
        return None
    raw = raw.strip("$")
    if raw.startswith("{") and raw.endswith("}") and len(raw) >= 2:
        raw = raw[1:-1].strip()
    numeric = normalize_numeric_string(raw)
    if numeric is not None:
        return numeric
    compact = re.sub(r"\s+", " ", raw).strip()
    if not compact:
        return None
    return compact.casefold()


def extract_normalized_boxed_answer(text: str) -> Optional[str]:
    boxed = extract_last_boxed_value(text)
    if boxed is None:
        return None
    return normalize_answer_string(boxed)


def normalize_choice_letter(text: str) -> Optional[str]:
    if text is None:
        return None
    raw = text.strip()
    if not raw:
        return None
    raw = raw.strip("$")
    if raw.startswith("{") and raw.endswith("}") and len(raw) >= 2:
        raw = raw[1:-1].strip()
    raw = raw.strip().strip("().:;,-_[]")
    if len(raw) == 1 and raw.upper() in "ABCDEFGHIJ":
        return raw.upper()
    return None


def extract_choice_letter_answer(text: str) -> Optional[str]:
    if not text:
        return None

    boxed = extract_last_boxed_value(text)
    normalized_boxed = normalize_choice_letter(boxed) if boxed is not None else None
    if normalized_boxed is not None:
        return normalized_boxed

    matches: list[str] = []
    for pattern in _CHOICE_PATTERN_CANDIDATES:
        matches.extend(match.group(1) for match in pattern.finditer(text))
    if matches:
        return normalize_choice_letter(matches[-1])

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        normalized = normalize_choice_letter(line)
        if normalized is not None:
            return normalized
    return None

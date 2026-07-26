from __future__ import annotations

import re
from dataclasses import dataclass

_QUOTED = re.compile(r"""["'“”‘’「」『』]([^"'“”‘’「」『』]+)["'“”‘’「」『』]""")
_DIALOGUE_HINT = re.compile(r"(대사|멘트|자막|음성|말하|말하는|라고|이라고)")
_REQUEST_SUFFIX = re.compile(
    r"\s*(?:을|를)?\s*"
    r"(?:찾아(?:\s*줘|주세요)?|검색해(?:\s*줘|주세요)?|"
    r"보여(?:\s*줘|주세요)?|찾고\s*싶(?:어|어요|습니다))"
    r"\s*[.!?]*$"
)
_GENERIC_SUFFIX = re.compile(r"\s*(?:장면|영상|컷|부분)\s*$")
_DIALOGUE_SUFFIXES = (
    re.compile(r"\s*(?:이?라는|라고|이라고)?\s*(?:대사|멘트|자막|음성)\s*$"),
    re.compile(r"\s*(?:라고|이라고)\s*말하(?:는|는\s*장면|는\s*부분)?\s*$"),
)


@dataclass(frozen=True)
class SearchIntent:
    original: str
    text_query: str
    visual_query: str | None
    dialogue_only: bool


def parse_search_intent(query: str) -> SearchIntent:
    original = " ".join(query.split()).strip()
    dialogue_only = bool(_DIALOGUE_HINT.search(original))
    quoted = _QUOTED.search(original)

    cleaned = _REQUEST_SUFFIX.sub("", original).strip()
    cleaned = _GENERIC_SUFFIX.sub("", cleaned).strip()

    if dialogue_only:
        if quoted:
            cleaned = quoted.group(1).strip()
        else:
            for pattern in _DIALOGUE_SUFFIXES:
                cleaned = pattern.sub("", cleaned).strip()

    cleaned = cleaned or original
    return SearchIntent(
        original=original,
        text_query=cleaned,
        visual_query=None if dialogue_only else cleaned,
        dialogue_only=dialogue_only,
    )

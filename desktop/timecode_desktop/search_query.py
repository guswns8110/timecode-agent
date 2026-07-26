from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SearchMode = Literal["auto", "scene", "dialogue"]

_QUOTED = re.compile(r"""["'“”‘’「」『』]([^"'“”‘’「」『』]+)["'“”‘’「」『』]""")
_DIALOGUE_HINT = re.compile(r"(대사|멘트|음성|말하|라고\s*(?:하|말))")
_SCREEN_TEXT_HINT = re.compile(
    r"(자막|화면\s*(?:글자|텍스트)|문구|글씨|타이틀|화면에)"
)
_SCREEN_TEXT_PATTERN = re.compile(
    r"(?:화면(?:에|의)?\s*)?"
    r"(?P<text>.+?)"
    r"(?:이?라는|라고)?\s*"
    r"(?:자막|문구|글자|글씨|텍스트|타이틀)"
    r"(?:이|가|을|를|은|는)?"
    r"(?:\s*(?:나오|보이|표시|등장).*)?$"
)
_REQUEST_SUFFIX = re.compile(
    r"\s*(?:을|를)?\s*"
    r"(?:찾아(?:\s*줘|주세요)?|검색해(?:\s*줘|주세요)?|"
    r"보여(?:\s*줘|주세요)?|찾고\s*싶(?:어|어요|습니다))"
    r"\s*[.!?]*$"
)
_GENERIC_SUFFIX = re.compile(r"\s+(?:장면|영상|컷|부분)\s*$")
_DIALOGUE_SUFFIXES = (
    re.compile(r"\s*(?:이?라는|라고|이라고)?\s*(?:대사|멘트|자막|음성)\s*$"),
    re.compile(r"\s*(?:라고|이라고)\s*말하(?:는|는\s*장면|는\s*부분)?\s*$"),
)

_NUMBER_WORDS = {
    "한": 1,
    "하나": 1,
    "한명": 1,
    "두": 2,
    "둘": 2,
    "두명": 2,
    "세": 3,
    "셋": 3,
    "세명": 3,
    "네": 4,
    "넷": 4,
    "네명": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
    "열": 10,
}
_ENGLISH_NUMBERS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}
_COUNT_TOKEN = (
    r"\d+|한명|두명|세명|네명|하나|둘|셋|넷|"
    r"한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|여러"
)

_OBJECTS = (
    (("남성", "남자", "남자들"), "남자", "a man", "men", "명"),
    (("여성", "여자", "여자들"), "여자", "a woman", "women", "명"),
    (("어린이", "아이", "아동"), "아이", "a child", "children", "명"),
    (("사람들", "사람", "인물"), "사람", "a person", "people", "명"),
    (("비행기", "항공기"), "비행기", "an airplane", "airplanes", "대"),
    (("자동차", "차량", "승용차"), "자동차", "a car", "cars", "대"),
    (("강아지", "반려견"), "강아지", "a dog", "dogs", "마리"),
    (("고양이",), "고양이", "a cat", "cats", "마리"),
)

_SCENE_TRANSLATIONS = (
    (re.compile(r"비행기[가-힣\s]{0,20}이륙"), "an airplane taking off"),
    (re.compile(r"비행기[가-힣\s]{0,20}착륙"), "an airplane landing"),
    (re.compile(r"사람(?:들)?[이]?\s*악수"), "people shaking hands"),
    (re.compile(r"회의(?:를)?\s*하"), "people having a meeting"),
    (re.compile(r"자동차[가이]?\s*달리"), "a car driving"),
    (re.compile(r"사람(?:들)?[이]?\s*달리"), "people running"),
    (re.compile(r"사람(?:들)?[이]?\s*걷"), "people walking"),
)
_SCENE_PARTS = (
    (re.compile(r"홍보\s*영상(?:\s*느낌)?"), "promotional commercial video style"),
    (re.compile(r"광고\s*영상(?:\s*느낌)?"), "commercial advertising style"),
    (re.compile(r"시네마틱|영화적"), "cinematic"),
    (re.compile(r"슬로우\s*모션|슬로모션|느린\s*화면"), "slow motion"),
    (re.compile(r"타임\s*랩스|타임랩스"), "time lapse"),
    (re.compile(r"다큐멘터리|다큐\s*느낌"), "documentary style"),
    (re.compile(r"감성적|감성적인"), "emotional"),
    (re.compile(r"웅장한|장엄한"), "epic"),
    (re.compile(r"긴장감(?:\s*있는)?"), "tense"),
    (re.compile(r"공포|무서운"), "horror"),
    (re.compile(r"로맨틱|낭만적"), "romantic"),
    (re.compile(r"밝은"), "bright"),
    (re.compile(r"어두운"), "dark"),
    (re.compile(r"따뜻한|웜톤"), "warm color grade"),
    (re.compile(r"차가운|푸른\s*톤|블루\s*톤"), "cool blue color grade"),
    (re.compile(r"드론\s*샷|항공\s*촬영"), "drone shot"),
    (re.compile(r"클로즈\s*업"), "close-up shot"),
    (re.compile(r"와이드\s*샷|풀\s*샷"), "wide shot"),
    (re.compile(r"사무실"), "office"),
    (re.compile(r"활주로"), "runway"),
    (re.compile(r"비행기|항공기"), "airplane"),
    (re.compile(r"자동차|차량"), "car"),
)


@dataclass(frozen=True)
class ObjectConstraint:
    display_name: str
    detector_label: str
    plural_label: str
    unit_name: str
    count: int | None = None
    comparison: Literal["any", "exact", "minimum", "maximum"] = "any"

    def matches(self, detected_count: int) -> bool:
        if self.count is None or self.comparison == "any":
            return detected_count >= 1
        if self.comparison == "minimum":
            return detected_count >= self.count
        if self.comparison == "maximum":
            return 1 <= detected_count <= self.count
        return detected_count == self.count

    def english_prompts(self) -> list[str]:
        if self.count is None:
            return [
                f"a photo containing {self.plural_label}",
                self.detector_label,
            ]
        number = _ENGLISH_NUMBERS.get(self.count, str(self.count))
        if self.comparison == "minimum":
            phrase = f"at least {number} {self.plural_label}"
        elif self.comparison == "maximum":
            phrase = f"at most {number} {self.plural_label}"
        else:
            phrase = f"exactly {number} {self.plural_label}"
        return [f"a photo of {phrase}", phrase]

    def evidence(self, detected_count: int) -> str:
        return f"{self.display_name} {detected_count}{self.unit_name} 감지"


@dataclass(frozen=True)
class SearchIntent:
    original: str
    text_query: str
    visual_query: str | None
    dialogue_only: bool
    mode: SearchMode
    object_constraint: ObjectConstraint | None = None
    object_constraints: tuple[ObjectConstraint, ...] = ()
    screen_text_query: str | None = None
    visual_variants: tuple[str, ...] = ()
    temporal_variants: tuple[str, ...] = ()


def _parse_count(token: str | None) -> int | None:
    if not token:
        return None
    compact = token.replace(" ", "")
    if compact.isdigit():
        return int(compact)
    if compact == "여러":
        return 2
    return _NUMBER_WORDS.get(compact)


def _object_constraints(query: str) -> tuple[ObjectConstraint, ...]:
    constraints: list[ObjectConstraint] = []
    for aliases, display_name, detector_label, plural_label, unit_name in _OBJECTS:
        alias_pattern = "|".join(
            re.escape(alias)
            for alias in sorted(aliases, key=len, reverse=True)
        )
        after = re.search(
            rf"(?:{alias_pattern})(?:이|가|은|는)?\s*"
            rf"(?P<count>{_COUNT_TOKEN})?\s*(?:명|마리|대|개)?",
            query,
        )
        before = re.search(
            rf"(?P<count>{_COUNT_TOKEN})\s*(?:명|마리|대|개)?(?:의)?\s*"
            rf"(?:{alias_pattern})",
            query,
        )
        match = before or after
        if not match:
            continue
        token = match.groupdict().get("count")
        count = _parse_count(token)
        context = query[match.start() : match.end() + 5]
        if token and token.replace(" ", "") == "여러":
            comparison = "minimum"
        elif re.search(r"(이상|최소|넘는|보다\s*많)", context):
            comparison = "minimum"
        elif re.search(r"(이하|최대|보다\s*적)", context):
            comparison = "maximum"
        elif count is not None:
            comparison = "exact"
        else:
            comparison = "any"
        constraints.append(
            ObjectConstraint(
                display_name=display_name,
                detector_label=detector_label,
                plural_label=plural_label,
                unit_name=unit_name,
                count=count,
                comparison=comparison,
            )
        )
    return tuple(constraints)


def _screen_text_query(
    original: str,
    cleaned: str,
) -> str | None:
    if not _SCREEN_TEXT_HINT.search(original):
        return None
    quoted = _QUOTED.search(original)
    if quoted:
        return quoted.group(1).strip()
    match = _SCREEN_TEXT_PATTERN.search(cleaned)
    if not match:
        return None
    value = match.group("text").strip(" \t\"'“”‘’「」『』")
    value = re.sub(r"^(?:화면(?:에|의)?\s*)", "", value).strip()
    if not value or value in {"자막", "문구", "글자", "텍스트"}:
        return None
    return value


def _screen_visual_residual(
    cleaned: str,
    screen_text: str,
) -> str | None:
    residual = _QUOTED.sub(" ", cleaned)
    residual = residual.replace(screen_text, " ")
    residual = re.sub(
        r"(?:화면(?:에|의)?|자막|문구|글자|글씨|텍스트|타이틀|"
        r"이?라는|라고|"
        r"(?:이|가|은|는|을|를)?\s*"
        r"(?:나오(?:는|고)?|보이(?:는|고)?|표시(?:되는|되고)?|"
        r"등장(?:하는|하고)?))",
        " ",
        residual,
    )
    residual = re.sub(r"[^0-9A-Za-z가-힣]+", " ", residual)
    words = [
        word
        for word in residual.split()
        if len(word) > 1 and word not in {"장면", "영상", "부분", "컷"}
    ]
    return " ".join(words) or None


def _visual_variants(
    cleaned: str,
    constraints: tuple[ObjectConstraint, ...],
) -> tuple[str, ...]:
    variants = [cleaned]
    for constraint in constraints:
        variants.extend(constraint.english_prompts())
    for pattern, translated in _SCENE_TRANSLATIONS:
        if pattern.search(cleaned):
            variants.append(translated)
    translated = cleaned
    changed = False
    for pattern, replacement in (*_SCENE_TRANSLATIONS, *_SCENE_PARTS):
        translated, count = pattern.subn(f" {replacement} ", translated)
        changed = changed or count > 0
    if changed:
        translated = re.sub(r"[가-힣]+", " ", translated)
        translated = re.sub(r"\s+", " ", translated).strip()
        translated = translated.replace(
            "cool blue color grade cool blue color grade",
            "cool blue color grade",
        )
        if translated:
            variants.append(translated)
    return tuple(dict.fromkeys(value for value in variants if value))


def _temporal_variants(
    visual_variants: tuple[str, ...],
    constraints: tuple[ObjectConstraint, ...],
) -> tuple[str, ...]:
    object_prompts = {
        prompt
        for constraint in constraints
        for prompt in constraint.english_prompts()
    }
    translated = [
        value
        for value in visual_variants
        if not re.search(r"[가-힣]", value)
        and value not in object_prompts
    ]
    if not translated:
        return ()
    return (max(translated, key=lambda value: len(value.split())),)


def parse_search_intent(
    query: str,
    mode: SearchMode = "auto",
) -> SearchIntent:
    if mode not in ("auto", "scene", "dialogue"):
        raise ValueError(f"지원하지 않는 검색 모드입니다: {mode}")

    original = " ".join(query.split()).strip()
    dialogue_hint = bool(_DIALOGUE_HINT.search(original))
    dialogue_only = mode == "dialogue" or (
        mode == "auto" and dialogue_hint
    )
    quoted = _QUOTED.search(original)

    cleaned = _REQUEST_SUFFIX.sub("", original).strip()
    cleaned = _GENERIC_SUFFIX.sub("", cleaned).strip()
    screen_text_query = _screen_text_query(original, cleaned)

    if dialogue_only:
        if quoted:
            cleaned = quoted.group(1).strip()
        else:
            for pattern in _DIALOGUE_SUFFIXES:
                cleaned = pattern.sub("", cleaned).strip()

    cleaned = cleaned or original
    scene_query = (
        _screen_visual_residual(cleaned, screen_text_query)
        if screen_text_query is not None
        else cleaned
    )
    constraints = (
        ()
        if dialogue_only
        else _object_constraints(scene_query or cleaned)
    )
    constraint = constraints[0] if constraints else None
    visual_query = None if dialogue_only else scene_query
    visual_variants = (
        ()
        if visual_query is None
        else _visual_variants(visual_query, constraints)
    )
    return SearchIntent(
        original=original,
        text_query=cleaned,
        visual_query=visual_query,
        dialogue_only=dialogue_only,
        mode=mode,
        object_constraint=constraint,
        object_constraints=constraints,
        screen_text_query=screen_text_query,
        visual_variants=visual_variants,
        temporal_variants=_temporal_variants(
            visual_variants,
            constraints,
        ),
    )

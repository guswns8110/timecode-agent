from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SearchMode = Literal["auto", "scene", "dialogue"]
PersonKind = Literal["man", "woman", "child", "person"]

_QUOTED = re.compile(r"""["'“”‘’「」『』]([^"'“”‘’「」『』]+)["'“”‘’「」『』]""")
_DIALOGUE_HINT = re.compile(r"(대사|멘트|음성|말하|라고\s*(?:하|말))")
_SCREEN_TEXT_HINT = re.compile(
    r"(자막|화면\s*(?:글자|텍스트)|문구|글씨|타이틀|화면에)"
)
_AUDIO_SEARCH_HINT = re.compile(
    r"(분위기|감성|몽환|꿈같|긴장|서스펜스|웅장|장엄|공포|무서|"
    r"로맨틱|평화|고요|잔잔|희망|우울|쓸쓸|외로운|"
    r"슬픈|즐거|행복|활기|음악|음향|효과음|소리)"
)
_TEMPORAL_SEARCH_HINT = re.compile(
    r"(슬로우?\s*모션|슬로모션|타임\s*랩스|이륙|착륙|"
    r"걷|달리|뛰|날아|싸우|악수|움직|춤|운전|폭발|"
    r"팬(?:닝)?|틸트|줌|트래킹|핸드\s*헬드|흔들|"
    r"빠른|역동|격렬|긴장)"
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
    (("남성", "남자", "남자들"), "남자", "a man", "men", "명", "man"),
    (("여성", "여자", "여자들"), "여자", "a woman", "women", "명", "woman"),
    (("어린이", "아이", "아동"), "아이", "a child", "children", "명", "child"),
    (("사람들", "사람", "인물"), "사람", "a person", "people", "명", "person"),
    (("비행기", "항공기"), "비행기", "an airplane", "airplanes", "대", None),
    (("자동차", "차량", "승용차"), "자동차", "a car", "cars", "대", None),
    (("강아지", "반려견"), "강아지", "a dog", "dogs", "마리", None),
    (("고양이",), "고양이", "a cat", "cats", "마리", None),
    (("방패",), "방패", "a shield", "shields", "개", None),
    (
        ("괴물", "몬스터", "괴수"),
        "괴물",
        "a monster creature",
        "monsters",
        "마리",
        None,
    ),
    (("검", "칼", "장검"), "검", "a sword", "swords", "자루", None),
    (("총", "총기"), "총", "a gun", "guns", "정", None),
    (("노트북", "랩톱"), "노트북", "a laptop", "laptops", "대", None),
    (("휴대폰", "스마트폰"), "휴대폰", "a smartphone", "smartphones", "대", None),
    (("기차", "열차"), "기차", "a train", "trains", "대", None),
    (("배", "선박"), "선박", "a ship", "ships", "척", None),
    (("자전거",), "자전거", "a bicycle", "bicycles", "대", None),
    (("오토바이", "바이크"), "오토바이", "a motorcycle", "motorcycles", "대", None),
    (("로봇",), "로봇", "a robot", "robots", "대", None),
)

_GENERIC_OBJECT_WORDS = {
    "장면",
    "영상",
    "부분",
    "컷",
    "화면",
    "느낌",
    "분위기",
    "검색",
    "결과",
    "모습",
    "스타일",
}
_ABSTRACT_OBJECT_WORDS = {
    "감성",
    "감성적",
    "긴장감",
    "긴장감있는",
    "고급",
    "고급스러운",
    "몽환적",
    "시네마틱",
    "영화적",
    "웅장한",
    "장엄한",
    "따뜻한",
    "차가운",
    "밝은",
    "어두운",
    "로맨틱",
    "낭만적",
    "공포",
    "액션",
    "역동적",
    "차분한",
    "잔잔한",
    "슬로우모션",
}
_ACTION_OBJECT_STEMS = {
    "걷",
    "달리",
    "뛰",
    "날아가",
    "날아오",
    "싸우",
    "악수하",
    "회의하",
    "이륙하",
    "착륙하",
    "바라보",
    "쳐다보",
    "웃",
    "울",
    "말하",
    "앉",
    "서",
    "움직이",
    "춤추",
    "요리하",
    "운전하",
}
_KNOWN_OBJECT_ALIASES = {
    alias
    for aliases, *_ in _OBJECTS
    for alias in aliases
}

_ATTRIBUTE_HINTS = (
    (re.compile(r"밝(?:은|고)|화사한|하이\s*키"), (("brightness", 1.0),)),
    (re.compile(r"어둡|어두운|암전|로우\s*키|야간"), (("brightness", -1.0),)),
    (re.compile(r"고대비|강한\s*명암"), (("contrast", 1.0),)),
    (re.compile(r"저대비|부드러운\s*명암"), (("contrast", -1.0),)),
    (re.compile(r"선명한|화려한|비비드"), (("saturation", 1.0),)),
    (re.compile(r"저채도|무채색|차분한|파스텔"), (("saturation", -1.0),)),
    (re.compile(r"따뜻|웜\s*톤|노을"), (("warmth", 1.0),)),
    (re.compile(r"차갑|차가운|쿨\s*톤|푸른\s*톤|블루\s*톤"), (("warmth", -1.0),)),
    (
        re.compile(r"역동적|빠른|격렬한|액션|긴장감"),
        (("motion", 1.0), ("contrast", 0.45)),
    ),
    (
        re.compile(r"정적|고요한|잔잔한|안정적"),
        (("motion", -1.0),),
    ),
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
    (re.compile(r"브랜드\s*필름|브랜디드"), "premium branded film style"),
    (re.compile(r"기업\s*홍보|코퍼레이트"), "corporate promotional film"),
    (re.compile(r"뮤직\s*비디오|뮤비"), "music video style"),
    (re.compile(r"뉴스\s*느낌|보도\s*영상"), "television news footage"),
    (re.compile(r"인터뷰\s*느낌"), "professional interview footage"),
    (re.compile(r"브이로그|VLOG"), "vlog video style"),
    (re.compile(r"시네마틱|영화적"), "cinematic"),
    (re.compile(r"슬로우\s*모션|슬로모션|느린\s*화면"), "slow motion"),
    (re.compile(r"타임\s*랩스|타임랩스"), "time lapse"),
    (re.compile(r"다큐멘터리|다큐\s*느낌"), "documentary style"),
    (re.compile(r"SF|공상\s*과학"), "science fiction"),
    (re.compile(r"판타지"), "fantasy"),
    (re.compile(r"액션"), "action"),
    (re.compile(r"코미디"), "comedy"),
    (re.compile(r"드라마"), "dramatic"),
    (re.compile(r"감성적|감성적인|감성\s*있는"), "emotional"),
    (re.compile(r"몽환적|꿈같은"), "dreamy ethereal"),
    (re.compile(r"웅장한|장엄한"), "epic"),
    (re.compile(r"긴장감(?:\s*있는)?"), "tense"),
    (re.compile(r"서스펜스|불안한"), "suspenseful uneasy"),
    (re.compile(r"공포|무서운"), "horror"),
    (re.compile(r"로맨틱|낭만적"), "romantic"),
    (re.compile(r"평화로운|고요한|잔잔한"), "peaceful calm"),
    (re.compile(r"희망적|희망찬"), "hopeful uplifting"),
    (re.compile(r"우울한|쓸쓸한|외로운"), "melancholic lonely"),
    (re.compile(r"즐거운|행복한|활기찬"), "joyful energetic"),
    (re.compile(r"고급스러운|하이엔드|럭셔리"), "luxury high-end"),
    (re.compile(r"미니멀|절제된"), "minimal refined"),
    (re.compile(r"깔끔한|클린한"), "clean polished"),
    (re.compile(r"빈티지"), "vintage"),
    (re.compile(r"레트로"), "retro"),
    (re.compile(r"미래적|퓨처리스틱"), "futuristic"),
    (re.compile(r"밝은"), "bright"),
    (re.compile(r"어두운"), "dark"),
    (re.compile(r"따뜻한|웜톤"), "warm color grade"),
    (
        re.compile(r"차갑(?:고|게|은)|차가운|푸른\s*톤|블루\s*톤"),
        "cool blue color grade",
    ),
    (re.compile(r"저채도|무채색"), "desaturated muted colors"),
    (re.compile(r"고채도|비비드|화려한\s*색"), "vivid saturated colors"),
    (re.compile(r"고대비|강한\s*명암"), "high contrast lighting"),
    (re.compile(r"역광"), "backlit"),
    (re.compile(r"실루엣"), "silhouette"),
    (re.compile(r"네온"), "neon lighting"),
    (re.compile(r"자연광"), "natural light"),
    (re.compile(r"드론\s*샷|항공\s*촬영"), "drone shot"),
    (re.compile(r"익스트림\s*클로즈\s*업|초근접"), "extreme close-up shot"),
    (re.compile(r"클로즈\s*업"), "close-up shot"),
    (re.compile(r"와이드\s*샷|풀\s*샷"), "wide shot"),
    (re.compile(r"로우\s*앵글"), "low angle shot"),
    (re.compile(r"하이\s*앵글|부감"), "high angle shot"),
    (re.compile(r"POV|1인칭|시점\s*샷"), "point of view shot"),
    (re.compile(r"오버\s*숄더"), "over the shoulder shot"),
    (re.compile(r"매크로|접사"), "macro shot"),
    (re.compile(r"핸드\s*헬드|흔들리는\s*카메라"), "handheld camera"),
    (re.compile(r"고정\s*샷|고정된\s*카메라"), "static locked camera"),
    (re.compile(r"트래킹\s*샷|따라가는\s*카메라"), "tracking shot"),
    (re.compile(r"팬(?:닝)?\s*샷"), "camera panning"),
    (re.compile(r"틸트\s*샷"), "camera tilting"),
    (re.compile(r"줌\s*(?:인|아웃)"), "camera zoom"),
    (re.compile(r"아웃\s*포커스|보케"), "shallow depth of field bokeh"),
    (re.compile(r"비\s*오는|빗속"), "rainy weather"),
    (re.compile(r"눈\s*오는|설경"), "snowy weather"),
    (re.compile(r"일몰|노을"), "sunset golden hour"),
    (re.compile(r"야경|밤\s*도시"), "city at night"),
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
    person_kind: PersonKind | None = None
    excluded: bool = False

    def matches(self, detected_count: int) -> bool:
        if self.excluded:
            return detected_count == 0
        if self.count is None or self.comparison == "any":
            return detected_count >= 1
        if self.comparison == "minimum":
            return detected_count >= self.count
        if self.comparison == "maximum":
            return 1 <= detected_count <= self.count
        return detected_count == self.count

    def english_prompts(self) -> list[str]:
        if self.excluded:
            return []
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
        if self.excluded:
            return f"{self.display_name} 없음 확인"
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
    open_object_terms: tuple[str, ...] = ()
    screen_text_query: str | None = None
    no_screen_text: bool = False
    facets: tuple[str, ...] = ()
    negative_visual_terms: tuple[str, ...] = ()
    attribute_preferences: tuple[tuple[str, float], ...] = ()
    use_audio: bool = False
    use_temporal: bool = False
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
    for (
        aliases,
        display_name,
        detector_label,
        plural_label,
        unit_name,
        person_kind,
    ) in _OBJECTS:
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
        context = query[max(0, match.start() - 5) : match.end() + 8]
        excluded = bool(
            re.search(
                r"(제외|없는|없이|나오지\s*않|안\s*나오)",
                context,
            )
        )
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
                person_kind=person_kind,
                excluded=excluded,
            )
        )
    return tuple(constraints)


def _open_object_terms(
    query: str,
    constraints: tuple[ObjectConstraint, ...],
) -> tuple[str, ...]:
    known = {constraint.display_name for constraint in constraints}
    candidates = re.findall(
        r"([가-힣A-Za-z0-9]{2,}?)(?:이|가|을|를|은|는|와|과|도)"
        r"(?=\s|$)",
        query,
    )
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", query)
    if len(tokens) <= 3 and tokens:
        candidates.append(
            re.sub(
                r"(?:이|가|을|를|은|는|와|과|도)$",
                "",
                tokens[0],
            )
        )
    output: list[str] = []
    for value in candidates:
        value = value.strip()
        if (
            len(value) < 2
            or bool(re.fullmatch(r"\d+(?:명|개|대|마리)?", value))
            or value in known
            or value in _KNOWN_OBJECT_ALIASES
            or value in _GENERIC_OBJECT_WORDS
            or value in _ABSTRACT_OBJECT_WORDS
            or value.endswith(("적인", "스러운", "로운", "있는"))
            or value.endswith(
                (
                    "하",
                    "되",
                    "나오",
                    "보이",
                    "등장하",
                    "이륙하",
                    "착륙하",
                    "찾아",
                )
            )
            or value in _ACTION_OBJECT_STEMS
        ):
            continue
        if value not in output:
            output.append(value)
    return tuple(output[:4])


def _query_facets(query: str) -> tuple[str, ...]:
    facets = [
        value.strip()
        for value in re.split(
            r"\s*(?:\+|,|/|그리고|동시에|하면서|하며)\s*",
            query,
        )
        if len(value.strip()) >= 2
    ]
    for pattern, translated in (*_SCENE_TRANSLATIONS, *_SCENE_PARTS):
        if pattern.search(query):
            facets.append(translated)
    return tuple(dict.fromkeys(facets))


def _negative_visual_terms(query: str) -> tuple[str, ...]:
    terms = [
        match.group("term").strip()
        for match in re.finditer(
            r"(?P<term>[가-힣A-Za-z0-9]{2,}(?:\s+[가-힣A-Za-z0-9]{2,}){0,2})"
            r"(?:은|는|이|가)?\s*(?:제외|없이)",
            query,
        )
    ]
    return tuple(dict.fromkeys(terms))


def _attribute_preferences(query: str) -> tuple[tuple[str, float], ...]:
    values: dict[str, float] = {}
    for pattern, hints in _ATTRIBUTE_HINTS:
        if not pattern.search(query):
            continue
        for name, direction in hints:
            previous = values.get(name)
            if previous is None or abs(direction) > abs(previous):
                values[name] = direction
    return tuple(values.items())


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
    no_screen_text = bool(
        re.search(
            r"(?:자막|문구|글자|글씨|텍스트)(?:이|가)?\s*(?:없는|없이|제외)",
            original,
        )
    )

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
        open_object_terms=_open_object_terms(
            scene_query or cleaned,
            constraints,
        ),
        screen_text_query=screen_text_query,
        no_screen_text=no_screen_text,
        facets=_query_facets(scene_query or cleaned),
        negative_visual_terms=_negative_visual_terms(original),
        attribute_preferences=_attribute_preferences(
            scene_query or cleaned
        ),
        use_audio=bool(_AUDIO_SEARCH_HINT.search(scene_query or cleaned)),
        use_temporal=bool(
            _TEMPORAL_SEARCH_HINT.search(scene_query or cleaned)
        ),
        visual_variants=visual_variants,
        temporal_variants=_temporal_variants(
            visual_variants,
            constraints,
        ),
    )

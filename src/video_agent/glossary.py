"""Domain-term glossary from corrections.jsonl -> whisper hotwords.

The cheapest accuracy lever: corrections accumulated by the agent loop
(OCR-evidenced ASR fixes) yield the proper nouns / domain terms whisper
keeps missing; injecting them as `hotwords` needs no fine-tuning at all.
Fine-tuning only becomes worth it for errors this lever cannot fix.

The lever cuts both ways, so harvesting is gated on precision. Measured
2026-07-25 on the 31-record corpus, the naive "tokens the correction added"
rule scored **0.353** precision: particle residue (a proper noun still
wearing its comparison particle), conjugated verb forms and plain everyday
vocabulary outnumbered the real domain terms 22 to 12. That is not cosmetic
— an oversized, noisy hotword list poisoned an unrelated clip on
2026-07-22 (a silent clip: the
glossary hallucinated onto footage with no speech at all), and it matches
the external finding that a decoder prompt induces hallucination in
proportion to how much irrelevant vocabulary it carries.

Two gates keep the list clean, each measured independently (0.353 -> 0.448
precision, recall 0.857 -> 0.929):

1. **Particle stripping** — a suffix list, longest-first, applied only while
   at least two characters of stem survive, so a two-syllable name whose
   final syllable doubles as a particle keeps that syllable.
2. **Verb-form drop** — conjugated endings are dropped outright rather than
   stripped; chopping a conjugated verb down to a stem yields garbage, and
   no proper noun carries a verb ending.

A third gate was built and **rejected on measurement**: dropping any term
that another workspace's transcript already contains verbatim, on the theory
that whisper transcribes it correctly unaided. It reads as sound and it is
circular. The transcripts it consults are themselves hotword output, so the
filter deletes precisely the terms the lever already succeeded on — measured
2026-07-25 it cost four domain terms to remove four noise terms, dropping
recall 0.929 -> 0.643 for no precision gain. Two of those four were
vindicated by the worst possible source: they counted as "already known"
only because the 2026-07-22 hallucination had printed them onto a silent
clip. Re-evaluating needs transcripts whose
config stamp proves they were produced *without* hotwords; the stamp landed
2026-07-24, so the corpus cannot answer this yet.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .cache_paths import cache_root
from .fsio import write_text_atomic

SCHEMA = "video-agent-glossary/v2"
LEGACY_SOURCE = "legacy"  # 평문 캐시에서 넘어온 자리표시자 — 워크스페이스가 아니다
GLOSSARY_PATH = cache_root() / "glossary.json"
LEGACY_GLOSSARY_PATH = cache_root() / "glossary.txt"

# 조사 — 긴 것부터 시도한다. 어간이 2글자 미만이 되면 제거하지 않는다.
# 선택조사 "나"는 의도적으로 뺐다: "바나나"→"바나"처럼 고유명사를 깎을 위험이
# 이 조사 하나가 회수하는 용어 수보다 크다(실측 코퍼스 기준 회수 1건).
_PARTICLES = (
    "에서는", "으로는", "이라고", "에게서", "한테서",
    "에서", "으로", "이랑", "처럼", "부터", "까지", "마다", "보다",
    "한테", "에게", "께서", "이나", "밖에", "조차", "라도", "이란", "라는",
    "을", "를", "이", "가", "은", "는", "의", "에", "와", "과", "로", "도", "만", "랑",
)
# 용언 활용 어미 — 어간을 남기려 잘라내면 쓰레기가 되므로 토큰째 버린다.
_VERB_ENDINGS = (
    "하라고", "라고", "는데", "는지", "야지", "니야", "었어", "았어", "겠어",
    "드라", "더라", "네요", "군요", "잖아", "거야", "해서", "니까", "면서",
    "지만", "구나", "습니다", "입니다", "합니다", "세요", "어요", "아요",
)
_MIN_STEM = 2
_WORD = re.compile(r"[0-9A-Za-z가-힣]+$")


def _norm(token: str) -> str:
    """Strip one trailing particle, longest match first, stem-length capped."""
    for particle in _PARTICLES:
        if token.endswith(particle) and len(token) - len(particle) >= _MIN_STEM:
            return token[: -len(particle)]
    return token


def _is_verb_form(token: str) -> bool:
    return token.endswith(_VERB_ENDINGS)


def extract_terms(asr: str, corrected: str) -> set[str]:
    """Tokens present in the corrected text but not in the ASR text.

    Annotation records — corrected starting with "(" (e.g. "(발화 아님 — …)")
    — are judgments about the audio, not transcript fixes: skip entirely,
    or their narrative words become hotwords and poison every next ingest
    (measured 2026-07-22: "환각·전사·hotwords" leaked in as glossary terms).
    Tokens with punctuation (separator artifacts like "BGM·고함") are
    dropped for the same reason, as are conjugated verb forms.
    """
    if corrected.lstrip().startswith("("):
        return set()
    asr_tokens = {_norm(t) for t in asr.split()}
    terms = set()
    for token in corrected.split():
        t = _norm(token)
        if len(t) < 2 or not _WORD.match(t) or t in asr_tokens:
            continue
        if _is_verb_form(t):
            continue
        terms.add(t)
    return terms


def _harvest(ws_dirs: list[Path]) -> dict[str, dict]:
    """term -> {sources, records} across every workspace's corrections."""
    per_ws: dict[str, dict[str, int]] = {}
    for ws in ws_dirs:
        corr = Path(ws) / "corrections.jsonl"
        if not corr.is_file():
            continue
        counts: dict[str, int] = {}
        for line in corr.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                found = extract_terms(obj["asr"], obj["corrected"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            for term in found:
                counts[term] = counts.get(term, 0) + 1
        if counts:
            per_ws[str(ws)] = counts

    harvested: dict[str, dict] = {}
    for ws, counts in per_ws.items():
        name = Path(ws).name
        for term, n in counts.items():
            entry = harvested.setdefault(term, {"sources": {}})
            entry["sources"][name] = entry["sources"].get(name, 0) + n
    return harvested


def _migrate_flat(text: str) -> dict[str, dict]:
    """Adopt a flat word-list glossary, re-running it through the gates.

    The legacy file is the old rule's output, so it carries particle residue
    and verb forms. Copying it over verbatim would apply the improved rule to
    new terms only and leave the existing contamination in place forever.
    """
    migrated: dict[str, dict] = {}
    for raw in text.split():
        term = _norm(raw)
        if len(term) >= 2 and _WORD.match(term) and not _is_verb_form(term):
            migrated[term] = {"sources": {LEGACY_SOURCE: 1}}
    return migrated


def _normalize_entry(entry: object) -> dict:
    """Coerce a stored entry to {"sources": {workspace: count}}.

    An earlier draft kept sources as a flat list plus a separate total. Both
    shapes are read so an existing cache keeps working; the map is the single
    source of truth going forward — the total is derived, never stored, so
    the two cannot drift apart.
    """
    if not isinstance(entry, dict):
        return {"sources": {}}
    sources = entry.get("sources")
    if isinstance(sources, dict):
        return {"sources": {str(k): int(v) for k, v in sources.items()}}
    if isinstance(sources, list):
        return {"sources": {str(name): 1 for name in sources}}
    return {"sources": {}}


def _load_store(path: Path) -> dict[str, dict]:
    """Read the JSON store; fall back to a flat word list either way.

    The flat form shows up two ways — the pre-v2 cache sitting next to the
    JSON path, and a GLOSSARY_PATH pointed straight at a .txt file — so the
    fallback keys off the content, not the filename.
    """
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return _migrate_flat(text)
        terms = data.get("terms") if isinstance(data, dict) else None
        if not isinstance(terms, dict):
            return {}
        return {term: _normalize_entry(entry) for term, entry in terms.items()}
    legacy = path.parent / "glossary.txt"
    return (
        _migrate_flat(legacy.read_text(encoding="utf-8"))
        if legacy.is_file()
        else {}
    )


def _priority(term: str, entry: dict) -> tuple[int, int, str]:
    """Terms seen across more workspaces, then more records, rank higher.

    The migration marker is not a workspace and must not be counted as one,
    or every term inherited from the flat cache would outrank freshly
    harvested ones forever and displace them at the cap.
    """
    sources = entry.get("sources", {})
    real = {ws: n for ws, n in sources.items() if ws != LEGACY_SOURCE}
    return (len(real), sum(sources.values()), term)


def update_glossary(
    ws_dirs: list[Path], path: Path | None = None
) -> tuple[Path, int, int]:
    """Merge terms from each workspace's corrections.jsonl. Idempotent.

    Provenance is kept per term (which workspace produced it, how many
    records) so a poisoned term can be traced back and retracted instead of
    being hand-edited out of an anonymous word list.

    Counts are stored per workspace, which makes the merge both additive and
    idempotent: re-running one workspace overwrites its own tally, while a
    separate run for another workspace adds to the total instead of being
    swallowed by it.
    """
    path = path or GLOSSARY_PATH  # runtime lookup so tests can repoint it
    store = _load_store(path)
    before = len(store)
    for term, entry in _harvest(ws_dirs).items():
        prior = store.setdefault(term, {"sources": {}})
        prior["sources"].update(entry["sources"])
        # 실제 출처가 밝혀지면 마이그레이션 자리표시자는 역할이 끝난다.
        if len(prior["sources"]) > 1:
            prior["sources"].pop(LEGACY_SOURCE, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "terms": {t: store[t] for t in sorted(store)},
    }
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path, len(store) - before, len(store)


def load_hotwords(path: Path | None = None, cap: int = 50) -> str | None:
    """Space-joined hotwords string for whisper, or None if no glossary.

    Capped: hotwords ride the decoder prompt per window — an oversized
    glossary degrades transcription instead of helping it. When the cap
    bites, the terms kept are the highest-priority ones (seen across the
    most workspaces), not whichever ones sort first alphabetically; the
    surviving terms are then emitted **lowest priority first** so the
    highest-value ones sit at the end of the prompt, which is the part
    whisper keeps when it truncates.
    """
    path = path or GLOSSARY_PATH
    store = _load_store(path)
    if not store:
        return None
    ranked = sorted(store.items(), key=lambda kv: _priority(*kv))
    return " ".join(term for term, _ in ranked[-cap:]) or None

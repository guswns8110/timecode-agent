from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from video_agent.search import search_workspaces
from video_agent.transcript_segments import load_transcript_segments

from .library_status import completed_workspaces
from .search_query import SearchIntent, parse_search_intent
from .visual_search import VisualSearchEngine


@dataclass
class SearchHit:
    workspace: str
    video: str
    start: float
    end: float
    score: float
    source: str
    text: str
    thumbnail: str | None = None


class UnifiedSearch:
    def __init__(self, library: Path, vision_model: Path):
        self.library = library
        self.vision = VisualSearchEngine(vision_model)

    def run(self, query: str, top: int = 12) -> list[SearchHit]:
        query = query.strip()
        if not query:
            return []
        intent = parse_search_intent(query)
        workspaces = completed_workspaces(self.library)
        by_name = {path.name: path for path in workspaces}
        hits: list[SearchHit] = []

        try:
            lexical = search_workspaces(
                intent.text_query,
                workspace_paths=workspaces,
                top=top * 2,
            )
        except ValueError:
            lexical = []
        for item in lexical:
            source = str(item["source"])
            if intent.dialogue_only and not source.startswith("transcript"):
                continue
            workspace = by_name.get(item["ws"])
            if not workspace:
                continue
            manifest = json.loads(
                (workspace / "manifest.json").read_text(encoding="utf-8")
            )
            hits.append(
                SearchHit(
                    workspace=item["ws"],
                    video=manifest["video"],
                    start=float(item["start"]),
                    end=float(item["end"]),
                    score=min(1.0, 0.72 + float(item["score"]) * 4),
                    source=_source_label(source),
                    text=item["text"],
                )
            )

        if intent.dialogue_only and not hits:
            hits.extend(_fuzzy_dialogue_hits(intent, workspaces, top))

        if intent.visual_query is not None:
            for item in self.vision.search(
                intent.visual_query,
                workspaces,
                top=top,
            ):
                hits.append(
                    SearchHit(
                        workspace=item.workspace,
                        video=item.video,
                        start=item.start,
                        end=item.end,
                        score=max(0.0, min(1.0, item.score)),
                        source="화면 의미",
                        text=f"화면 검색: {intent.visual_query}",
                        thumbnail=item.frame,
                    )
                )

        hits.sort(key=lambda item: item.score, reverse=True)
        deduped: list[SearchHit] = []
        for hit in hits:
            duplicate = any(
                old.workspace == hit.workspace and abs(old.start - hit.start) < 2.0
                for old in deduped
            )
            if not duplicate:
                deduped.append(hit)
            if len(deduped) >= top:
                break
        return sorted(
            deduped,
            key=lambda item: (
                Path(item.video).name.casefold(),
                item.start,
                -item.score,
            ),
        )

    @staticmethod
    def as_dicts(hits: list[SearchHit]) -> list[dict]:
        return [asdict(hit) for hit in hits]


def _source_label(source: str) -> str:
    labels = {
        "transcript": "대사",
        "transcript-window": "대사",
        "ocr": "화면 글자",
        "checkpoint": "장면 설명",
    }
    return labels.get(source.split(":", 1)[0], source)


def _normalized_phrase(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", text).casefold()


def _phrase_similarity(query: str, text: str) -> float:
    wanted = _normalized_phrase(query)
    candidate = _normalized_phrase(text)
    if not wanted or not candidate:
        return 0.0
    if wanted in candidate:
        return 1.0

    minimum = max(1, len(wanted) - 2)
    maximum = min(len(candidate), len(wanted) + 2)
    best = 0.0
    for size in range(minimum, maximum + 1):
        for start in range(0, len(candidate) - size + 1):
            sample = candidate[start : start + size]
            best = max(
                best,
                difflib.SequenceMatcher(None, wanted, sample).ratio(),
            )
    return best


def _fuzzy_dialogue_hits(
    intent: SearchIntent,
    workspaces: list[Path],
    top: int,
) -> list[SearchHit]:
    if len(_normalized_phrase(intent.text_query)) < 4:
        return []
    hits: list[SearchHit] = []
    for workspace in workspaces:
        manifest = json.loads(
            (workspace / "manifest.json").read_text(encoding="utf-8")
        )
        for segment in load_transcript_segments(workspace / "transcript.json"):
            text = str(segment.get("text") or "")
            similarity = _phrase_similarity(intent.text_query, text)
            if similarity < 0.72:
                continue
            hits.append(
                SearchHit(
                    workspace=workspace.name,
                    video=manifest["video"],
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    score=min(0.98, 0.72 + similarity * 0.26),
                    source="대사 유사 일치",
                    text=text,
                )
            )
    hits.sort(key=lambda item: item.score, reverse=True)
    return hits[:top]

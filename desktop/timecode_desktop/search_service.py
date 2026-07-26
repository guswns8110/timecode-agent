from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from video_agent.search import search_workspaces
from video_agent.workspace_discovery import find_workspaces

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

    def run(self, query: str, top: int = 20) -> list[SearchHit]:
        query = query.strip()
        if not query:
            return []
        workspaces = find_workspaces([str(self.library)])
        by_name = {path.name: path for path in workspaces}
        hits: list[SearchHit] = []

        try:
            lexical = search_workspaces(
                query,
                roots=[str(self.library)],
                top=top,
            )
        except ValueError:
            lexical = []
        for item in lexical:
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
                    source=item["source"],
                    text=item["text"],
                )
            )

        for item in self.vision.search(query, workspaces, top=top):
            hits.append(
                SearchHit(
                    workspace=item.workspace,
                    video=item.video,
                    start=item.start,
                    end=item.end,
                    score=max(0.0, min(1.0, item.score)),
                    source="화면 의미",
                    text=query,
                    thumbnail=item.frame,
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        deduped: list[SearchHit] = []
        for hit in hits:
            duplicate = any(
                old.workspace == hit.workspace and abs(old.start - hit.start) < 1.5
                for old in deduped
            )
            if not duplicate:
                deduped.append(hit)
        return deduped[:top]

    @staticmethod
    def as_dicts(hits: list[SearchHit]) -> list[dict]:
        return [asdict(hit) for hit in hits]

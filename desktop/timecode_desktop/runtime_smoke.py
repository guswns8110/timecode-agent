from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from video_agent.fsio import write_text_atomic
from video_agent.proc import run
from video_agent.workspace import Workspace

from .library_status import completed_workspaces, in_progress_marker
from .visual_search import _pooled_features


class PooledOutput:
    def __init__(self, value: object):
        self.pooler_output = value


def main() -> int:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        destination = root / "manifest.json"
        write_text_atomic(destination, '{"ok": true}')
        if destination.read_text(encoding="utf-8") != '{"ok": true}':
            raise RuntimeError("Windows 원자 저장 회귀 테스트 실패")

        workspace = Workspace(root / "library" / "한글 영상")
        workspace.root.mkdir(parents=True)
        workspace.save_manifest({"video": "C:/영상/테스트.mp4"})
        if workspace.manifest["video"] != "C:/영상/테스트.mp4":
            raise RuntimeError("Windows UTF-8 manifest 회귀 테스트 실패")
        workspace.transcript_path.write_text("[]", encoding="utf-8")
        index = workspace.root / "visual-index" / "index.npz"
        index.parent.mkdir()
        index.touch()
        if completed_workspaces(workspace.root.parent) != [workspace.root.resolve()]:
            raise RuntimeError("완료 영상 검색 회귀 테스트 실패")
        in_progress_marker(workspace.root).touch()
        if completed_workspaces(workspace.root.parent):
            raise RuntimeError("불완전 영상 제외 회귀 테스트 실패")

    payload = '{"title": "한글 영상"}'
    completed = run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.stdout.buffer.write({payload.encode()!r})",
        ],
        capture_output=True,
        text=True,
    )
    if json.loads(completed.stdout)["title"] != "한글 영상":
        raise RuntimeError("Windows UTF-8 subprocess 회귀 테스트 실패")

    sentinel = object()
    if _pooled_features(PooledOutput(sentinel)) is not sentinel:
        raise RuntimeError("Transformers 5.x pooled output 회귀 테스트 실패")
    if _pooled_features(sentinel) is not sentinel:
        raise RuntimeError("Transformers 4.x tensor output 회귀 테스트 실패")

    print("runtime-regressions-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

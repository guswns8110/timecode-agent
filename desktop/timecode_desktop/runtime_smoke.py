from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from video_agent.fsio import write_text_atomic

from .visual_search import _pooled_features


class PooledOutput:
    def __init__(self, value: object):
        self.pooler_output = value


def main() -> int:
    with TemporaryDirectory() as temporary:
        destination = Path(temporary) / "manifest.json"
        write_text_atomic(destination, '{"ok": true}')
        if destination.read_text(encoding="utf-8") != '{"ok": true}':
            raise RuntimeError("Windows 원자 저장 회귀 테스트 실패")

    sentinel = object()
    if _pooled_features(PooledOutput(sentinel)) is not sentinel:
        raise RuntimeError("Transformers 5.x pooled output 회귀 테스트 실패")
    if _pooled_features(sentinel) is not sentinel:
        raise RuntimeError("Transformers 4.x tensor output 회귀 테스트 실패")

    print("runtime-regressions-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

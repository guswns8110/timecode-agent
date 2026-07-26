"""파생 아티팩트의 원자적 파일 쓰기 — 부분쓰기 손상 클래스 차단."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_text_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    """같은 디렉터리의 고유 임시 파일에 기록·fsync한 뒤 교체한다.

    중단(SIGKILL·전원차단)돼도 대상 경로는 이전 정상본 또는 새 완전본
    중 하나로만 존재한다: 임시 파일 fsync 후 rename, 부모 디렉터리
    fsync로 rename 메타데이터까지 내린다. 임시 파일은 호출마다
    고유(mkstemp)해 동시 기록자가 서로의 임시 inode를 밟지 않고,
    기존 대상의 권한 모드는 보존된다(신규는 umask 적용 통상 모드).
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_file():
            os.chmod(temporary, path.stat().st_mode & 0o7777)
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(temporary, 0o666 & ~umask)
        temporary.replace(path)
        # Windows에서는 Python의 os.open()으로 디렉터리 핸들을 열 수 없다.
        # 파일 자체의 fsync와 원자 replace는 이미 완료됐으므로, 디렉터리
        # 메타데이터 fsync를 지원하는 POSIX에서만 마지막 단계를 수행한다.
        if os.name != "nt":
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        temporary.unlink(missing_ok=True)

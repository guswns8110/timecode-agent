"""획득 계층 — yt-dlp 영상·자막 다운로드와 사이드카 선택.

ingest 파사드의 내부 모듈(전수점검 2026-07-26 A-3 분해). 다운로드 진행률
스트리밍(tty 한정)·실패 진단·자막 우선순위 규칙이 여기 산다.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .proc import run

_SUB_PRIORITY = (".ko.", ".ko-orig.", ".en.")


def _pick_subtitle(paths: list[Path]) -> Path | None:
    """Manual Korean > auto Korean > English — manual-first caption order."""
    for marker in _SUB_PRIORITY:
        for p in paths:
            if marker in p.name:
                return p
    return paths[0] if paths else None


def _download_url(
    url: str,
    dest_dir: Path,
    max_height: int,
    cookies_from_browser: str | None = None,
) -> Path:
    """yt-dlp: resolution-capped video + subtitles into dest_dir.

    cookies_from_browser is passed through ONLY when the user asked for it
    (login-walled sites like Instagram) — never injected automatically.
    """
    import shutil as _shutil

    if _shutil.which("yt-dlp") is None:
        raise RuntimeError(
            "yt-dlp is required for URL ingest but was not found on PATH; "
            "install it with your OS package manager or the official yt-dlp "
            "release, then retry"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp", url,
        "-f", f"bv*[height<={max_height}]+ba/b[height<={max_height}]",
        "--merge-output-format", "mp4",
        # uploader chapters -> mp4 chapter atoms -> probe() pre-map (free
        # semantic checkpoint drafts; no-op when the video has none)
        "--embed-chapters",
        # manual subtitles only: yt-dlp saves auto-captions under the same
        # <lang>.srt names, and their rolling/misheard text is worse than
        # whisper+hotwords — auto tracks measurably beat manual EN picks.
        "--write-subs",
        "--sub-langs", "ko,en", "--convert-subs", "srt",
        # subtitles are a bonus: a caption 429/absence must not abort the
        # video download itself (whisper is the fallback)
        "--ignore-errors",
        "--no-playlist", "--no-warnings", "-q",
        # 사람 표시 계층: 원제목을 함께 저장 — INDEX·scene-log 표시명의 원료
        # (url-<md5>류 기계 디렉터리명은 사람이 못 읽는다)
        "--print-to-file", "%(title)s", str(dest_dir / "title.txt"),
        # 업로더가 이미 적어 둔 사전지식 — 팀명·인명·대회명이 여기 정자로
        # 있는데 ASR은 그걸 처음 듣는 소리로 받아쓴다. 공짜로 얻어 두고
        # hotwords 프라이어와 브리핑 표시에 쓴다.
        "--print-to-file", "%(uploader)s", str(dest_dir / "uploader.txt"),
        "--print-to-file", "%(description)s", str(dest_dir / "description.txt"),
        "-o", str(dest_dir / "source.%(ext)s"),
    ]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    from .proc import DOWNLOAD_TIMEOUT
    res = _run_download(cmd, timeout=DOWNLOAD_TIMEOUT)
    video = dest_dir / "source.mp4"
    if res.returncode != 0 or not video.is_file():
        found = sorted(dest_dir.glob("source.*"))
        vids = [p for p in found if p.suffix in (".mp4", ".mkv", ".webm")]
        if not vids:
            if "Unsupported URL" in res.stderr:
                raise RuntimeError(
                    "yt-dlp가 지원하지 않는 사이트입니다 (Threads 등 — "
                    "지원 목록: yt-dlp --list-extractors). 브라우저에서 "
                    "영상 파일을 저장한 뒤 로컬 경로로 `va ingest` 하세요."
                )
            raise RuntimeError(
                f"yt-dlp download failed ({res.returncode}): "
                f"{res.stderr.strip()[-300:]}"
            )
        video = vids[0]
    return video


def _run_download(cmd: list[str], *, timeout: float):
    """사람 터미널에서만 yt-dlp 진행률을 실시간으로 흘린다.

    capture_output은 프로세스 종료까지 파이프를 배수하지 않아 `--progress`를
    더해도 다운로드가 끝날 때까지 아무것도 보이지 않는다. tty에서는 stderr를
    줄 단위로 에코+버퍼링해 진행률과 실패 진단("Unsupported URL" 등)을 모두
    지킨다. 비-tty(에이전트·CI)는 기존 캡처 경로 그대로다.
    """
    if not sys.stderr.isatty():
        return run(cmd, timeout=timeout, capture_output=True, text=True)
    import os
    import signal
    import subprocess
    import threading
    from types import SimpleNamespace

    # 자손(yt-dlp가 띄우는 ffmpeg 등)이 stderr 파이프를 물려받으면 부모만
    # 죽여선 EOF가 안 와 readline이 계속 막힌다 — 프로세스 그룹째 종료한다.
    proc = subprocess.Popen(
        cmd + ["--progress", "--newline"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=(sys.platform != "win32"),
    )
    timed_out = threading.Event()

    def _kill_on_timeout() -> None:
        timed_out.set()
        if sys.platform == "win32":
            proc.kill()
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(timeout, _kill_on_timeout)
    timer.start()
    tail: list[str] = []
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            sys.stderr.write(line)
            tail.append(line)
            if len(tail) > 200:
                del tail[0]
        returncode = proc.wait()
    finally:
        timer.cancel()
    if timed_out.is_set():
        raise RuntimeError(
            f"명령 시간 초과({timeout:.0f}s): {cmd[0]} — "
            "미디어 손상 또는 인코더 행(hang) 의심"
        )
    return SimpleNamespace(returncode=returncode, stderr="".join(tail))


def _subtitle_sidecars(video: Path) -> list[Path]:
    return sorted(p for p in video.parent.glob(f"{video.stem}.*")
                  if p.suffix in (".srt", ".vtt") and p != video)


def _download_auto_subs(
    url: str, dest_dir: Path, cookies_from_browser: str | None = None
) -> bool:
    """수동 자막이 없을 때만 — 업로더 채널의 자동생성(ASR) 자막을 받는다.

    업로더가 이미 만들어 둔 전사를 두고 오디오를 처음부터 다시 받아쓸
    이유가 없다(실측 2026-07-25, 43분 영상: 자막 2.7초 vs whisper-small
    12분, 본문도 자막이 앞섬 — "세계 정복"을 whisper는 "세계정보"로 받아씀).

    원어 트랙(`*-orig`)을 먼저 요청한다: 자동자막의 다른 언어 트랙은 원어를
    기계번역한 것이라 원본을 직접 받아쓰는 것보다 나쁘다. 롤링 중복은
    dedup_rolling_cues가 정규화한다(자동 트랙 전용 장치).

    영상 다운로드와 호출을 분리하는 이유: yt-dlp가 자동자막도 <lang>.srt로
    저장해, 한 호출에 섞으면 사람이 만든 자막과 기계가 받아쓴 자막의
    구분이 소실된다.
    """
    from .proc import DOWNLOAD_TIMEOUT

    base = ["yt-dlp", url, "--skip-download", "--write-auto-subs",
            "--convert-subs", "srt", "--ignore-errors", "--no-playlist",
            "--no-warnings", "-q",
            "-o", str(dest_dir / "source.%(ext)s")]
    if cookies_from_browser:
        base += ["--cookies-from-browser", cookies_from_browser]
    probe = dest_dir / "source.mp4"
    for langs in (".*-orig", "ko,en"):
        run(base + ["--sub-langs", langs], timeout=DOWNLOAD_TIMEOUT,
            capture_output=True, text=True)
        if _subtitle_sidecars(probe):
            return True
    return False


def _promote_subtitle(video: Path) -> None:
    """source.ko.srt -> source.srt so find_subtitle_cues sees a sidecar."""
    best = _pick_subtitle(_subtitle_sidecars(video))
    if best:
        target = video.with_suffix(best.suffix)
        if not target.exists():
            best.rename(target)

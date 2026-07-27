"""전사 계층 — whisper 전사 구현·품질 가드·세그먼트 변환.

ingest 파사드의 내부 모듈이다(전수점검 2026-07-26 A-3: 획득·전사·복구·
신호 오케스트레이션이 한 모듈에 있던 대형 모듈 분해). 백엔드 선택·MLX
폴백(`_transcribe`)은 파사드에 남는다 — 폴백이 ingest 모듈 전역을 통해
`_transcribe_faster_whisper`를 해석해야 기존 몽키패치 계약
(`ingest._transcribe_faster_whisper` setattr)이 분해 전과 동일하게 동작한다.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    # ASR-level uncertainty (VideoAgent2 "tool confidence"): low logprob or
    # high no_speech_prob marks a segment whose *transcription itself* is
    # suspect — an independent frame-verification trigger.
    logprob: float | None = None
    no_speech_prob: float | None = None
    # Mean word-level probability — discriminative per segment, unlike
    # avg_logprob which faster-whisper shares across a decode window.
    conf: float | None = None


def segments_from_whisper(raw_segments) -> tuple[list[Segment], list[dict]]:
    """Whisper segments -> (Segments, word list).

    Word timestamps fix the VAD silent-tail problem: a cue like "두 개 더"
    stretched to 42s ends at its last spoken word instead.
    """
    segments: list[Segment] = []
    words_all: list[dict] = []
    for s in raw_segments:
        text = s.text.strip()
        if not text:
            continue
        end = s.end
        conf = None
        words = list(s.words) if getattr(s, "words", None) else []
        if words:
            end = words[-1].end
            conf = round(sum(w.probability for w in words) / len(words), 3)
            words_all.extend(
                {"start": round(w.start, 3), "end": round(w.end, 3),
                 "word": w.word, "p": round(w.probability, 3)}
                for w in words
            )
        segments.append(Segment(
            id=len(segments), start=round(s.start, 3), end=round(end, 3),
            text=text,
            logprob=round(s.avg_logprob, 3)
            if s.avg_logprob is not None else None,
            no_speech_prob=round(s.no_speech_prob, 3)
            if s.no_speech_prob is not None else None,
            conf=conf,
        ))
    return segments, words_all


# (모델명, 인스턴스) 단일 슬롯 — lru_cache(maxsize=1)는 미스 시 새 모델
# 생성이 끝난 뒤에야 이전 항목을 퇴거해 교체 순간 두 모델이 동시 상주한다
# (whisper 모델은 수백 MB~GB). 교체 전 슬롯을 비워 참조를 먼저 놓는다.
_WHISPER_MODEL: tuple[str, str, WhisperModel] | None = None


def _requested_whisper_device() -> str:
    """Resolve the opt-in ASR device without changing the CLI's CPU default.

    ``auto`` is used by the desktop shell: try CUDA first when CTranslate2 can
    see it, then fall back to CPU if model construction fails. Existing CLI
    installs remain on the measured CPU/int8 path unless they opt in.
    """
    requested = os.environ.get("VIDEO_AGENT_ASR_DEVICE", "cpu").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        return "cpu"
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _whisper_model(model: str) -> WhisperModel:
    """프로세스 내 모델 재사용 — 오염·붕괴 재전사가 같은 ingest에서 같은
    모델을 2~3회 로드하던 것을 봉합. 단일 슬롯이라 CLI 한 실행(모델 1개)은
    물론, 장수 프로세스의 모델 교체도 이전 모델을 붙잡지 않는다."""
    global _WHISPER_MODEL
    device = _requested_whisper_device()
    if (
        _WHISPER_MODEL is not None
        and _WHISPER_MODEL[0] == model
        and _WHISPER_MODEL[1] == device
    ):
        return _WHISPER_MODEL[2]
    _WHISPER_MODEL = None
    from faster_whisper import WhisperModel  # lazy: heavy import + model download

    try:
        instance = WhisperModel(
            model,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )
    except Exception:
        if device != "cuda":
            raise
        device = "cpu"
        instance = WhisperModel(model, device="cpu", compute_type="int8")
    _WHISPER_MODEL = (model, device, instance)
    return instance


def _transcribe_faster_whisper(
    wav: Path, model: str, lang: str | None, hotwords: str | None = None,
    condition_on_previous_text: bool = True,
) -> tuple[list[Segment], list[dict], str | None]:
    wm = _whisper_model(model)
    raw_segments, info = wm.transcribe(
        str(wav), language=lang, vad_filter=True, word_timestamps=True,
        hotwords=hotwords,
        condition_on_previous_text=condition_on_previous_text,
        # 수 분짜리 전사가 완전 침묵이면 행(hang)과 구분이 안 된다 —
        # 사람 터미널에서만 진행률(tqdm, 오디오-초 기준)을 켠다.
        # 비-tty(에이전트·CI)는 rate-limited 라인 폭주라 끈다.
        log_progress=sys.stderr.isatty(),
    )
    segments, words = segments_from_whisper(raw_segments)
    return segments, words, getattr(info, "language", None)


# 붕괴 판정 임계 — 실측(2026-07-25, 43분 영상): 붕괴 0.211 vs 정상 0.8x.
COLLAPSE_COVERAGE = 0.5
# 짧은 오디오는 이 비율이 요란하다(한 문장이 전체를 좌우) — 게이트 면제.
COLLAPSE_MIN_SPEECH = 60.0


def _vad_speech_chunks(wav: Path) -> list[tuple[float, float]] | None:
    """전사와 같은 VAD로 본 발화 구간 목록(초) — 붕괴 형상 판정용.

    총량(_vad_speech_seconds)은 "얼마나 빠졌는가"만 말한다. 빠진 발화가
    스톨 지점 앞에 있는지 뒤에 있는지(프리픽스 절단 vs 중간 구멍)는
    구간 목록이 있어야 가른다.
    """
    try:
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError:      # 불완전 설치여도 VAD 판정만 건너뛰고 전사는 계속
        return None
    try:
        audio = decode_audio(str(wav), sampling_rate=16000)
        if isinstance(audio, tuple):
            return None
        chunks = get_speech_timestamps(audio, VadOptions())
    except Exception:        # VAD 실패로 전사를 버릴 이유는 없다 (fail-open)
        return None
    return [(c["start"] / 16000, c["end"] / 16000) for c in chunks]


def _vad_speech_seconds(wav: Path) -> float | None:
    """전사가 쓰는 것과 같은 VAD로 본 발화 총량 — 붕괴 판정의 접지선.

    전사량이 적은 게 '조용한 영상'인지 '죽은 디코딩'인지는 전사문만
    봐서는 구분되지 않는다. 판정에는 전사와 독립된 기준이 필요하다.
    """
    chunks = _vad_speech_chunks(wav)
    if chunks is None:
        return None
    return sum(end - start for start, end in chunks)


def _spoken_seconds(segments: list[Segment]) -> float:
    return sum(s.end - s.start for s in segments)


def _transcript_collapsed(
    segments: list[Segment], speech_seconds: float | None
) -> bool:
    """장문 디코딩이 도중에 죽었는지 — VAD 발화 대비 전사량으로 본다.

    faster-whisper 장문 루프는 오염된 이전 문맥이 프롬프트에 들어가면
    이후 윈도를 계속 no-speech로 흘려보내다 조용히 멈춘다. 예외도 0이 아닌
    exit code도 남지 않는다 — 실측(2026-07-25, 43분 영상): 동일 입력 3회 중
    1회가 1092초에서 정지(발화 2303.8초 중 486.2초 = 0.211), 나머지 2회는
    757·895 세그먼트로 끝까지 완주. 비결정적이라 재현 대기가 아니라
    산출물 검사로 잡아야 한다.
    """
    if not speech_seconds or speech_seconds < COLLAPSE_MIN_SPEECH:
        return False
    return _spoken_seconds(segments) / speech_seconds < COLLAPSE_COVERAGE


def _hotword_contaminated(segments: list[Segment], hotwords: str) -> bool:
    """Repetition-based hallucination check.

    Measured margins (2026-07-22): contaminated no-speech run = hotword-hit
    1.00 / unique-text 0.18; legitimate term-heavy speech ≤ 0.13 / ≥ 0.96.
    Confidence cannot separate them (hallucinated conf reached 0.92).
    """
    if len(segments) < 5:
        return False
    terms = hotwords.split()
    hit = sum(1 for s in segments if any(t in s.text for t in terms))
    uniq = len({s.text.strip() for s in segments})
    return hit / len(segments) >= 0.5 and uniq / len(segments) <= 0.3


def _write_srt(segments: list[Segment], path: Path) -> None:
    def stamp(t: float) -> str:
        ms = round(t * 1000)
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1_000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = [
        f"{seg.id + 1}\n{stamp(seg.start)} --> {stamp(seg.end)}\n{seg.text}\n"
        for seg in segments
    ]
    path.write_text("\n".join(blocks), encoding="utf-8")


def resegment_by_words(
    segments: list[Segment],
    words: list[dict],
    max_gap: float = 1.5,
) -> list[Segment]:
    """Split merged segments at word-gaps > max_gap (BGM-show fix).

    Whisper VAD groups distant utterances into one segment when music sits
    between them; word timestamps reveal the real gaps.
    """
    if not words:
        return segments
    out: list[Segment] = []
    for seg in segments:
        seg_words = [w for w in words
                     if seg.start - 0.01 <= w["start"] < seg.end + 0.01]
        if len(seg_words) < 2:
            out.append(seg)
            continue
        groups: list[list[dict]] = [[seg_words[0]]]
        for prev, cur in zip(seg_words, seg_words[1:]):
            if cur["start"] - prev["end"] > max_gap:
                groups.append([cur])
            else:
                groups[-1].append(cur)
        if len(groups) == 1:
            out.append(seg)
            continue
        for g in groups:
            out.append(Segment(
                id=0,
                start=round(g[0]["start"], 3),
                end=round(g[-1]["end"], 3),
                text="".join(w["word"] for w in g).strip(),
                logprob=seg.logprob, no_speech_prob=seg.no_speech_prob,
                conf=round(sum(w.get("p", 0) for w in g) / len(g), 3)
                if g and g[0].get("p") is not None else seg.conf,
            ))
    for i, seg in enumerate(out):
        seg.id = i
    return out

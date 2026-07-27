from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from video_agent.fsio import write_text_atomic
from video_agent.proc import run
from video_agent.workspace import Workspace

from .audio_search import _audio_keyword, audio_relevance
from .library_status import completed_workspaces, in_progress_marker
from .object_search import (
    PersonPrediction,
    _nms_indices,
    _person_observations,
    exact_person_total_matches,
)
from .search_query import parse_search_intent
from .search_service import UnifiedSearch
from .temporal_search import temporal_relevance
from .translation_search import contains_hangul
from .visual_search import (
    VisualHit,
    _attribute_relevance,
    _pooled_features,
    _shot_span,
    _sigmoid_relevance,
)


class PooledOutput:
    def __init__(self, value: object):
        self.pooler_output = value


class CurrentAudioProcessor:
    def __call__(self, *, audio, **kwargs):
        return audio, kwargs


class LegacyAudioProcessor:
    def __call__(self, *, audios, **kwargs):
        return audios, kwargs


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
        workspace.transcript_path.write_text(
            json.dumps(
                [
                    {
                        "start": 30.0,
                        "end": 32.0,
                        "text": "이거 정말 완벽하네",
                    },
                    {
                        "start": 10.0,
                        "end": 12.0,
                        "text": "완벽하네, 아주 좋아",
                    },
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        index = workspace.root / "visual-index" / "index.npz"
        index.parent.mkdir()
        index.touch()
        if completed_workspaces(workspace.root.parent) != [workspace.root.resolve()]:
            raise RuntimeError("완료 영상 검색 회귀 테스트 실패")
        in_progress_marker(workspace.root).touch()
        if completed_workspaces(workspace.root.parent):
            raise RuntimeError("불완전 영상 제외 회귀 테스트 실패")
        in_progress_marker(workspace.root).unlink()

        dialogue = UnifiedSearch(
            workspace.root.parent,
            root / "unused-model",
        ).run("완벽하네 대사")
        if [hit.start for hit in dialogue] != [10.0, 30.0]:
            raise RuntimeError("자연어 대사·시간순 검색 회귀 테스트 실패")
        if any(hit.source != "대사" for hit in dialogue):
            raise RuntimeError("대사 전용 검색 분리 회귀 테스트 실패")
        selected_dialogue = UnifiedSearch(
            workspace.root.parent,
            root / "unused-model",
        ).run("완벽하네", mode="dialogue")
        if [hit.start for hit in selected_dialogue] != [10.0, 30.0]:
            raise RuntimeError("선택형 대사 검색 회귀 테스트 실패")

        scene_only = UnifiedSearch(
            workspace.root.parent,
            root / "unused-model",
        )
        scene_only.vision.search = lambda *args, **kwargs: []
        if scene_only.run("완벽하네", mode="scene"):
            raise RuntimeError("장면·대사 검색 모드 분리 회귀 테스트 실패")

        visual_intent = parse_search_intent(
            "남자 세 명 나오는 부분 찾아줘",
            mode="scene",
        )
        constraint = visual_intent.object_constraint
        if constraint is None or constraint.count != 3:
            raise RuntimeError("장면 인원수 해석 회귀 테스트 실패")
        if constraint.detector_label != "a man" or not constraint.matches(3):
            raise RuntimeError("장면 객체 검증 회귀 테스트 실패")
        if visual_intent.dialogue_only or visual_intent.visual_query is None:
            raise RuntimeError("자연어 화면 검색 해석 회귀 테스트 실패")

        multiple_objects = parse_search_intent(
            "남자 세 명과 여자 두 명이 회의하는 장면",
            mode="scene",
        )
        if [
            (item.display_name, item.count)
            for item in multiple_objects.object_constraints
        ] != [("남자", 3), ("여자", 2)]:
            raise RuntimeError("복수 객체·인원 조건 해석 회귀 테스트 실패")

        mixed_group = parse_search_intent(
            "남자 세 명 나오는 장면",
            mode="scene",
        ).object_constraints
        mixed_observations = _person_observations(
            [
                PersonPrediction("man", 0.9),
                PersonPrediction("man", 0.9),
                PersonPrediction("man", 0.9),
                PersonPrediction("woman", 0.9),
            ],
            mixed_group,
        )
        if exact_person_total_matches(mixed_group, mixed_observations):
            raise RuntimeError("성별 혼합 인원 제외 회귀 테스트 실패")
        graphic_observation = _person_observations(
            [PersonPrediction("non_person", 0.9)],
            (
                parse_search_intent(
                    "사람 나오는 장면",
                    mode="scene",
                ).object_constraints
            ),
        )
        if graphic_observation[0].count != 0:
            raise RuntimeError("그래픽 비인물 제외 회귀 테스트 실패")

        shield = parse_search_intent(
            "방패가 나오는 장면",
            mode="scene",
        )
        monster = parse_search_intent(
            "괴물이 싸우는 장면",
            mode="scene",
        )
        if shield.object_constraints[0].display_name != "방패":
            raise RuntimeError("방패 오픈셋 객체 해석 회귀 테스트 실패")
        if shield.use_audio or shield.use_temporal:
            raise RuntimeError("정적 객체 검색 모달리티 분리 회귀 테스트 실패")
        if monster.object_constraints[0].display_name != "괴물":
            raise RuntimeError("괴물 오픈셋 객체 해석 회귀 테스트 실패")
        if monster.open_object_terms:
            raise RuntimeError("객체 검색 동작어 오해 회귀 테스트 실패")

        helicopter = parse_search_intent(
            "헬리콥터가 산 위로 날아가는 장면",
            mode="scene",
        )
        if helicopter.open_object_terms != ("헬리콥터",):
            raise RuntimeError("임의 사물명 추출 회귀 테스트 실패")

        mood_intent = parse_search_intent(
            "홍보영상 느낌의 슬로우모션의 비행기가 이륙하는 장면",
            mode="scene",
        )
        translated = " ".join(mood_intent.visual_variants)
        if "slow motion" not in translated or "airplane taking off" not in translated:
            raise RuntimeError("장면 분위기·동작 해석 회귀 테스트 실패")

        arbitrary_intent = parse_search_intent(
            "절제된 하이엔드 화장품 광고처럼 차갑고 몽환적인 장면",
            mode="scene",
        )
        if "차갑고 몽환적인" not in (arbitrary_intent.visual_query or ""):
            raise RuntimeError("임의 자연어 장면 설명 보존 회귀 테스트 실패")
        if not contains_hangul(arbitrary_intent.visual_query or ""):
            raise RuntimeError("한국어 자연어 번역 입력 회귀 테스트 실패")

        abstract_intent = parse_search_intent(
            "자막 없이 차갑고 긴장감 있는 어두운 장면",
            mode="scene",
        )
        if not abstract_intent.no_screen_text:
            raise RuntimeError("화면 글자 제외 조건 해석 회귀 테스트 실패")
        if not abstract_intent.use_audio or not abstract_intent.use_temporal:
            raise RuntimeError("추상 분위기 복합 모달리티 회귀 테스트 실패")
        attribute_names = {
            name
            for name, _ in abstract_intent.attribute_preferences
        }
        if not {"brightness", "warmth", "motion"} <= attribute_names:
            raise RuntimeError("추상 분위기 속성 해석 회귀 테스트 실패")

        screen_intent = parse_search_intent(
            '"성공"이라는 자막이 나오는 장면',
            mode="scene",
        )
        if screen_intent.screen_text_query != "성공":
            raise RuntimeError("화면 글자 자연어 해석 회귀 테스트 실패")
        if screen_intent.dialogue_only:
            raise RuntimeError("화면 자막·음성 대사 분리 회귀 테스트 실패")

        (workspace.root / "ocr_transcript.json").write_text(
            json.dumps(
                [
                    {
                        "start": 40.0,
                        "end": 42.0,
                        "text": "성공을 향한 새로운 시작 성공을향한새로운시작",
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        screen_search = UnifiedSearch(
            workspace.root.parent,
            root / "unused-model",
        )
        screen_search.vision.search = lambda *args, **kwargs: []
        screen_hits = screen_search.run(
            '"성공"이라는 자막이 나오는 장면',
            mode="scene",
        )
        if (
            len(screen_hits) != 1
            or screen_hits[0].source != "화면 글자"
            or screen_hits[0].start != 40.0
        ):
            raise RuntimeError("화면 글자 전용 검색 회귀 테스트 실패")

        combined_search = UnifiedSearch(
            workspace.root.parent,
            root / "unused-model",
        )
        combined_search.vision.search = lambda *args, **kwargs: [
            VisualHit(
                workspace=workspace.root.name,
                video="C:/영상/테스트.mp4",
                start=40.0,
                end=42.0,
                score=0.8,
                frame="frame.jpg",
            )
        ]
        combined_hits = combined_search.run(
            '어두운 화면에 "성공"이라는 자막이 나오는 장면',
            mode="scene",
        )
        if (
            len(combined_hits) != 1
            or combined_hits[0].source != "화면 글자·장면 의미"
            or combined_hits[0].thumbnail != "frame.jpg"
        ):
            raise RuntimeError("화면 글자·장면 복합 조건 회귀 테스트 실패")

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
    relevance = _sigmoid_relevance(
        np.asarray([0.0], dtype=np.float32),
        0.0,
        0.0,
    )
    if not np.isclose(relevance[0], 0.5):
        raise RuntimeError("SigLIP2 관련도 보정 회귀 테스트 실패")
    boxes = np.asarray(
        [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    if _nms_indices(boxes, scores) != [0, 2]:
        raise RuntimeError("객체 중복 제거 회귀 테스트 실패")
    if not np.allclose(
        temporal_relevance(np.asarray([0.08, 0.36])),
        np.asarray([0.0, 1.0]),
    ):
        raise RuntimeError("장면 동작 관련도 회귀 테스트 실패")
    if not np.allclose(
        audio_relevance(np.asarray([0.05, 0.35])),
        np.asarray([0.0, 1.0]),
    ):
        raise RuntimeError("오디오 분위기 관련도 회귀 테스트 실패")
    if _audio_keyword(CurrentAudioProcessor()) != "audio":
        raise RuntimeError("현재 오디오 프로세서 호환성 회귀 테스트 실패")
    if _audio_keyword(LegacyAudioProcessor()) != "audios":
        raise RuntimeError("구형 오디오 프로세서 호환성 회귀 테스트 실패")
    attributes = np.asarray(
        [
            [0.9, 0.5, 0.5, 0.1, 0.8],
            [0.1, 0.5, 0.5, 0.9, 0.2],
        ],
        dtype=np.float32,
    )
    attribute_scores = _attribute_relevance(
        attributes,
        (("brightness", -1.0), ("motion", 1.0)),
    )
    if not np.allclose(attribute_scores, np.asarray([0.45, 0.55])):
        raise RuntimeError("화면 톤·움직임 관련도 회귀 테스트 실패")
    if _shot_span(12.0, [0.0, 10.0, 20.0], 1.0) != (10.0, 20.0):
        raise RuntimeError("실제 컷 구간화 회귀 테스트 실패")

    print("runtime-regressions-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

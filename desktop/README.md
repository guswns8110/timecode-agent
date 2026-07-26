# Timecode Agent Desktop

Windows 10/11 x64용 로컬 GUI다. 설치 프로그램은 작게 유지하고 설치 중
다음 구성요소를 내려받는다.

- 전용 Python 3.12 실행환경
- timecode-agent와 데스크톱 GUI
- CUDA 지원 PyTorch 런타임
- faster-whisper `medium`, `large-v3`
- 다국어 화면 의미 검색용 SigLIP 2
- 영상 동작 검색용 X-CLIP
- 객체·인원 검증용 Grounding DINO
- 한국어 자연어 번역 모델과 한국어/영어 화면 글자 OCR

설치 후 영상과 분석 결과는 외부 서비스로 전송하지 않는다. URL 영상을
직접 가져오거나 모델을 새로 받는 경우에만 네트워크를 사용한다.

## 검색 방식

- `장면 검색`: 장르·형식, 인물·사물·수량, 행동, 카메라와 속도,
  톤·색감, 감정·분위기, 화면 자막·문구를 자연어로 검색한다.
- `대사 검색`: Whisper가 만든 음성 대사 타임라인만 검색한다.
- `자동 검색`: `대사`, `말하는`, `화면 글자`, `자막` 같은 문맥으로
  검색 대상을 자동 선택한다.

복합 장면 문장은 문장 전체를 번역한 뒤 이미지 의미와 영상 동작 점수를
결합한다. 객체·인원 수는 후보 프레임에서 다시 검증하고, 화면 문구는
OCR 적중 구간과 장면 의미를 같은 시간대에서 결합한다.

0.1.8 이전 버전에서 분석한 영상은 영상 동작 및 OCR 인덱스가 없으므로
새 버전에서 한 번 다시 분석해야 한다.

## 개발 실행

Windows PowerShell:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e . -e .\desktop
.\.venv\Scripts\python.exe -m timecode_desktop.main
```

## 설치파일 만들기

GitHub Actions에서 `Windows Desktop Installer` 워크플로를 수동 실행한다.
완료 후 `TimecodeAgent-Setup-x64` artifact를 내려받는다.

설치 프로그램은 사용자별로 설치되므로 관리자 권한이 필요 없다. 설치
중 약 15GB의 여유공간과 안정적인 인터넷 연결이 필요하다.

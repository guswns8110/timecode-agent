# Timecode Agent Desktop

Windows 10/11 x64용 로컬 GUI다. 설치 프로그램은 작게 유지하고 설치 중
다음 구성요소를 내려받는다.

- 전용 Python 3.12 실행환경
- timecode-agent와 데스크톱 GUI
- CUDA 지원 PyTorch 런타임
- faster-whisper `medium`, `large-v3`
- 한국어 화면 의미 검색용 SigLIP 2

설치 후 영상과 분석 결과는 외부 서비스로 전송하지 않는다. URL 영상을
직접 가져오거나 모델을 새로 받는 경우에만 네트워크를 사용한다.

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

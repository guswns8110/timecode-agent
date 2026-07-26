param(
    [Parameter(Mandatory = $true)]
    [string]$InstallRoot
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Uv = Join-Path $InstallRoot "tools\uv.exe"
$Runtime = Join-Path $InstallRoot "runtime"
$Source = Join-Path $InstallRoot "source"
$Desktop = Join-Path $Source "desktop"
$PythonInstall = Join-Path $InstallRoot "python"
$Ffmpeg = Join-Path $InstallRoot "ffmpeg"

$env:UV_PYTHON_INSTALL_DIR = $PythonInstall
$env:UV_CACHE_DIR = Join-Path $env:LOCALAPPDATA "TimecodeAgent\package-cache"
$env:HF_HOME = Join-Path $env:LOCALAPPDATA "TimecodeAgent\hf-cache"

Write-Host "Python 3.12 설치 중..."
& $Uv python install 3.12

Write-Host "전용 실행환경 생성 중..."
& $Uv venv --python 3.12 $Runtime

$Python = Join-Path $Runtime "Scripts\python.exe"

if (-not (Test-Path (Join-Path $Ffmpeg "bin\ffmpeg.exe"))) {
    Write-Host "FFmpeg 설치 중..."
    $FfmpegZip = Join-Path $env:TEMP "timecode-agent-ffmpeg.zip"
    $FfmpegExtract = Join-Path $env:TEMP "timecode-agent-ffmpeg"
    Invoke-WebRequest `
        -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" `
        -OutFile $FfmpegZip
    if (Test-Path $FfmpegExtract) {
        Remove-Item $FfmpegExtract -Recurse -Force
    }
    Expand-Archive -Path $FfmpegZip -DestinationPath $FfmpegExtract -Force
    $ExtractedRoot = Get-ChildItem $FfmpegExtract -Directory | Select-Object -First 1
    New-Item -ItemType Directory -Force $Ffmpeg | Out-Null
    Copy-Item (Join-Path $ExtractedRoot.FullName "*") $Ffmpeg -Recurse -Force
    Remove-Item $FfmpegZip -Force
    Remove-Item $FfmpegExtract -Recurse -Force
}

Write-Host "NVIDIA/CPU AI 런타임 설치 중..."
& $Uv pip install --python $Python `
    torch torchvision `
    --index-url "https://download.pytorch.org/whl/cu128"

Write-Host "timecode-agent와 데스크톱 앱 설치 중..."
& $Uv pip install --python $Python -e $Source -e $Desktop

Write-Host "Medium, Large-v3, 화면 검색 모델 다운로드 중..."
& $Python -m timecode_desktop.preinstall_models

Write-Host "설치 상태 확인 중..."
$env:PATH = (Join-Path $Ffmpeg "bin") + ";" + $env:PATH
& $Python -c "import av, ctranslate2, faster_whisper, torch, transformers, video_agent; print('runtime-ok')"
& (Join-Path $Ffmpeg "bin\ffmpeg.exe") -version

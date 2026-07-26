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
$PythonInstall = Join-Path $InstallRoot "python-runtime-v3"
$Ffmpeg = Join-Path $InstallRoot "ffmpeg"
$Log = Join-Path $InstallRoot "install.log"
$CompleteMarker = Join-Path $InstallRoot "install-complete.txt"

$env:UV_PYTHON_INSTALL_DIR = $PythonInstall
$env:UV_CACHE_DIR = Join-Path $env:LOCALAPPDATA "TimecodeAgent\package-cache"
$env:HF_HOME = Join-Path $env:LOCALAPPDATA "TimecodeAgent\hf-cache"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList,
        [Parameter(Mandatory = $true)]
        [string]$StepName
    )

    Write-Host ""
    Write-Host "[$StepName]"
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName 실패 (종료 코드: $LASTEXITCODE)"
    }
}

if (Test-Path $CompleteMarker) {
    Remove-Item $CompleteMarker -Force
}

try {
    try {
        $Host.UI.RawUI.WindowTitle = "Timecode Agent 설치"
    }
    catch {
        # 콘솔 제목을 변경할 수 없는 호스트에서도 설치는 계속한다.
    }

    Start-Transcript -Path $Log -Force | Out-Null

    if (-not (Test-Path $Uv)) {
        throw "설치 도구를 찾을 수 없습니다: $Uv"
    }

    $ManagedPython = Join-Path $PythonInstall "python.exe"
    if (-not (Test-Path $ManagedPython)) {
        throw "설치 프로그램에 포함된 Python 3.12를 찾을 수 없습니다."
    }

    if (Test-Path $Runtime) {
        Remove-Item $Runtime -Recurse -Force
    }

    Invoke-Checked `
        -FilePath $Uv `
        -ArgumentList @(
            "venv",
            "--python", $ManagedPython,
            $Runtime
        ) `
        -StepName "전용 Python 실행환경 생성"

    $Python = Join-Path $Runtime "Scripts\python.exe"
    $PythonWindowed = Join-Path $Runtime "Scripts\pythonw.exe"
    if (-not (Test-Path $Python) -or -not (Test-Path $PythonWindowed)) {
        throw "전용 Python 실행환경 생성 후 실행 파일을 찾을 수 없습니다."
    }

    if (-not (Test-Path (Join-Path $Ffmpeg "bin\ffmpeg.exe"))) {
        Write-Host ""
        Write-Host "[FFmpeg 다운로드]"
        $FfmpegZip = Join-Path $env:TEMP "timecode-agent-ffmpeg.zip"
        $FfmpegExtract = Join-Path $env:TEMP "timecode-agent-ffmpeg"
        Invoke-WebRequest `
            -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" `
            -OutFile $FfmpegZip
        if (Test-Path $FfmpegExtract) {
            Remove-Item $FfmpegExtract -Recurse -Force
        }
        Expand-Archive -Path $FfmpegZip -DestinationPath $FfmpegExtract -Force
        $ExtractedRoot = Get-ChildItem $FfmpegExtract -Directory |
            Select-Object -First 1
        if ($null -eq $ExtractedRoot) {
            throw "FFmpeg 압축 파일의 내용을 확인할 수 없습니다."
        }
        New-Item -ItemType Directory -Force $Ffmpeg | Out-Null
        Copy-Item `
            (Join-Path $ExtractedRoot.FullName "*") `
            $Ffmpeg `
            -Recurse `
            -Force
        Remove-Item $FfmpegZip -Force
        Remove-Item $FfmpegExtract -Recurse -Force
    }

    Invoke-Checked `
        -FilePath $Uv `
        -ArgumentList @(
            "pip", "install",
            "--python", $Python,
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu128"
        ) `
        -StepName "NVIDIA/CPU AI 런타임 설치"

    Invoke-Checked `
        -FilePath $Uv `
        -ArgumentList @(
            "pip", "install",
            "--python", $Python,
            "-e", $Source,
            "-e", $Desktop
        ) `
        -StepName "Timecode Agent 프로그램 설치"

    Invoke-Checked `
        -FilePath $Python `
        -ArgumentList @("-m", "timecode_desktop.preinstall_models") `
        -StepName "음성, 화면, 객체, 장면 동작, 오디오 분위기, 한국어 자연어 모델 다운로드"

    $env:PATH = (Join-Path $Ffmpeg "bin") + ";" + $env:PATH
    Invoke-Checked `
        -FilePath $Python `
        -ArgumentList @(
            "-c",
            "import av, ctranslate2, faster_whisper, torch, transformers, video_agent; print('runtime-ok')"
        ) `
        -StepName "AI 런타임 확인"

    Invoke-Checked `
        -FilePath (Join-Path $Ffmpeg "bin\ffmpeg.exe") `
        -ArgumentList @("-version") `
        -StepName "FFmpeg 확인"

    Set-Content `
        -Path $CompleteMarker `
        -Value "Timecode Agent runtime installation completed." `
        -Encoding UTF8

    Write-Host ""
    Write-Host "설치가 완료되었습니다."
}
catch {
    Write-Host ""
    Write-Host "설치 실패: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "설치 로그: $Log" -ForegroundColor Yellow
    exit 1
}
finally {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
        # Transcript가 시작되기 전 실패했을 수 있다.
    }
}

exit 0

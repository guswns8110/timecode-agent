from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GpuInfo:
    name: str
    vram_total_gb: float
    vram_free_gb: float
    utilization: int | None


@dataclass
class EnvironmentReport:
    os: str
    cpu: str
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    ffmpeg_ok: bool
    cuda_ok: bool
    gpus: list[GpuInfo]
    recommendation: str
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _nvidia_smi_gpus() -> list[GpuInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    gpus: list[GpuInfo] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append(
                GpuInfo(
                    name=parts[0],
                    vram_total_gb=round(float(parts[1]) / 1024, 1),
                    vram_free_gb=round(float(parts[2]) / 1024, 1),
                    utilization=int(parts[3]),
                )
            )
        except ValueError:
            continue
    return gpus


def _cuda_import_ok() -> bool:
    try:
        import ctranslate2

        return bool(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:
        return False


def inspect_environment(storage_path: Path) -> EnvironmentReport:
    import psutil

    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(storage_path)
    gpus = _nvidia_smi_gpus()
    cuda_ok = bool(gpus) and _cuda_import_ok()
    max_vram = max((gpu.vram_total_gb for gpu in gpus), default=0)
    warnings: list[str] = []

    if disk.free < 15 * 1024**3:
        warnings.append("저장공간이 15GB 미만입니다.")
    if memory.total < 16 * 1024**3:
        warnings.append("RAM 16GB 미만에서는 large-v3가 느릴 수 있습니다.")
    if gpus and not cuda_ok:
        warnings.append("NVIDIA GPU는 보이지만 CUDA 런타임 테스트에 실패했습니다.")

    if cuda_ok and max_vram >= 10:
        recommendation = "large-v3 권장"
    elif cuda_ok and max_vram >= 6:
        recommendation = "medium 권장"
    else:
        recommendation = "CPU 모드: medium 권장"

    return EnvironmentReport(
        os=f"{platform.system()} {platform.release()} {platform.machine()}",
        cpu=platform.processor() or "알 수 없음",
        cpu_threads=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=round(memory.total / 1024**3, 1),
        ram_available_gb=round(memory.available / 1024**3, 1),
        disk_free_gb=round(disk.free / 1024**3, 1),
        ffmpeg_ok=shutil.which("ffmpeg") is not None,
        cuda_ok=cuda_ok,
        gpus=gpus,
        recommendation=recommendation,
        warnings=warnings,
    )


def live_usage() -> dict:
    import psutil

    memory = psutil.virtual_memory()
    gpus = _nvidia_smi_gpus()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": round((memory.total - memory.available) / 1024**3, 1),
        "ram_total_gb": round(memory.total / 1024**3, 1),
        "gpus": [asdict(gpu) for gpu in gpus],
    }

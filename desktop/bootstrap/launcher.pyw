from __future__ import annotations

import os
import sys
from pathlib import Path


def _add_runtime_dlls() -> None:
    runtime = Path(sys.executable).resolve().parent.parent
    install_root = runtime.parent
    candidates = [
        install_root / "ffmpeg" / "bin",
        runtime / "Lib" / "site-packages" / "torch" / "lib",
        runtime / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        runtime / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
    ]
    existing = [str(path) for path in candidates if path.is_dir()]
    if not existing:
        return
    os.environ["PATH"] = os.pathsep.join(existing + [os.environ.get("PATH", "")])
    for path in existing:
        try:
            os.add_dll_directory(path)
        except (AttributeError, OSError):
            pass


_add_runtime_dlls()

from timecode_desktop.main import main

raise SystemExit(main())

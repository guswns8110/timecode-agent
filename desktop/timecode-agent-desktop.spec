# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

root = Path(SPECPATH).parent.parent

analysis = Analysis(
    ["timecode_desktop/main.py"],
    pathex=[str(root / "src"), str(root / "desktop")],
    binaries=[],
    datas=[],
    hiddenimports=[
        "av",
        "ctranslate2",
        "faster_whisper",
        "huggingface_hub",
        "numpy",
        "PIL",
        "psutil",
        "torch",
        "transformers",
        "video_agent",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="TimecodeAgent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

collect = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TimecodeAgent",
)

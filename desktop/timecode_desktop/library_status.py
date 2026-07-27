from __future__ import annotations

from pathlib import Path

from video_agent.workspace_discovery import find_workspaces

IN_PROGRESS_MARKER = ".desktop-analysis-in-progress"


def in_progress_marker(workspace: Path) -> Path:
    return workspace / IN_PROGRESS_MARKER


def workspace_ready(workspace: Path) -> bool:
    workspace = Path(workspace)
    return (
        not in_progress_marker(workspace).exists()
        and (workspace / "manifest.json").is_file()
        and (workspace / "transcript.json").is_file()
        and (workspace / "visual-index" / "index.npz").is_file()
    )


def completed_workspaces(library: Path | str) -> list[Path]:
    return [
        workspace
        for workspace in find_workspaces([str(library)])
        if workspace_ready(workspace)
    ]

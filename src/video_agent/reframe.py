"""16:9 -> 9:16 reframing for shorts delivery (pan-between-faces / split).

Deterministic half of the clipify (MIT) technique, adapted to the va loop:
the agent picks face ROIs from ONE captured frame (selection), then ffmpeg
does everything else with zero tokens (placement) — per-frame motion energy
per ROI via tblend-difference + signalstats, an argmax speaker timeline with
minimum dwell, and a hard-cut crop x-expression render. No cv2, no tracking
model; a static camera within a clip is assumed (typical talk/interview cut).

Side product: the speaker timeline is a visual active-speaker signal that can
cross-check diarization labels (S0 = left person).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .proc import run
from .workspace import Workspace

MIN_DWELL_DEFAULT = 1.0


def parse_roi(spec: str) -> dict:
    """"x,y,w,h" -> dict; pixel coords in the clip's own space."""
    try:
        x, y, w, h = (int(v) for v in spec.split(","))
    except ValueError:
        raise ValueError(f"ROI must be 'x,y,w,h' integers, got {spec!r}") from None
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        raise ValueError(f"ROI out of range: {spec!r}")
    return {"x": x, "y": y, "w": w, "h": h}


def _parse_metadata_file(path: Path) -> list[tuple[float, float]]:
    """ffmpeg metadata=print file -> [(pts_time, YAVG), ...]."""
    out: list[tuple[float, float]] = []
    t = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("frame:"):
            t = None
            for field in line.split():
                if field.startswith("pts_time:"):
                    t = float(field.removeprefix("pts_time:"))
        elif "signalstats.YAVG=" in line and t is not None:
            out.append((t, float(line.partition("=")[2])))
    return out


def roi_motion(video: Path, rois: list[dict]) -> list[list[tuple[float, float]]]:
    """Per-ROI motion energy series in ONE ffmpeg pass (no frames extracted)."""
    with tempfile.TemporaryDirectory() as td:
        files = [Path(td) / f"roi{i}.txt" for i in range(len(rois))]
        n = len(rois)
        parts = [f"[0:v]split={n}" + "".join(f"[s{i}]" for i in range(n)) + ";"]
        for i, (r, f) in enumerate(zip(rois, files)):
            parts.append(
                f"[s{i}]crop={r['w']}:{r['h']}:{r['x']}:{r['y']},format=gray,"
                f"tblend=all_mode=difference,signalstats,"
                f"metadata=print:key=lavfi.signalstats.YAVG:file='{f}'[o{i}];"
            )
        graph = "".join(parts).rstrip(";")
        cmd = ["ffmpeg", "-loglevel", "error", "-i", str(video),
               "-filter_complex", graph]
        for i in range(n):
            cmd += ["-map", f"[o{i}]", "-f", "null", "-"]
        res = run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"ffmpeg motion pass failed ({res.returncode}): "
                f"{res.stderr.strip()}")
        return [_parse_metadata_file(f) for f in files]


def speaker_timeline(
    motions: list[list[tuple[float, float]]],
    min_dwell: float = MIN_DWELL_DEFAULT,
) -> list[dict]:
    """Per-frame argmax ROI -> merged segments [{start, end, roi}].

    Segments shorter than min_dwell (interjections, noise blips) merge into
    the previous speaker, mirroring clipify's dwell rule.
    """
    if not motions or not motions[0]:
        return []
    length = min(len(m) for m in motions)
    raw: list[dict] = []
    for j in range(length):
        t = motions[0][j][0]
        winner = max(range(len(motions)), key=lambda i: motions[i][j][1])
        if raw and raw[-1]["roi"] == winner:
            raw[-1]["end"] = t
        else:
            raw.append({"start": t, "end": t, "roi": winner})
    merged: list[dict] = []
    for seg in raw:
        if merged and seg["end"] - seg["start"] < min_dwell:
            merged[-1]["end"] = seg["end"]
        elif (merged and merged[-1]["roi"] == seg["roi"]):
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    for seg in merged:
        seg["start"] = round(seg["start"], 3)
        seg["end"] = round(seg["end"], 3)
    return merged


def build_pan_expr(segments: list[dict], xs: list[int]) -> str:
    """Hard-cut crop-x expression: if(lt(t,e0),x0, if(lt(t,e1),x1, ...))."""
    if not segments:
        raise ValueError("empty speaker timeline")
    expr = str(xs[segments[-1]["roi"]])
    for seg in reversed(segments[:-1]):
        expr = f"if(lt(t\\,{seg['end']})\\,{xs[seg['roi']]}\\,{expr})"
    return expr


def _probe_size(video: Path) -> tuple[int, int]:
    res = run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(video)], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr.strip()}")
    w, h = res.stdout.strip().split(",")
    return int(w), int(h)


def _run_render(cmd: list[str], dest: Path) -> Path:
    res = run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not dest.is_file():
        raise RuntimeError(
            f"ffmpeg reframe failed ({res.returncode}): {res.stderr.strip()}")
    return dest


def render_pan(
    clip: Path, dest: Path, segments: list[dict], rois: list[dict],
    out_w: int = 1080, out_h: int = 1920,
) -> Path:
    """Vertical strip crop that hard-cuts between ROI centers."""
    src_w, src_h = _probe_size(clip)
    strip_w = round(src_h * out_w / out_h / 2) * 2
    xs = [min(max(r["x"] + r["w"] // 2 - strip_w // 2, 0), src_w - strip_w)
          for r in rois]
    expr = build_pan_expr(segments, xs)
    vf = (f"crop={strip_w}:{src_h}:x='{expr}':y=0,"
          f"scale={out_w}:{out_h}:flags=lanczos,setsar=1")
    return _run_render(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip),
         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest)], dest)


def render_split(
    clip: Path, dest: Path, rois: list[dict],
    out_w: int = 1080, out_h: int = 1920,
) -> Path:
    """Two ROI tiles stacked vertically (both faces always visible)."""
    if len(rois) != 2:
        raise ValueError("split mode needs exactly 2 ROIs")
    src_w, src_h = _probe_size(clip)
    tile_h = out_h // 2
    # widen each ROI to the tile aspect, clamped to the source
    crops = []
    for r in rois:
        cw = min(src_w, round(r["h"] * out_w / tile_h / 2) * 2)
        cx = min(max(r["x"] + r["w"] // 2 - cw // 2, 0), src_w - cw)
        crops.append((cw, r["h"], cx, r["y"]))
    fc = (
        f"[0:v]split=2[a][b];"
        f"[a]crop={crops[0][0]}:{crops[0][1]}:{crops[0][2]}:{crops[0][3]},"
        f"scale={out_w}:{tile_h},setsar=1[t0];"
        f"[b]crop={crops[1][0]}:{crops[1][1]}:{crops[1][2]}:{crops[1][3]},"
        f"scale={out_w}:{tile_h},setsar=1[t1];"
        f"[t0][t1]vstack[v]"
    )
    return _run_render(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip),
         "-filter_complex", fc, "-map", "[v]", "-map", "0:a?",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest)], dest)


def reframe(
    ws: Workspace, clip: str, rois: list[dict], mode: str = "pan",
    min_dwell: float = MIN_DWELL_DEFAULT, name: str | None = None,
) -> tuple[Path, list[dict]]:
    """Reframe a clip in clips/ (or any path) to 9:16. Returns (dest, timeline)."""
    src = Path(clip)
    if not src.is_file():
        src = ws.clips_dir / clip
    if not src.is_file():
        raise FileNotFoundError(f"clip not found: {clip}")
    if not rois:
        raise ValueError("at least one --roi required")
    dest = ws.clips_dir / (name or f"{src.stem}_916_{mode}.mp4")
    if mode == "split":
        return render_split(src, dest, rois), []
    if len(rois) == 1:
        segments = [{"start": 0.0, "end": 10 ** 9, "roi": 0}]
    else:
        segments = speaker_timeline(roi_motion(src, rois), min_dwell=min_dwell)
        if not segments:
            raise RuntimeError("motion pass produced no frames")
    return render_pan(src, dest, segments, rois), segments

from __future__ import annotations

import base64
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class EvidenceImage:
    label: str
    path: Path
    note: str


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def image_to_data_uri(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def ffprobe_duration(video_path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(proc.stdout.strip())


def resolve_video_path(video_dir: Path, qid: str) -> Path:
    candidate = video_dir / f"{qid}.mp4"
    if candidate.exists():
        return candidate
    matches = sorted(video_dir.glob(f"*{qid}.mp4"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No video found for {qid} in {video_dir}")


def question_window(question: dict[str, Any], video_duration: float) -> tuple[float, float]:
    start = max(float(question.get("question_start_time") or 0.0), 0.0)
    stop = float(question.get("question_stop_time") or video_duration)
    stop = min(max(stop, start + 0.1), video_duration)
    return start, stop


def even_times(start: float, end: float, count: int) -> list[float]:
    count = max(1, count)
    if count == 1:
        return [(start + end) / 2]
    return [start + (end - start) * i / (count - 1) for i in range(count)]


def classify_question(question: dict[str, Any]) -> str:
    question_text = str(question.get("question_text") or "").lower()
    options_text = " ".join(str(value) for value in (question.get("options") or {}).values()).lower()
    text = f"{question_text} {options_text}"
    motion_terms = ["move", "moving", "direction", "trajectory", "toward", "away", "flip", "turn", "rotate", "jump", "roll", "pass", "enter", "exit"]
    if any(term in question_text for term in ["from first to last", "in order", "order of", "arrange in order", "sequence"]):
        return "motion_trajectory"
    if any(term in text for term in ["how many times", "number of times"]) and any(term in text for term in motion_terms):
        return "motion_trajectory"
    if "how many" in text and any(term in text for term in motion_terms) and any(term in text for term in [" does ", " do ", " did ", " is ", " are "]):
        return "motion_trajectory"
    if any(term in text for term in ["how many", "number of", "count", "total"]):
        return "inferred_counting"
    if any(term in text for term in ["why", "cause", "caused", "because", "reason", "motivated"]):
        return "causal_motivational"
    if any(term in question_text for term in ["see", "visible", "visibility", "looking", "look at", "viewing direction"]):
        return "viewpoint_visibility"
    if any(term in text for term in ["talk", "speak", "conversation", "interact", "relationship"]):
        return "social_interaction"
    if any(term in question_text for term in motion_terms + ["first", "last", "facing", "orientation"]):
        return "motion_trajectory"
    if any(term in text for term in ["above", "below", "top", "bottom", "higher", "lower", "under", "over"]):
        return "vertical_spatial"
    if any(term in text for term in ["front", "behind", "near", "closer", "farther", "depth", "proximity"]):
        return "relative_depth"
    if any(term in text for term in ["left", "right", "next to", "between", "beside", "side"]):
        return "lateral_spatial"
    if any(term in text for term in ["where", "located", "position", "on the", "in the"]):
        return "physical_environment"
    return "other"


def normalize_pct_box(box: Iterable[float] | None) -> tuple[float, float, float, float]:
    if not box:
        return (0.0, 0.0, 100.0, 100.0)
    values = [float(v) for v in list(box)[:4]]
    if len(values) != 4:
        return (0.0, 0.0, 100.0, 100.0)
    x1, y1, x2, y2 = values
    x1, x2 = sorted((max(0.0, min(100.0, x1)), max(0.0, min(100.0, x2))))
    y1, y2 = sorted((max(0.0, min(100.0, y1)), max(0.0, min(100.0, y2))))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return (0.0, 0.0, 100.0, 100.0)
    return (x1, y1, x2, y2)


def as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_frame(
    video_path: Path,
    time_sec: float,
    out_path: Path,
    *,
    label: str | None = None,
    crop_pct: Iterable[float] | None = None,
    max_side: int = 960,
    jpeg_quality: int = 88,
    draw_grid: bool = False,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_suffix(".raw.jpg")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(time_sec, 0.0):.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(raw_path),
        ],
        check=True,
    )
    with Image.open(raw_path) as img:
        img = img.convert("RGB")
        x1, y1, x2, y2 = normalize_pct_box(crop_pct)
        if (x1, y1, x2, y2) != (0.0, 0.0, 100.0, 100.0):
            width, height = img.size
            img = img.crop(
                (
                    round(width * x1 / 100),
                    round(height * y1 / 100),
                    round(width * x2 / 100),
                    round(height * y2 / 100),
                )
            )
        img = resize_max_side(img, max_side)
        if draw_grid:
            draw_screen_grid_on_image(img)
        if label:
            draw_label(img, label)
        img.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)
    raw_path.unlink(missing_ok=True)
    return out_path


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    width, height = img.size
    scale = min(max_side / max(width, height), 1.0)
    if scale >= 1.0:
        return img
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    return img.resize((round(width * scale), round(height * scale)), resample)


def draw_label(img: Image.Image, label: str) -> None:
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    pad = 5
    bbox = draw.textbbox((0, 0), label, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.rectangle([0, 0, w + pad * 2, h + pad * 2], fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)


def draw_screen_grid_on_image(img: Image.Image) -> None:
    draw = ImageDraw.Draw(img)
    width, height = img.size
    line = (255, 240, 0)
    center = (255, 60, 60)
    for frac in (0.25, 0.5, 0.75):
        x = round(width * frac)
        y = round(height * frac)
        color = center if frac == 0.5 else line
        draw.line([(x, 0), (x, height)], fill=color, width=2)
        draw.line([(0, y), (width, y)], fill=color, width=2)


def extract_time_series(
    video_path: Path,
    out_dir: Path,
    times: list[float],
    *,
    prefix: str,
    crop_pct: Iterable[float] | None = None,
    max_side: int = 800,
    jpeg_quality: int = 86,
    draw_grid: bool = False,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for idx, t in enumerate(times, 1):
        label = f"{prefix}{idx:02d} t={t:.1f}s"
        path = out_dir / f"{prefix.lower()}_{idx:03d}.jpg"
        frames.append(
            extract_frame(
                video_path,
                t,
                path,
                label=label,
                crop_pct=crop_pct,
                max_side=max_side,
                jpeg_quality=jpeg_quality,
                draw_grid=draw_grid,
            )
        )
    return frames


def make_contact_sheet(
    frame_paths: list[Path],
    output_path: Path,
    *,
    columns: int = 6,
    cell_width: int = 300,
    title: str | None = None,
    jpeg_quality: int = 86,
) -> Path:
    if not frame_paths:
        raise ValueError("frame_paths is empty")
    columns = max(1, columns)
    rows = math.ceil(len(frame_paths) / columns)
    font = ImageFont.load_default()
    resized: list[Image.Image] = []
    cell_heights: list[int] = []
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    for path in frame_paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size
            scale = cell_width / max(width, 1)
            cell_height = max(1, round(height * scale))
            resized.append(img.resize((cell_width, cell_height), resample))
            cell_heights.append(cell_height)
    max_cell_height = max(cell_heights)
    pad = 6
    title_h = 26 if title else 0
    sheet = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * pad,
            title_h + rows * max_cell_height + (rows + 1) * pad,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.rectangle([0, 0, sheet.size[0], title_h], fill=(0, 0, 0))
        draw.text((8, 7), title[:180], fill=(255, 255, 255), font=font)
    for idx, img in enumerate(resized):
        row = idx // columns
        col = idx % columns
        x = pad + col * (cell_width + pad)
        y = title_h + pad + row * (max_cell_height + pad)
        sheet.paste(img, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)
    return output_path


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_scout_board(
    video_path: Path,
    question: dict[str, Any],
    evidence_dir: Path,
    *,
    scout_frames: int = 12,
    jpeg_quality: int = 86,
) -> EvidenceImage:
    duration = ffprobe_duration(video_path)
    start, end = question_window(question, duration)
    frames = extract_time_series(
        video_path,
        evidence_dir / "scout_frames",
        even_times(start, end, scout_frames),
        prefix="S",
        max_side=720,
        jpeg_quality=jpeg_quality,
    )
    sheet = make_contact_sheet(
        frames,
        evidence_dir / "scout_timeline.jpg",
        columns=4,
        cell_width=320,
        title=f"Scout timeline: {start:.1f}s to {end:.1f}s",
        jpeg_quality=jpeg_quality,
    )
    return EvidenceImage("scout_timeline", sheet, "Uniform overview frames across the question time window.")


def build_policy_boards(
    video_path: Path,
    question: dict[str, Any],
    evidence_dir: Path,
    question_type: str,
    *,
    jpeg_quality: int = 84,
) -> list[EvidenceImage]:
    duration = ffprobe_duration(video_path)
    start, end = question_window(question, duration)
    images: list[EvidenceImage] = []
    text = str(question.get("question_text") or "").lower()

    def add_series_board(
        *,
        label: str,
        times: list[float],
        prefix: str,
        title: str,
        note: str,
        crop_pct: Iterable[float] | None = None,
        max_side: int = 760,
        columns: int = 6,
        cell_width: int = 300,
        draw_grid: bool = False,
    ) -> None:
        frames = extract_time_series(
            video_path,
            evidence_dir / f"{label}_frames",
            times,
            prefix=prefix,
            crop_pct=crop_pct,
            max_side=max_side,
            jpeg_quality=jpeg_quality,
            draw_grid=draw_grid,
        )
        images.append(
            EvidenceImage(
                label,
                make_contact_sheet(
                    frames,
                    evidence_dir / f"{label}.jpg",
                    columns=columns,
                    cell_width=cell_width,
                    title=title,
                    jpeg_quality=jpeg_quality,
                ),
                note,
            )
        )

    def add_endpoint_board(*, draw_grid: bool) -> None:
        endpoint_terms = ["end", "final", "start", "beginning", "at first", "last"]
        if not any(term in text for term in endpoint_terms):
            return
        endpoint_times = sorted(
            set(
                [
                    start,
                    start + (end - start) * 0.05,
                    (start + end) / 2,
                    end - (end - start) * 0.1,
                    end - 0.5,
                    end,
                ]
            )
        )
        endpoint_times = [min(max(t, start), end) for t in endpoint_times]
        add_series_board(
            label="endpoint_check",
            times=endpoint_times,
            prefix="E",
            title="Endpoint/start-state check",
            note="Start/end state board for endpoint-worded questions.",
            max_side=900,
            columns=3,
            cell_width=380,
            draw_grid=draw_grid,
        )

    if question_type == "inferred_counting":
        add_series_board(
            label="count_dense_full",
            times=even_times(start, end, min(64, max(18, round((end - start) * 2)))),
            prefix="D",
            title="Counting sweep: dense full-frame timeline",
            note="Dense full-frame timeline for high-recall object/event counting.",
            max_side=640,
            columns=8,
            cell_width=240,
        )
        add_series_board(
            label="count_highres_keyframes",
            times=even_times(start, end, 8),
            prefix="K",
            title="Counting high-resolution keyframes",
            note="Larger keyframes for identifying small, partial, distant, or crowded count candidates.",
            max_side=1100,
            columns=4,
            cell_width=420,
        )
        sweep_times = even_times(start, end, min(24, max(10, round((end - start) * 0.75))))
        for label, box in (
            ("count_left_half", (0, 0, 55, 100)),
            ("count_center_band", (20, 0, 80, 100)),
            ("count_right_half", (45, 0, 100, 100)),
        ):
            add_series_board(
                label=label,
                times=sweep_times,
                prefix=label[:2].upper(),
                title=f"Counting regional sweep: {label}",
                note="Regional crop sweep to catch edge, partial, or occluded count candidates.",
                crop_pct=box,
                max_side=760,
                columns=6,
                cell_width=300,
            )
        return images

    if question_type == "causal_motivational":
        add_series_board(
            label="causal_event_timeline",
            times=even_times(start, end, min(32, max(16, round((end - start) * 1.5)))),
            prefix="C",
            title="Causal/motivational sweep: before -> trigger -> effect",
            note="Dense event timeline for visible triggers, reactions, and state changes.",
            max_side=720,
            columns=8,
            cell_width=260,
        )
        add_series_board(
            label="causal_key_states",
            times=even_times(start, end, 6),
            prefix="K",
            title="Causal key states: start/middle/end comparison",
            note="Key states for checking temporal precedence and whether a visible cause explains the outcome.",
            max_side=900,
            columns=3,
            cell_width=380,
        )
        add_endpoint_board(draw_grid=False)
        return images

    if question_type == "motion_trajectory":
        add_series_board(
            label="motion_dense_timeline",
            times=even_times(start, end, min(64, max(24, round((end - start) * 6)))),
            prefix="M",
            title="Motion/trajectory sweep: dense ordered timeline",
            note="Dense motion timeline for direction, path, action cycles, and before/after order.",
            max_side=720,
            columns=8,
            cell_width=240,
        )
        add_series_board(
            label="motion_start_mid_end_grid",
            times=even_times(start, end, 7),
            prefix="G",
            title="Motion start/mid/end grid",
            note="Sparse grid board for comparing displacement and action completion across the clip.",
            max_side=900,
            columns=4,
            cell_width=340,
            draw_grid=True,
        )
        add_endpoint_board(draw_grid=True)
        return images

    if question_type == "lateral_spatial":
        add_series_board(
            label="lateral_relation_grid",
            times=even_times(start, end, 20),
            prefix="L",
            title="Lateral spatial grid: target/reference relation over time",
            note="Grid timeline for left/right/between/side relations and camera-vs-scene frame checks.",
            max_side=820,
            columns=5,
            cell_width=310,
            draw_grid=True,
        )
        add_series_board(
            label="lateral_wide_keyframes",
            times=even_times(start, end, 6),
            prefix="W",
            title="Lateral wide keyframes",
            note="Higher-resolution wide frames for identity continuity and scene-world relation stitching.",
            max_side=960,
            columns=3,
            cell_width=380,
        )
        add_endpoint_board(draw_grid=True)
        return images

    if question_type == "vertical_spatial":
        add_series_board(
            label="vertical_relation_grid",
            times=even_times(start, end, 20),
            prefix="V",
            title="Vertical spatial grid: above/below/support relation",
            note="Grid timeline for above/below, support surfaces, stacking, suspension, and visible height.",
            max_side=820,
            columns=5,
            cell_width=310,
            draw_grid=True,
        )
        add_series_board(
            label="vertical_full_height_keyframes",
            times=even_times(start, end, 6),
            prefix="H",
            title="Vertical full-height keyframes",
            note="Higher-resolution keyframes to avoid confusing depth overlap with physical above/below.",
            max_side=960,
            columns=3,
            cell_width=380,
            draw_grid=True,
        )
        add_endpoint_board(draw_grid=True)
        return images

    if question_type == "relative_depth":
        add_series_board(
            label="depth_relation_grid",
            times=even_times(start, end, 20),
            prefix="D",
            title="Depth/proximity grid: near/far/front/behind",
            note="Grid timeline for co-visible depth, occlusion, and near/far relation checks.",
            max_side=820,
            columns=5,
            cell_width=310,
            draw_grid=True,
        )
        add_series_board(
            label="depth_contact_lower_band",
            times=even_times(start, end, 12),
            prefix="F",
            title="Depth contact/floor band",
            note="Lower-scene crop for floor contact, support points, occlusion layers, and proximity cues.",
            crop_pct=(0, 35, 100, 100),
            max_side=900,
            columns=4,
            cell_width=360,
            draw_grid=True,
        )
        add_endpoint_board(draw_grid=True)
        return images

    if question_type == "physical_environment":
        add_series_board(
            label="environment_context_timeline",
            times=even_times(start, end, 18),
            prefix="P",
            title="Physical/environment context timeline",
            note="Wide timeline for objects, setting, terrain, lighting, containers, surfaces, and affordances.",
            max_side=820,
            columns=6,
            cell_width=300,
        )
        add_series_board(
            label="environment_key_states",
            times=even_times(start, end, 6),
            prefix="K",
            title="Environment key states",
            note="High-resolution keyframes for stable scene context and physical state changes.",
            max_side=960,
            columns=3,
            cell_width=380,
        )
        add_series_board(
            label="environment_surface_band",
            times=even_times(start, end, 10),
            prefix="S",
            title="Environment surface/support band",
            note="Lower-scene crop for support, containment, floor/terrain, and object placement cues.",
            crop_pct=(0, 35, 100, 100),
            max_side=900,
            columns=5,
            cell_width=330,
        )
        add_endpoint_board(draw_grid=False)
        return images

    if question_type == "social_interaction":
        add_series_board(
            label="social_participant_timeline",
            times=even_times(start, end, 24),
            prefix="S",
            title="Social interaction timeline",
            note="Timeline for participant identity, approach/avoidance, body orientation, turn-taking, and shared attention.",
            max_side=760,
            columns=6,
            cell_width=300,
        )
        add_series_board(
            label="social_upper_body_focus",
            times=even_times(start, end, 18),
            prefix="U",
            title="Social upper-body/gaze focus",
            note="Upper-frame crop for faces, gaze, posture, and interaction direction.",
            crop_pct=(0, 0, 100, 72),
            max_side=900,
            columns=6,
            cell_width=300,
        )
        add_series_board(
            label="social_center_interaction",
            times=even_times(start, end, 12),
            prefix="I",
            title="Social central interaction region",
            note="Central crop for gestures, proximity, touch, and who is interacting with whom.",
            crop_pct=(8, 5, 92, 96),
            max_side=900,
            columns=4,
            cell_width=360,
        )
        add_endpoint_board(draw_grid=False)
        return images

    if question_type == "viewpoint_visibility":
        add_series_board(
            label="visibility_line_of_sight_grid",
            times=even_times(start, end, 20),
            prefix="Y",
            title="Viewpoint/visibility grid",
            note="Grid timeline for camera visibility, named-viewpoint visibility, occluders, and line of sight.",
            max_side=820,
            columns=5,
            cell_width=310,
            draw_grid=True,
        )
        add_series_board(
            label="visibility_upper_body_focus",
            times=even_times(start, end, 18),
            prefix="U",
            title="Visibility upper-body/line-of-sight focus",
            note="Upper-frame crop for head/eye/body orientation and occlusion of sight lines.",
            crop_pct=(0, 0, 100, 75),
            max_side=900,
            columns=6,
            cell_width=300,
            draw_grid=True,
        )
        add_endpoint_board(draw_grid=True)
        return images

    add_series_board(
        label=f"{question_type}_general_timeline",
        times=even_times(start, end, 20),
        prefix="R",
        title=f"{question_type}: general evidence timeline",
        note="General timeline for fallback reasoning.",
        max_side=760,
        columns=5,
        cell_width=300,
    )
    return images


def build_requested_evidence(
    video_path: Path,
    question: dict[str, Any],
    evidence_dir: Path,
    requests: list[dict[str, Any]],
    *,
    limit: int = 6,
    jpeg_quality: int = 84,
) -> list[EvidenceImage]:
    duration = ffprobe_duration(video_path)
    start, end = question_window(question, duration)
    images: list[EvidenceImage] = []
    for idx, request in enumerate(requests[:limit], 1):
        kind = str(request.get("kind") or "").lower()
        label = str(request.get("label") or f"request_{idx}")[:48]
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label).strip("_") or f"request_{idx}"
        if kind in {"frames", "frame_times"}:
            times = [as_float(t, start) for t in request.get("times_sec") or request.get("times") or []][:24]
            times = [min(max(t, start), end) for t in times]
            if not times:
                continue
            frames = extract_time_series(
                video_path,
                evidence_dir / f"{idx:02d}_{safe_label}_frames",
                times,
                prefix=f"Q{idx}",
                max_side=820,
                jpeg_quality=jpeg_quality,
            )
            images.append(
                EvidenceImage(
                    safe_label,
                    make_contact_sheet(
                        frames,
                        evidence_dir / f"{idx:02d}_{safe_label}.jpg",
                        columns=min(6, max(1, len(frames))),
                        cell_width=320,
                        title=f"Requested frames: {label}",
                        jpeg_quality=jpeg_quality,
                    ),
                    str(request.get("reason") or "Planner-requested exact frames."),
                )
            )
        elif kind in {"dense_sheet", "time_window"}:
            req_start = min(max(as_float(request.get("start_sec"), start), start), end)
            req_end = min(max(as_float(request.get("end_sec"), end), req_start + 0.1), end)
            fps = min(max(as_float(request.get("fps"), 1.5), 0.2), 3.0)
            count = min(36, max(6, round((req_end - req_start) * fps)))
            frames = extract_time_series(
                video_path,
                evidence_dir / f"{idx:02d}_{safe_label}_dense",
                even_times(req_start, req_end, count),
                prefix=f"W{idx}",
                max_side=760,
                jpeg_quality=jpeg_quality,
            )
            images.append(
                EvidenceImage(
                    safe_label,
                    make_contact_sheet(
                        frames,
                        evidence_dir / f"{idx:02d}_{safe_label}_dense.jpg",
                        columns=6,
                        cell_width=300,
                        title=f"Requested window: {label} {req_start:.1f}-{req_end:.1f}s",
                        jpeg_quality=jpeg_quality,
                    ),
                    str(request.get("reason") or "Planner-requested time-window sweep."),
                )
            )
        elif kind == "crop":
            t = min(max(as_float(request.get("time_sec"), (start + end) / 2), start), end)
            box = normalize_pct_box(request.get("box_pct") or request.get("box"))
            path = extract_frame(
                video_path,
                t,
                evidence_dir / f"{idx:02d}_{safe_label}_crop.jpg",
                label=f"{label} t={t:.1f}s box={tuple(round(x) for x in box)}",
                crop_pct=box,
                max_side=1100,
                jpeg_quality=jpeg_quality,
            )
            images.append(
                EvidenceImage(
                    safe_label,
                    path,
                    str(request.get("reason") or "Planner-requested high-resolution crop."),
                )
            )
    return images

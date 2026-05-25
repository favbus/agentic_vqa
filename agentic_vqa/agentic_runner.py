from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from api_client import RelayClient, extract_json_object, merged_env, required_env, response_text
from evidence import (
    EvidenceImage,
    build_policy_boards,
    build_requested_evidence,
    build_scout_board,
    classify_question,
    clean_dir,
    image_to_data_uri,
    load_json,
    resolve_video_path,
    save_json,
)
from prompts import answer_prompt, scout_prompt, verify_prompt


TYPE_ALIASES = {
    "causalandmotivationalreasoning": "causal_motivational",
    "causalmotivational": "causal_motivational",
    "causal": "causal_motivational",
    "inferredcounting": "inferred_counting",
    "counting": "inferred_counting",
    "count": "inferred_counting",
    "lateralspatialreasoning": "lateral_spatial",
    "lateralspatial": "lateral_spatial",
    "spatialleft/right": "lateral_spatial",
    "motionandtrajectorydynamics": "motion_trajectory",
    "motiontrajectory": "motion_trajectory",
    "motion": "motion_trajectory",
    "physicalandenvironmentalcontext": "physical_environment",
    "physicalenvironment": "physical_environment",
    "environment": "physical_environment",
    "relativedepthandproximity": "relative_depth",
    "relativedepth": "relative_depth",
    "depth": "relative_depth",
    "socialinteractionandrelationships": "social_interaction",
    "socialinteraction": "social_interaction",
    "social": "social_interaction",
    "verticalspatialreasoning": "vertical_spatial",
    "verticalspatial": "vertical_spatial",
    "vertical": "vertical_spatial",
    "viewpointandvisibility": "viewpoint_visibility",
    "viewpointvisibility": "viewpoint_visibility",
    "visibility": "viewpoint_visibility",
    "other": "other",
}


def normalize_question_type(value: Any, fallback: str) -> str:
    raw = str(value or "").strip().lower()
    normalized = "".join(ch for ch in raw if ch.isalnum() or ch in "_/")
    if normalized in TYPE_ALIASES:
        return TYPE_ALIASES[normalized]
    snake = normalized.replace("/", "_")
    allowed = {
        "causal_motivational",
        "inferred_counting",
        "lateral_spatial",
        "motion_trajectory",
        "physical_environment",
        "relative_depth",
        "social_interaction",
        "vertical_spatial",
        "viewpoint_visibility",
        "other",
    }
    if snake in allowed:
        return snake
    return fallback


def should_keep_motion_heuristic(question: dict[str, Any], heuristic_type: str, scout_type: str) -> bool:
    if heuristic_type != "motion_trajectory" or scout_type == "motion_trajectory":
        return False
    text = str(question.get("question_text") or "").lower()
    motion_priority_terms = [
        "from first to last",
        "in order",
        "order of",
        "arrange in order",
        "sequence",
        "how many flips",
        "how many times",
        "number of times",
        "direction",
        "moving",
        "facing",
        "orientation",
    ]
    return any(term in text for term in motion_priority_terms)


def find_question_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("video_*.json"))
    if not files:
        files = sorted(input_dir.rglob("video_*.json"))
    if not files:
        raise FileNotFoundError(f"No video_*.json files found under {input_dir}")
    return files


def load_question(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty list in {path}")
    if len(data) != 1:
        raise ValueError(f"Agentic runner expects one question per file; got {len(data)} in {path}")
    return data[0]


def input_items(text: str, images: list[EvidenceImage]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    for image in images:
        content.append({"type": "input_text", "text": f"Evidence image: {image.label}. {image.note}"})
        content.append({"type": "input_image", "image_url": image_to_data_uri(image.path)})
    return [{"role": "user", "content": content}]


def call_json(
    client: RelayClient,
    text: str,
    images: list[EvidenceImage],
    *,
    max_output_tokens: int,
    temperature: float,
    retries: int,
    retry_sleep: float,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            payload = client.create(
                input_items(text, images),
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                retries=1,
                retry_sleep=retry_sleep,
            )
            raw = response_text(payload)
            return extract_json_object(raw), raw, payload
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"Could not get valid JSON after {retries} attempts: {last_error}")


def normalize_answer_choice(value: Any, options: dict[str, Any]) -> str:
    allowed = {str(key).upper() for key in options}
    answer = str(value or "").strip().upper()
    if answer.startswith("OPTION "):
        answer = answer.split()[-1]
    if answer not in allowed:
        raise ValueError(f"Invalid answer_choice={answer!r}; allowed={sorted(allowed)}")
    return answer


def image_notes(images: list[EvidenceImage]) -> list[dict[str, str]]:
    return [{"label": image.label, "path": str(image.path), "note": image.note} for image in images]


def process_question(
    index: int,
    total: int,
    question_file: Path,
    args: argparse.Namespace,
    client: RelayClient | None,
) -> tuple[str, str]:
    qid = question_file.stem[len("video_") :]
    out_file = args.out_dir / "clips" / f"{qid}.json"
    if out_file.exists() and not args.force:
        return "skip", f"[{index}/{total}] skip {qid}"

    question = load_question(question_file)
    video_path = resolve_video_path(args.video_dir, qid)
    clip_dir = args.out_dir / "work" / qid
    if args.force:
        clean_dir(clip_dir)
    else:
        clip_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = clip_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    scout = build_scout_board(
        video_path,
        question,
        evidence_dir,
        scout_frames=args.scout_frames,
        jpeg_quality=args.jpeg_quality,
    )
    heuristic_type = classify_question(question)

    if args.dry_run:
        policy_images = build_policy_boards(
            video_path,
            question,
            evidence_dir,
            heuristic_type,
            jpeg_quality=args.jpeg_quality,
        )
        result = {
            "question_id": qid,
            "question_file": str(question_file),
            "video_path": str(video_path),
            "dry_run": True,
            "heuristic_type": heuristic_type,
            "evidence": image_notes([scout] + policy_images),
        }
        save_json(out_file, result)
        return "dry", f"[{index}/{total}] dry {qid} evidence={len(policy_images) + 1} boards"

    if client is None:
        raise RuntimeError("client is required unless --dry-run is set")

    scout_obj, scout_raw, _ = call_json(
        client,
        scout_prompt(question, heuristic_type),
        [scout],
        max_output_tokens=args.scout_tokens,
        temperature=args.temperature,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    question_type = normalize_question_type(scout_obj.get("question_type"), heuristic_type)
    if should_keep_motion_heuristic(question, heuristic_type, question_type):
        question_type = heuristic_type
    if question_type == "other" and heuristic_type != "other":
        question_type = heuristic_type
    scout_obj["question_type"] = question_type
    scout_obj["heuristic_type"] = heuristic_type

    policy_images = build_policy_boards(
        video_path,
        question,
        evidence_dir,
        question_type,
        jpeg_quality=args.jpeg_quality,
    )
    requested_images = build_requested_evidence(
        video_path,
        question,
        evidence_dir,
        scout_obj.get("evidence_requests") or [],
        limit=args.request_limit,
        jpeg_quality=args.jpeg_quality,
    )
    answer_images = ([scout] + policy_images + requested_images)[: args.max_images]

    answer_obj, answer_raw, _ = call_json(
        client,
        answer_prompt(question, scout_obj, image_notes(answer_images)),
        answer_images,
        max_output_tokens=args.answer_tokens,
        temperature=args.temperature,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    answer_obj["answer_choice"] = normalize_answer_choice(answer_obj.get("answer_choice"), question.get("options") or {})

    verifier_extra: list[EvidenceImage] = []
    if args.max_rounds >= 3 and answer_obj.get("needs_verification"):
        verifier_extra = build_requested_evidence(
            video_path,
            question,
            evidence_dir,
            answer_obj.get("verification_requests") or [],
            limit=max(2, args.request_limit // 2),
            jpeg_quality=args.jpeg_quality,
        )
    verify_images = (answer_images + verifier_extra)[: args.max_images]
    if args.max_rounds >= 3:
        final_obj, verify_raw, _ = call_json(
            client,
            verify_prompt(question, scout_obj, answer_obj, image_notes(verify_images)),
            verify_images,
            max_output_tokens=args.verify_tokens,
            temperature=args.temperature,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        final_raw = verify_raw
    else:
        final_obj = {
            "question_id": qid,
            "answer_choice": answer_obj["answer_choice"],
            "confidence": answer_obj.get("confidence", 0.0),
            "final_rationale": "; ".join(str(x) for x in answer_obj.get("evidence_ledger") or [])[:500],
            "changed_from_draft": False,
            "audit_notes": [],
        }
        final_raw = ""
    final_obj["answer_choice"] = normalize_answer_choice(final_obj.get("answer_choice"), question.get("options") or {})

    elapsed = time.time() - started
    result = {
        "question_id": qid,
        "question_file": str(question_file),
        "video_path": str(video_path),
        "elapsed_sec": round(elapsed, 3),
        "question": question,
        "heuristic_type": heuristic_type,
        "scout": scout_obj,
        "draft_answer": answer_obj,
        "final": final_obj,
        "answer_choice": final_obj["answer_choice"],
        "evidence": image_notes(verify_images),
        "raw_text": {
            "scout": scout_raw,
            "answer": answer_raw,
            "verify": final_raw,
        },
        "model": args.model,
        "base_url": args.base_url,
        "api_style": args.api_style,
        "reasoning_effort": args.reasoning_effort,
    }
    save_json(out_file, result)
    return "done", f"[{index}/{total}] done {qid} -> {final_obj['answer_choice']} in {elapsed:.1f}s"


def export_submission(out_dir: Path) -> Path:
    rows = []
    for path in sorted((out_dir / "clips").glob("*.json")):
        data = load_json(path)
        if data.get("dry_run"):
            continue
        answer = data.get("answer_choice") or (data.get("final") or {}).get("answer_choice")
        if answer:
            rows.append({"question_id": data.get("question_id") or path.stem, "answer_choice": str(answer).upper()})
    submission = out_dir / "submission.json"
    save_json(submission, rows)
    return submission


def parse_args() -> argparse.Namespace:
    env = merged_env()
    parser = argparse.ArgumentParser(description="Agentic multi-round VRR-QA runner")
    parser.add_argument("--input-dir", type=Path, default=Path("test题目合集/inputs"))
    parser.add_argument("--video-dir", type=Path, default=Path("test题目合集/videos"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/agentic_vqa/latest"))
    parser.add_argument("--base-url", default=env.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--model", default=env.get("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--api-style", choices=["auto", "responses", "chat"], default=(env.get("OPENAI_API_STYLE") or "auto").strip().lower())
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"], default=(env.get("OPENAI_REASONING_EFFORT") or "high").strip().lower())
    parser.add_argument("--request-timeout", type=float, default=float(env.get("VRR_API_REQUEST_TIMEOUT", "240") or "240"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="Optional question ids to run.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build evidence boards without calling the API.")
    parser.add_argument("--max-rounds", type=int, default=3, choices=[2, 3])
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--request-limit", type=int, default=6)
    parser.add_argument("--scout-frames", type=int, default=12)
    parser.add_argument("--jpeg-quality", type=int, default=84)
    parser.add_argument("--scout-tokens", type=int, default=1800)
    parser.add_argument("--answer-tokens", type=int, default=2600)
    parser.add_argument("--verify-tokens", type=int, default=2200)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    args = parser.parse_args()

    if args.api_style == "auto" and not args.base_url and not args.dry_run:
        required_env(env, "OPENAI_BASE_URL")
    return args


def main() -> None:
    args = parse_args()
    env = merged_env()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "clips").mkdir(parents=True, exist_ok=True)
    client = None
    if not args.dry_run:
        client = RelayClient(
            base_url=args.base_url or required_env(env, "OPENAI_BASE_URL"),
            api_key=required_env(env, "OPENAI_API_KEY"),
            model=args.model,
            api_style=args.api_style,
            request_timeout=args.request_timeout,
            reasoning_effort=args.reasoning_effort,
        )

    question_files = find_question_files(args.input_dir)
    if args.only:
        wanted = {qid.strip() for qid in args.only}
        question_files = [path for path in question_files if path.stem[len("video_") :] in wanted]
    if args.limit is not None:
        question_files = question_files[: args.limit]
    total = len(question_files)
    print(f"Found {total} question files")

    items = [(idx, total, path, args, client) for idx, path in enumerate(question_files, 1)]
    if args.workers <= 1:
        for item in items:
            _, message = process_question(*item)
            print(message, flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_question, *item) for item in items]
            for future in as_completed(futures):
                _, message = future.result()
                print(message, flush=True)

    submission = export_submission(args.out_dir)
    print(f"submission written: {submission}")


if __name__ == "__main__":
    main()

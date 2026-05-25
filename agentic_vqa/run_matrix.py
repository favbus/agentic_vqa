from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_profiles(path: Path) -> list[dict[str, Any]]:
    profiles = load_json(path)
    if not isinstance(profiles, list):
        raise ValueError(f"Expected a list of profiles in {path}")
    for profile in profiles:
        if not profile.get("name"):
            raise ValueError(f"Profile without name in {path}: {profile}")
        if not profile.get("model"):
            raise ValueError(f"Profile without model in {path}: {profile}")
    return profiles


def selected_profiles(all_profiles: list[dict[str, Any]], names: list[str] | None) -> list[dict[str, Any]]:
    if not names or names == ["all"]:
        return all_profiles
    by_name = {str(profile["name"]): profile for profile in all_profiles}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown profile(s): {missing}. Available: {sorted(by_name)}")
    return [by_name[name] for name in names]


def runner_command(args: argparse.Namespace, profile: dict[str, Any], out_dir: Path) -> list[str]:
    root = repo_root()
    cmd = [
        sys.executable,
        str(root / "agentic_vqa" / "agentic_runner.py"),
        "--input-dir",
        str(args.input_dir),
        "--video-dir",
        str(args.video_dir),
        "--out-dir",
        str(out_dir),
        "--model",
        str(profile["model"]),
        "--workers",
        str(args.workers),
        "--max-rounds",
        str(profile.get("max_rounds", args.max_rounds)),
        "--max-images",
        str(profile.get("max_images", args.max_images)),
    ]
    for key, flag in (
        ("scout_model", "--scout-model"),
        ("answer_model", "--answer-model"),
        ("verify_model", "--verify-model"),
    ):
        if profile.get(key):
            cmd.extend([flag, str(profile[key])])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.only:
        cmd.append("--only")
        cmd.extend(args.only)
    if args.force:
        cmd.append("--force")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def score_command(args: argparse.Namespace, out_dir: Path) -> list[str] | None:
    if args.dry_run or not args.gold:
        return None
    gold = Path(args.gold)
    if not gold.exists():
        return None
    root = repo_root()
    return [
        sys.executable,
        str(root / "agentic_vqa" / "score_agentic.py"),
        "--pred",
        str(out_dir / "submission.json"),
        "--gold",
        str(gold),
        "--input-dir",
        str(args.input_dir),
        "--output",
        str(out_dir / "score.json"),
    ]


def run_command(cmd: list[str], cwd: Path) -> int:
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run reproducible model/profile matrices for Agentic VQA.")
    parser.add_argument("--profiles-file", type=Path, default=root / "configs" / "model_profiles.json")
    parser.add_argument("--profiles", nargs="*", default=["gpt55_strong"], help="Profile names to run, or 'all'.")
    parser.add_argument("--input-dir", type=Path, default=Path("test题目合集/inputs"))
    parser.add_argument("--video-dir", type=Path, default=Path("test题目合集/videos"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/model_matrix"))
    parser.add_argument("--gold", type=Path, default=None, help="Optional gold answers file for post-run scoring.")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    profiles = selected_profiles(load_profiles(args.profiles_file), args.profiles)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    matrix_dir = args.out_root / stamp
    matrix_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for profile in profiles:
        profile_out = matrix_dir / str(profile["name"])
        cmd = runner_command(args, profile, profile_out)
        started = time.time()
        code = run_command(cmd, root)
        elapsed = time.time() - started
        score = None
        score_cmd = score_command(args, profile_out)
        if code == 0 and score_cmd:
            score_code = run_command(score_cmd, root)
            if score_code == 0 and (profile_out / "score.json").exists():
                score = load_json(profile_out / "score.json")
        results.append(
            {
                "profile": profile,
                "out_dir": str(profile_out),
                "returncode": code,
                "elapsed_sec": round(elapsed, 3),
                "score": score,
                "command": cmd,
            }
        )
    save_json(
        matrix_dir / "matrix_summary.json",
        {
            "created_at": stamp,
            "profiles_file": str(args.profiles_file),
            "input_dir": str(args.input_dir),
            "video_dir": str(args.video_dir),
            "gold": str(args.gold) if args.gold else None,
            "results": results,
        },
    )
    print(f"matrix summary written: {matrix_dir / 'matrix_summary.json'}")


if __name__ == "__main__":
    main()


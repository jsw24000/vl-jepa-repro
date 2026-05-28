"""Convert downloaded MSR-VTT JSON files into this project's CSV manifests.

The Hugging Face MSR-VTT files store one JSON object per video. This project
trains on one video-caption pair per CSV row, so train captions are expanded
into multiple rows when a video has multiple captions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "msr-vtt"


def parse_args() -> argparse.Namespace:
    dataset_root = _default_dataset_root()
    parser = argparse.ArgumentParser(
        description="Prepare train/val CSV manifests from downloaded MSR-VTT JSON files."
    )
    parser.add_argument("--dataset-root", type=Path, default=dataset_root)
    parser.add_argument("--train-json", type=Path, default=None)
    parser.add_argument("--val-json", type=Path, default=None)
    parser.add_argument("--videos-root", type=Path, default=None)
    parser.add_argument(
        "--train-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "train_manifest.csv",
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "val_manifest.csv",
    )
    parser.add_argument(
        "--caption-mode",
        choices=("all", "first"),
        default="all",
        help="Use every caption for a video, or only the first caption.",
    )
    parser.add_argument(
        "--path-style",
        choices=("relative", "absolute"),
        default="relative",
        help="Write video paths relative to the project root, or absolute paths.",
    )
    parser.add_argument(
        "--strict-videos",
        action="store_true",
        help="Fail if any referenced video file cannot be found under --videos-root.",
    )
    return parser.parse_args()


def load_json_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"MSR-VTT JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


def captions_from_record(record: dict, caption_mode: str) -> list[str]:
    captions = record.get("caption", "")
    if isinstance(captions, str):
        values = [captions]
    elif isinstance(captions, list):
        values = [str(item) for item in captions]
    else:
        values = [str(captions)]

    values = [caption.strip() for caption in values if caption and str(caption).strip()]
    if caption_mode == "first":
        return values[:1]
    return values


def video_filename_from_record(record: dict) -> str:
    filename = record.get("video")
    if filename:
        return str(filename)
    video_id = record.get("video_id")
    if video_id:
        return f"{video_id}.mp4"
    raise ValueError(f"Record has neither 'video' nor 'video_id': {record}")


def index_video_files(videos_root: Path) -> dict[str, Path]:
    if not videos_root.exists():
        return {}
    index: dict[str, Path] = {}
    for path in videos_root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, path)
    return index


def format_path(path: Path, path_style: str) -> str:
    path = path.resolve()
    if path_style == "absolute":
        return str(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def make_rows(
    records: Iterable[dict],
    split: str,
    videos_root: Path,
    video_index: dict[str, Path],
    caption_mode: str,
    path_style: str,
) -> tuple[list[dict[str, str]], set[str]]:
    rows: list[dict[str, str]] = []
    missing_videos: set[str] = set()

    for record in records:
        filename = video_filename_from_record(record)
        video_path = video_index.get(filename, videos_root / filename)
        if not video_path.exists():
            missing_videos.add(filename)
        for caption in captions_from_record(record, caption_mode):
            rows.append(
                {
                    "video_path": format_path(video_path, path_style),
                    "caption": caption,
                    "split": split,
                }
            )
    return rows, missing_videos


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "caption", "split"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    train_json = args.train_json or dataset_root / "msrvtt_train_7k.json"
    val_json = args.val_json or dataset_root / "msrvtt_test_1k.json"
    videos_root = (args.videos_root or dataset_root / "videos").resolve()

    train_records = load_json_records(train_json)
    val_records = load_json_records(val_json)
    video_index = index_video_files(videos_root)

    train_rows, missing_train = make_rows(
        train_records,
        split="train",
        videos_root=videos_root,
        video_index=video_index,
        caption_mode=args.caption_mode,
        path_style=args.path_style,
    )
    val_rows, missing_val = make_rows(
        val_records,
        split="val",
        videos_root=videos_root,
        video_index=video_index,
        caption_mode=args.caption_mode,
        path_style=args.path_style,
    )

    missing = missing_train | missing_val
    if missing and args.strict_videos:
        examples = ", ".join(sorted(missing)[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} video files under {videos_root}. Examples: {examples}"
        )

    write_manifest(args.train_out, train_rows)
    write_manifest(args.val_out, val_rows)

    print(f"Wrote {len(train_rows):,} rows to {args.train_out}")
    print(f"Wrote {len(val_rows):,} rows to {args.val_out}")
    if missing:
        examples = ", ".join(sorted(missing)[:10])
        print(
            f"Warning: {len(missing):,} referenced videos were not found under "
            f"{videos_root}. Examples: {examples}"
        )
        print("Download/extract MSRVTT_Videos.zip, or pass --videos-root to its mp4 folder.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare a validated YOLO-format road-hazard dataset for RoadEyeQ.

The script copies raw image/label pairs into a train/validation/test structure,
validates YOLO bounding-box labels, writes a dataset manifest, and generates a
Ultralytics-compatible data.yaml file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ("pothole", "road_crack", "flooded_surface")


@dataclass(frozen=True)
class DatasetRecord:
    """A validated image and annotation pair ready for splitting."""

    image_path: Path
    label_path: Path
    relative_image_path: Path
    annotation_count: int
    class_counts: Counter[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and prepare a YOLO road-hazard dataset for RoadEyeQ."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing raw images. Nested directories are supported.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Directory containing YOLO .txt labels matching the image structure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the prepared dataset.",
    )
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help=(
            "Comma-separated class names in YOLO class-id order. "
            f"Default: {','.join(DEFAULT_CLASSES)}"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help=(
            "Include images with no matching label file as negative samples. "
            "Empty .txt label files will be generated for them."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report results without copying any files.",
    )
    return parser.parse_args()


def parse_classes(value: str) -> list[str]:
    classes = [item.strip() for item in value.split(",") if item.strip()]
    if not classes:
        raise ValueError("At least one class name must be provided with --classes.")
    if len(classes) != len(set(classes)):
        raise ValueError("Class names must be unique.")
    return classes


def validate_ratios(train_ratio: float, val_ratio: float) -> tuple[float, float, float]:
    if not (0 < train_ratio < 1):
        raise ValueError("--train-ratio must be greater than 0 and less than 1.")
    if not (0 <= val_ratio < 1):
        raise ValueError("--val-ratio must be between 0 (inclusive) and 1 (exclusive).")

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError("--train-ratio and --val-ratio must sum to 1 or less.")
    return train_ratio, val_ratio, test_ratio


def iter_images(images_root: Path) -> Iterable[Path]:
    for path in images_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def validate_yolo_label(label_path: Path, class_count: int) -> tuple[int, Counter[int], list[str]]:
    """Validate normalized YOLO labels and return count, class distribution, errors."""
    annotation_count = 0
    class_counts: Counter[int] = Counter()
    errors: list[str] = []

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return 0, class_counts, ["Label file is not valid UTF-8 text."]
    except OSError as exc:
        return 0, class_counts, [f"Unable to read label file: {exc}"]

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        values = line.split()
        if len(values) != 5:
            errors.append(f"Line {line_number}: expected 5 values, found {len(values)}.")
            continue

        try:
            class_id = int(values[0])
        except ValueError:
            errors.append(f"Line {line_number}: class id must be an integer.")
            continue

        if not 0 <= class_id < class_count:
            errors.append(
                f"Line {line_number}: class id {class_id} is outside 0-{class_count - 1}."
            )
            continue

        try:
            x_center, y_center, width, height = (float(value) for value in values[1:])
        except ValueError:
            errors.append(f"Line {line_number}: box values must be numeric.")
            continue

        coordinates = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in coordinates):
            errors.append(f"Line {line_number}: box values must be finite numbers.")
            continue
        if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
            errors.append(f"Line {line_number}: center coordinates must be between 0 and 1.")
            continue
        if not 0 < width <= 1 or not 0 < height <= 1:
            errors.append(f"Line {line_number}: width and height must be greater than 0 and at most 1.")
            continue

        annotation_count += 1
        class_counts[class_id] += 1

    return annotation_count, class_counts, errors


def create_output_name(relative_path: Path) -> str:
    """Create a deterministic, collision-resistant filename for flattened output."""
    readable_stem = "__".join(relative_path.with_suffix("").parts)
    readable_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", readable_stem).strip("_") or "image"
    digest = hashlib.sha1(str(relative_path).encode("utf-8")).hexdigest()[:8]
    return f"{readable_stem}_{digest}{relative_path.suffix.lower()}"


def allocate_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    """Allocate deterministic split counts while keeping the requested proportions close."""
    raw_counts = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw_counts]
    remaining = total - sum(counts)

    order = sorted(
        range(len(ratios)),
        key=lambda index: (raw_counts[index] - counts[index], ratios[index]),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1

    positive_splits = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if total >= len(positive_splits):
        for index in positive_splits:
            if counts[index] == 0:
                donor = max(
                    (candidate for candidate in range(len(counts)) if counts[candidate] > 1),
                    key=lambda candidate: counts[candidate],
                    default=None,
                )
                if donor is not None:
                    counts[donor] -= 1
                    counts[index] += 1

    return counts


def split_records(
    records: list[DatasetRecord], ratios: tuple[float, float, float], seed: int
) -> dict[str, list[DatasetRecord]]:
    shuffled_records = records[:]
    random.Random(seed).shuffle(shuffled_records)
    train_count, val_count, test_count = allocate_counts(len(records), ratios)

    return {
        "train": shuffled_records[:train_count],
        "val": shuffled_records[train_count : train_count + val_count],
        "test": shuffled_records[train_count + val_count : train_count + val_count + test_count],
    }


def write_data_yaml(output_dir: Path, classes: list[str]) -> None:
    lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(classes))
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    output_dir: Path,
    report: dict,
    manifest_rows: list[dict[str, str]],
) -> None:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with (reports_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [
            "split", "source_image", "source_label", "output_image", "annotations"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)


def prepare_dataset(args: argparse.Namespace) -> dict:
    images_root = args.images.resolve()
    labels_root = args.labels.resolve()
    output_dir = args.output_dir.resolve()
    classes = parse_classes(args.classes)
    ratios = validate_ratios(args.train_ratio, args.val_ratio)

    if not images_root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_root}")
    if not labels_root.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {labels_root}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to continue."
        )

    valid_records: list[DatasetRecord] = []
    missing_labels: list[str] = []
    invalid_labels: dict[str, list[str]] = {}
    source_images = sorted(iter_images(images_root))

    for image_path in source_images:
        relative_image_path = image_path.relative_to(images_root)
        label_path = labels_root / relative_image_path.with_suffix(".txt")

        if not label_path.exists():
            if args.include_unlabeled:
                valid_records.append(
                    DatasetRecord(
                        image_path=image_path,
                        label_path=label_path,
                        relative_image_path=relative_image_path,
                        annotation_count=0,
                        class_counts=Counter(),
                    )
                )
            else:
                missing_labels.append(relative_image_path.as_posix())
            continue

        annotation_count, class_counts, errors = validate_yolo_label(label_path, len(classes))
        if errors:
            invalid_labels[relative_image_path.as_posix()] = errors
            continue

        valid_records.append(
            DatasetRecord(
                image_path=image_path,
                label_path=label_path,
                relative_image_path=relative_image_path,
                annotation_count=annotation_count,
                class_counts=class_counts,
            )
        )

    splits = split_records(valid_records, ratios, args.seed)
    all_class_counts: Counter[int] = Counter()
    for record in valid_records:
        all_class_counts.update(record.class_counts)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_images_found": len(source_images),
        "valid_image_label_pairs": len(valid_records),
        "missing_label_files": missing_labels,
        "invalid_label_files": invalid_labels,
        "classes": {str(index): name for index, name in enumerate(classes)},
        "class_annotation_counts": {
            classes[index]: all_class_counts.get(index, 0) for index in range(len(classes))
        },
        "split_image_counts": {split_name: len(records) for split_name, records in splits.items()},
        "configuration": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": ratios[2],
            "seed": args.seed,
            "include_unlabeled": args.include_unlabeled,
        },
    }

    if args.dry_run:
        return report

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []

    for split_name, records in splits.items():
        image_destination_dir = output_dir / "images" / split_name
        label_destination_dir = output_dir / "labels" / split_name
        image_destination_dir.mkdir(parents=True, exist_ok=True)
        label_destination_dir.mkdir(parents=True, exist_ok=True)

        for record in records:
            output_image_name = create_output_name(record.relative_image_path)
            output_image_path = image_destination_dir / output_image_name
            output_label_path = label_destination_dir / f"{Path(output_image_name).stem}.txt"

            shutil.copy2(record.image_path, output_image_path)
            if record.label_path.exists():
                shutil.copy2(record.label_path, output_label_path)
            else:
                output_label_path.write_text("", encoding="utf-8")

            manifest_rows.append(
                {
                    "split": split_name,
                    "source_image": record.relative_image_path.as_posix(),
                    "source_label": (
                        record.relative_image_path.with_suffix(".txt").as_posix()
                        if record.label_path.exists()
                        else ""
                    ),
                    "output_image": (Path("images") / split_name / output_image_name).as_posix(),
                    "annotations": str(record.annotation_count),
                }
            )

    write_data_yaml(output_dir, classes)
    write_report(output_dir, report, manifest_rows)
    return report


def main() -> int:
    args = parse_args()
    try:
        report = prepare_dataset(args)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("\nDry run complete. No files were copied.")
    else:
        print("\nDataset preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

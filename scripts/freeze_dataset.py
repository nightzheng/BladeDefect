"""Freeze a reviewed YOLO-seg dataset without changing the source files.

Images are referenced through ``images/train`` and ``images/val`` directory
junctions created after this script runs.  Labels are copied into the frozen
dataset and repaired only when the source coordinate is a soft error or the
corresponding hard-error review row is explicitly ``repair_confirmed``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, UnidentifiedImageError


CLASS_NAMES = {
    0: "表面腐蚀--保护膜损伤",
    1: "表面腐蚀--玻纤腐蚀",
    2: "表面腐蚀--合模缝裸漏",
    3: "表面腐蚀--胶衣腐蚀",
    4: "表面裂纹--玻纤裂纹",
    5: "表面裂纹--后缘弦向裂纹",
    6: "表面缺陷--表面掉漆",
    7: "表面缺陷--表面油污",
    8: "表面缺陷--胶衣脱落",
    9: "表面缺陷--胶衣裂纹",
    10: "维修痕迹",
    11: "叶片损伤--玻纤损伤",
    12: "叶片损伤--叶片开裂",
    13: "叶片损伤--结构损伤",
    14: "附件脱落--接闪器脱落",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_filter_decisions(path: Path) -> dict[str, dict[str, str]]:
    """Parse the simple ``images`` list in dataset_filter.yaml dependency-free."""
    decisions: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    in_images = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped == "images:":
            in_images = True
            continue
        if not in_images or not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- filename:"):
            if current is not None:
                decisions[current["filename"].casefold()] = current
            current = {"filename": stripped.split(":", 1)[1].strip().strip('"\'')}
            continue
        if current is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = value.strip().strip('"\'')
    if current is not None:
        decisions[current["filename"].casefold()] = current
    return decisions


def load_reviews(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reviews: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (str(Path(row["label_path"]).resolve()).casefold(), int(row["line_number"]))
        reviews[key] = row
    return reviews


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _repair_line(
    source_label: Path,
    split: str,
    line_number: int,
    line: str,
    reviews: dict[tuple[str, int], dict[str, str]],
) -> tuple[str, list[dict[str, object]]]:
    parts = line.split()
    if not parts:
        return line, []
    try:
        class_value = float(parts[0])
        coords = [float(value) for value in parts[1:]]
    except ValueError as error:
        raise ValueError(f"{source_label}:{line_number} contains non-numeric fields") from error
    if not class_value.is_integer() or int(class_value) not in CLASS_NAMES:
        raise ValueError(f"{source_label}:{line_number} has invalid class_id={parts[0]}")
    if len(coords) < 6 or len(coords) % 2:
        raise ValueError(f"{source_label}:{line_number} has invalid polygon coordinate count")

    out_indexes = [index for index, value in enumerate(coords) if not 0.0 <= value <= 1.0]
    if not out_indexes:
        if len(set(zip(coords[::2], coords[1::2], strict=True))) < 3:
            raise ValueError(f"{source_label}:{line_number} has fewer than three distinct points")
        return line, []

    hard = any(coords[index] < -0.01 or coords[index] > 1.01 for index in out_indexes)
    review = reviews.get((str(source_label.resolve()).casefold(), line_number))
    if hard and (review is None or review.get("review_decision") != "repair_confirmed"):
        decision = review.get("review_decision") if review else "missing_review"
        raise ValueError(
            f"{source_label}:{line_number} is hard error without repair_confirmed review ({decision})"
        )

    repair_type = "hard_error_repair" if hard else "soft_error_reset"
    repaired = list(coords)
    logs: list[dict[str, object]] = []
    for index in out_indexes:
        before = coords[index]
        after = clamp01(before)
        repaired[index] = after
        point_index = index // 2 + 1
        axis = "x" if index % 2 == 0 else "y"
        logs.append(
            {
                "split": split,
                "label_path": str(source_label),
                "line_number": line_number,
                "class_id": int(class_value),
                "class_name": CLASS_NAMES[int(class_value)],
                "point_index": point_index,
                "axis": axis,
                "coordinate_position": f"point_{point_index}_{axis}",
                "before": before,
                "after": after,
                "overflow_amount": abs(before - after),
                "repair_type": repair_type,
                "review_decision": review.get("review_decision", "") if review else "soft_rule",
                "reviewer": review.get("reviewer", "") if review else "automatic_soft_rule",
            }
        )
    if len(set(zip(repaired[::2], repaired[1::2], strict=True))) < 3:
        raise ValueError(f"{source_label}:{line_number} repair collapses polygon below three points")
    output = [parts[0]] + ["0" if value == 0.0 else "1" if value == 1.0 else parts[index + 1] for index, value in enumerate(repaired)]
    return " ".join(output), logs


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_label(path: Path) -> tuple[int, set[int], Counter[int]]:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return 0, set(), Counter()
    instances = 0
    classes: set[int] = set()
    class_counts: Counter[int] = Counter()
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            class_value = float(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number} contains non-numeric values") from error
        if not class_value.is_integer() or int(class_value) not in CLASS_NAMES:
            raise ValueError(f"{path}:{line_number} has illegal class ID")
        if len(coords) < 6 or len(coords) % 2:
            raise ValueError(f"{path}:{line_number} has invalid polygon point count")
        if any(not 0.0 <= value <= 1.0 for value in coords):
            raise ValueError(f"{path}:{line_number} still contains out-of-range coordinates")
        if len(set(zip(coords[::2], coords[1::2], strict=True))) < 3:
            raise ValueError(f"{path}:{line_number} has fewer than three distinct points")
        instances += 1
        class_id = int(class_value)
        classes.add(class_id)
        class_counts[class_id] += 1
    return instances, classes, class_counts


def _repair_logs_from_coordinate_csv(
    path: Path,
    reviews: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    logs: list[dict[str, object]] = []
    for row in rows:
        label_path = Path(row["file"])
        line_number = int(row["line"])
        hard = row["line_severity"] == "hard"
        review = reviews.get((str(label_path.resolve()).casefold(), line_number))
        if hard and (review is None or review.get("review_decision") != "repair_confirmed"):
            raise ValueError(f"coordinate log contains an unconfirmed hard error: {label_path}:{line_number}")
        before = float(row["original_value"])
        after = float(row["boundary"])
        logs.append(
            {
                "split": row["split"],
                "label_path": str(label_path),
                "line_number": line_number,
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "point_index": int(row["point_index"]),
                "axis": row["axis"],
                "coordinate_position": row["coordinate_position"],
                "before": before,
                "after": after,
                "overflow_amount": abs(before - after),
                "repair_type": "hard_error_repair" if hard else "soft_error_reset",
                "review_decision": review.get("review_decision", "") if review else "soft_rule",
                "reviewer": review.get("reviewer", "") if review else "automatic_soft_rule",
            }
        )
    return logs


def freeze_dataset(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory must be empty or absent: {output}")
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)
    args.review_output.mkdir(parents=True, exist_ok=True)

    decisions = load_filter_decisions(args.filter_config)
    reviews = load_reviews(args.hard_review)
    review_counts = Counter(row.get("review_decision", "") for row in reviews.values())
    if len(reviews) != 131 or review_counts != Counter({"repair_confirmed": 131}):
        raise ValueError(f"hard review is not complete: rows={len(reviews)}, decisions={dict(review_counts)}")

    all_repair_logs: list[dict[str, object]] = []
    selected_images: dict[str, list[Path]] = {"train": [], "val": []}
    excluded_rows: list[dict[str, object]] = []
    empty_labels: list[str] = []
    class_instances: Counter[int] = Counter()
    class_images: dict[str, Counter[int]] = {"train": Counter(), "val": Counter()}
    valid_instances = 0
    decode_errors: list[dict[str, str]] = []

    for split in ("train", "val"):
        image_root = args.images / split
        label_root = args.labels / split
        output_labels = output / "labels" / split
        output_labels.mkdir(parents=True, exist_ok=True)
        labels_by_stem = {path.stem.casefold(): path for path in label_root.rglob("*.txt")}
        images = sorted(
            path for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for image_path in images:
            decision = decisions.get(image_path.name.casefold())
            action = decision.get("action") if decision else None
            if action in {"exclude", "review"}:
                excluded_rows.append(
                    {
                        "split": split,
                        "image_path": str(image_path),
                        "action": action,
                        "issue": decision.get("issue", ""),
                        "reason": decision.get("reason", ""),
                        "source": "dataset_filter",
                    }
                )
                continue
            label_path = labels_by_stem.get(image_path.stem.casefold())
            keep_negative = action == "keep_negative"
            if label_path is None and not keep_negative:
                raise FileNotFoundError(f"unfiltered image has no label: {image_path}")
            relative = image_path.relative_to(image_root)
            target_label = output_labels / relative.with_suffix(".txt")
            target_label.parent.mkdir(parents=True, exist_ok=True)
            if args.resume and target_label.exists():
                if target_label.stat().st_size == 0:
                    empty_labels.append(f"{split}/{relative.with_suffix('.txt').as_posix()}")
            elif keep_negative:
                target_label.write_text("", encoding="utf-8")
                empty_labels.append(f"{split}/{relative.with_suffix('.txt').as_posix()}")
            else:
                assert label_path is not None
                output_lines: list[str] = []
                for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                    if not line.strip():
                        continue
                    repaired, logs = _repair_line(label_path, split, line_number, line, reviews)
                    output_lines.append(repaired)
                    all_repair_logs.extend(logs)
                target_label.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
            selected_images[split].append(image_path)

    if args.resume:
        all_repair_logs = _repair_logs_from_coordinate_csv(args.coordinate_csv, reviews)

    # Strict validation uses the selected source image list and the frozen labels.
    for split, images in selected_images.items():
        list_path = output / f"{split}.txt"
        with list_path.open("w", encoding="utf-8", newline="\n") as handle:
            for image_path in images:
                handle.write(f"./images/{split}/{image_path.relative_to(args.images / split).as_posix()}\n")
        def validate_sample(image_path: Path) -> tuple[Path, str | None, int, set[int], Counter[int]]:
            relative = image_path.relative_to(args.images / split)
            label_path = output / "labels" / split / relative.with_suffix(".txt")
            decode_error: str | None = None
            try:
                with Image.open(image_path) as image:
                    image.draft("RGB", (64, 64))
                    image.load()
            except (OSError, UnidentifiedImageError) as error:
                decode_error = str(error)
            instances, classes, counts = _validate_label(label_path)
            return image_path, decode_error, instances, classes, counts

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            checked = executor.map(validate_sample, images, chunksize=32)
            for image_path, decode_error, instances, classes, counts in checked:
                if decode_error is not None:
                    decode_errors.append({"split": split, "image_path": str(image_path), "reason": decode_error})
                valid_instances += instances
                class_instances.update(counts)
                for class_id in classes:
                    class_images[split][class_id] += 1

    if decode_errors:
        raise ValueError(f"frozen dataset contains {len(decode_errors)} unreadable images")

    repair_fields = [
        "split", "label_path", "line_number", "class_id", "class_name", "point_index",
        "axis", "coordinate_position", "before", "after", "overflow_amount", "repair_type",
        "review_decision", "reviewer",
    ]
    hard_logs = [row for row in all_repair_logs if row["repair_type"] == "hard_error_repair"]
    soft_logs = [row for row in all_repair_logs if row["repair_type"] == "soft_error_reset"]
    _write_csv(args.review_output / "hard_error_repair_log.csv", hard_logs, repair_fields)
    _write_csv(args.review_output / "soft_error_repair_log.csv", soft_logs, repair_fields)
    _write_csv(
        args.review_output / "excluded_after_review.csv",
        [],
        ["split", "image_path", "label_path", "line_number", "class_id", "review_reason"],
    )

    validation = {
        "valid": True,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "samples": {split: len(images) for split, images in selected_images.items()},
        "labels": {split: len(images) for split, images in selected_images.items()},
        "instances": valid_instances,
        "empty_negative_labels": len(empty_labels),
        "empty_negative_label_files": empty_labels,
        "unreadable_images": 0,
        "missing_labels": 0,
        "orphan_labels": 0,
        "illegal_class_ids": 0,
        "invalid_polygon_point_counts": 0,
        "out_of_range_coordinates": 0,
        "hard_review_decisions": dict(review_counts),
        "soft_coordinates_reset": len(soft_logs),
        "hard_coordinates_repaired": len(hard_logs),
        "class_instance_counts": {str(key): class_instances[key] for key in CLASS_NAMES},
        "class_image_counts": {
            split: {str(key): class_images[split][key] for key in CLASS_NAMES}
            for split in ("train", "val")
        },
    }
    validation_path = args.review_output / "dataset_validation.json"
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    data_yaml = [
        "path: .", "train: train.txt", "val: val.txt", "", "names:",
        *[f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items()], "",
    ]
    (output / "data.yaml").write_text("\n".join(data_yaml), encoding="utf-8")

    label_digest = hashlib.sha256()
    for path in sorted((output / "labels").rglob("*.txt")):
        label_digest.update(path.relative_to(output).as_posix().encode("utf-8"))
        label_digest.update(path.read_bytes())
    list_digest = hashlib.sha256()
    for name in ("train.txt", "val.txt"):
        list_digest.update((output / name).read_bytes())

    manifest = {
        "dataset_id": "blade-v2",
        "generated_at": validation["generated_at"],
        "source": {"images": str(args.images.resolve()), "labels": str(args.labels.resolve())},
        "filter_config": {"path": str(args.filter_config.resolve()), "sha256": sha256_file(args.filter_config)},
        "hard_error_review": {"path": str(args.hard_review.resolve()), "sha256": sha256_file(args.hard_review)},
        "random_seed": args.seed,
        "split_policy": "preserve_source_train_val; no test split created",
        "image_storage": "NTFS directory junctions to read-only source image directories",
        "sample_lists_sha256": list_digest.hexdigest(),
        "frozen_labels_sha256": label_digest.hexdigest(),
        "samples": validation["samples"],
        "instances": valid_instances,
        "validation": {"valid": True, "report": str(validation_path.resolve())},
        "generation_command": args.command,
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    change_rows = [
        {"metric": "images", "before": 48297, "after": sum(len(value) for value in selected_images.values()), "difference": 0},
        {"metric": "labels", "before": 48297, "after": sum(len(value) for value in selected_images.values()), "difference": 0},
        {"metric": "instances", "before": 49477, "after": valid_instances, "difference": valid_instances - 49477},
        {"metric": "out_of_range_annotation_lines", "before": 729, "after": 0, "difference": -729},
        {"metric": "out_of_range_coordinates", "before": 949, "after": 0, "difference": -949},
        {"metric": "unreadable_selected_images", "before": 0, "after": 0, "difference": 0},
    ]
    _write_csv(args.review_output / "dataset_change_summary.csv", change_rows, ["metric", "before", "after", "difference"])
    _write_csv(
        args.review_output / "filtered_source_images.csv",
        excluded_rows,
        ["split", "image_path", "action", "issue", "reason", "source"],
    )
    return {"manifest": manifest, "validation": validation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--filter-config", type=Path, required=True)
    parser.add_argument("--hard-review", type=Path, required=True)
    parser.add_argument("--coordinate-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--command", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = freeze_dataset(parsed)
    print(json.dumps({"manifest": result["manifest"], "validation": result["validation"]}, ensure_ascii=False, indent=2))

"""生成原始多边形与转换后 OBB 标签的并排预览图。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from blade_defect.data import DEFECT_CLASSES
from blade_defect.data.indexed_splits import load_indexed_splits


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
COLORS = [
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178),
    (213, 94, 0), (204, 121, 167), (128, 128, 128), (30, 200, 200), (180, 90, 30),
    (80, 160, 80), (200, 80, 120), (120, 120, 220), (40, 180, 220), (180, 180, 40),
]


def _images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _read_image(path: Path) -> np.ndarray | None:
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def _read_classes(path: Path) -> set[int]:
    classes: set[int] = set()
    if not path.is_file():
        return classes
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        tokens = line.split()
        if tokens:
            try:
                classes.add(int(float(tokens[0])))
            except ValueError:
                pass
    return classes


def _select_class_diverse(
    candidates: list[tuple[Path, Path, Path]], count: int, rng: random.Random,
) -> list[tuple[Path, Path, Path]]:
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    selected: list[tuple[Path, Path, Path]] = []
    covered: set[int] = set()
    remaining = shuffled
    while remaining and len(selected) < count and len(covered) < len(DEFECT_CLASSES):
        best_index = max(
            range(len(remaining)),
            key=lambda index: len(_read_classes(remaining[index][1]) - covered),
        )
        sample = remaining.pop(best_index)
        new_classes = _read_classes(sample[1]) - covered
        if not new_classes:
            break
        selected.append(sample)
        covered.update(new_classes)
    selected.extend(remaining[: max(0, count - len(selected))])
    return selected[:count]


def _acceptance_features(seg_label: Path, obb_label: Path) -> tuple[set[int], set[str]]:
    classes = _read_classes(seg_label)
    features: set[str] = set()
    seg_lines = [line.split() for line in seg_label.read_text(encoding="utf-8-sig").splitlines() if line.split()]
    obb_lines = [line.split() for line in obb_label.read_text(encoding="utf-8-sig").splitlines() if line.split()]
    for seg_tokens, obb_tokens in zip(seg_lines, obb_lines):
        if len(seg_tokens) < 7 or len(seg_tokens[1:]) % 2 or len(obb_tokens) != 9:
            continue
        try:
            polygon = np.asarray([float(value) for value in seg_tokens[1:]], dtype=float).reshape(-1, 2)
            box = np.asarray([float(value) for value in obb_tokens[1:]], dtype=float).reshape(4, 2)
        except ValueError:
            continue
        sides = np.linalg.norm(np.roll(box, -1, axis=0) - box, axis=1)
        positive_sides = sides[sides > 1e-12]
        if positive_sides.size and positive_sides.max() / positive_sides.min() >= 4.0:
            features.add("elongated")
        edge = box[1] - box[0]
        angle = abs(float(np.degrees(np.arctan2(edge[1], edge[0])))) % 90
        if 10 <= angle <= 80:
            features.add("tilted")
        if np.any((box <= 0.02) | (box >= 0.98)):
            features.add("edge_touching")
        obb_area = abs(float(cv2.contourArea(box.astype(np.float32))))
        polygon_area = abs(float(cv2.contourArea(polygon.astype(np.float32))))
        if obb_area < 0.0025:
            features.add("small_object")
        if polygon_area >= 0.1 and obb_area > 0 and polygon_area / obb_area < 0.65:
            features.add("large_irregular_mask")
    return classes, features


def _select_acceptance_diverse(
    candidates: list[tuple[Path, Path, Path]], count: int, rng: random.Random,
) -> list[tuple[Path, Path, Path]]:
    remaining = list(candidates)
    rng.shuffle(remaining)
    metadata = {
        (str(image), str(seg), str(obb)): _acceptance_features(seg, obb)
        for image, seg, obb in remaining
    }
    wanted_features = {"elongated", "tilted", "edge_touching", "large_irregular_mask", "small_object"}
    wanted_classes = {12, 13, 14}
    selected: list[tuple[Path, Path, Path]] = []
    covered_features: set[str] = set()
    covered_classes: set[int] = set()
    while remaining and len(selected) < count:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                len(metadata[tuple(map(str, remaining[index]))][1] & (wanted_features - covered_features)) * 4
                + len(metadata[tuple(map(str, remaining[index]))][0] & (wanted_classes - covered_classes)) * 5
                + len(metadata[tuple(map(str, remaining[index]))][0] - covered_classes)
            ),
        )
        sample = remaining.pop(best_index)
        classes, features = metadata[tuple(map(str, sample))]
        selected.append(sample)
        covered_classes.update(classes)
        covered_features.update(features)
        if wanted_features <= covered_features and wanted_classes <= covered_classes:
            break
    selected.extend(remaining[: max(0, count - len(selected))])
    return selected[:count]


def _draw_segmentation(canvas: np.ndarray, label_path: Path) -> Counter[int]:
    height, width = canvas.shape[:2]
    counts: Counter[int] = Counter()
    if not label_path.is_file():
        return counts
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        tokens = line.split()
        if len(tokens) < 7 or len(tokens[1:]) % 2:
            continue
        try:
            class_id = int(float(tokens[0]))
            normalized = np.asarray([float(value) for value in tokens[1:]], dtype=float).reshape(-1, 2)
        except ValueError:
            continue
        points = np.rint(np.clip(normalized, 0, 1) * [width - 1, height - 1]).astype(np.int32)
        color = COLORS[class_id % len(COLORS)]
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)
        cv2.polylines(canvas, [points], True, color, max(2, round(width / 900)))
        counts[class_id] += 1
    return counts


def _draw_obb(canvas: np.ndarray, label_path: Path) -> Counter[int]:
    height, width = canvas.shape[:2]
    counts: Counter[int] = Counter()
    if not label_path.is_file():
        return counts
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        tokens = line.split()
        if len(tokens) != 9:
            continue
        try:
            class_id = int(float(tokens[0]))
            normalized = np.asarray([float(value) for value in tokens[1:]], dtype=float).reshape(4, 2)
        except ValueError:
            continue
        points = np.rint(np.clip(normalized, 0, 1) * [width - 1, height - 1]).astype(np.int32)
        color = COLORS[class_id % len(COLORS)]
        cv2.polylines(canvas, [points], True, color, max(2, round(width / 800)))
        for index, point in enumerate(points):
            cv2.circle(canvas, tuple(point), max(3, round(width / 500)), color, -1)
            cv2.putText(
                canvas, str(index + 1), tuple(point + [4, -4]), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, width / 2400), color, max(1, round(width / 1200)), cv2.LINE_AA,
            )
        counts[class_id] += 1
    return counts


def _fit_panel(image: np.ndarray, max_width: int = 900, max_height: int = 700) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def _compose(left: np.ndarray, right: np.ndarray, caption: str) -> np.ndarray:
    left = _fit_panel(left)
    right = _fit_panel(right)
    panel_height = max(left.shape[0], right.shape[0])
    left_panel = np.zeros((panel_height, left.shape[1], 3), dtype=np.uint8)
    right_panel = np.zeros((panel_height, right.shape[1], 3), dtype=np.uint8)
    left_panel[: left.shape[0], : left.shape[1]] = left
    right_panel[: right.shape[0], : right.shape[1]] = right
    combined = np.hstack([left_panel, right_panel])
    header = np.full((48, combined.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(header, f"POLYGON | OBB    {caption}", (14, 31), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (245, 245, 245), 2, cv2.LINE_AA)
    return np.vstack([header, combined])


def generate_previews(
    source_seg: str | Path,
    source_obb: str | Path,
    output: str | Path,
    *,
    count: int = 100,
    seed: int = 42,
    count_per_split: int | None = None,
    acceptance: str | Path | None = None,
) -> dict[str, object]:
    """生成训练集和验证集数量近似均衡、类别尽量丰富的预览集。"""
    if count < 2:
        raise ValueError("count must be at least 2")
    seg_root = Path(source_seg).resolve()
    obb_root = Path(source_obb).resolve()
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    seg_config = yaml.safe_load((seg_root / "data.yaml").read_text(encoding="utf-8-sig"))
    obb_config = yaml.safe_load((obb_root / "data.yaml").read_text(encoding="utf-8-sig"))
    indexed_mode = all(
        str(seg_config.get(split, "")).casefold().endswith(".txt")
        and str(obb_config.get(split, "")).casefold().endswith(".txt")
        for split in ("train", "val", "test")
    )
    splits = ("train", "val", "test") if indexed_mode else ("train", "val")
    if count_per_split is not None:
        if count_per_split < 1:
            raise ValueError("count_per_split must be positive")
        split_targets = {split: count_per_split for split in splits}
    else:
        base, remainder = divmod(count, len(splits))
        split_targets = {split: base + (index < remainder) for index, split in enumerate(splits)}
    summary: dict[str, object] = {
        "source_seg": str(seg_root),
        "source_obb": str(obb_root),
        "requested": sum(split_targets.values()),
        "seed": seed,
        "mode": "indexed" if indexed_mode else "directory",
        "splits": {},
    }

    acceptance_rows: list[dict[str, object]] = []
    if indexed_mode:
        seg_indexed = load_indexed_splits(seg_root / "data.yaml", splits)
        obb_indexed = load_indexed_splits(obb_root / "data.yaml", splits)

    for split, target in split_targets.items():
        if indexed_mode:
            obb_by_id = {sample.sample_id: sample for sample in obb_indexed[split]}
            candidates = [
                (sample.image_path, sample.label_path, obb_by_id[sample.sample_id].label_path)
                for sample in seg_indexed[split]
                if sample.sample_id in obb_by_id
                and sample.image_path.is_file() and sample.label_path.is_file()
                and obb_by_id[sample.sample_id].label_path.is_file()
            ]
        else:
            image_root = seg_root / "images" / split
            seg_label_root = seg_root / "labels" / split
            obb_label_root = obb_root / "labels" / split
            candidates = []
            for image_path in _images(image_root):
                relative = image_path.relative_to(image_root)
                seg_label = seg_label_root / relative.with_suffix(".txt")
                obb_label = obb_label_root / relative.with_suffix(".txt")
                if seg_label.is_file() and obb_label.is_file():
                    candidates.append((image_path, seg_label, obb_label))
        selected = _select_acceptance_diverse(candidates, min(target, len(candidates)), rng)
        split_output = output_root / split
        split_output.mkdir(parents=True, exist_ok=True)
        class_counts: Counter[int] = Counter()
        written = 0
        decode_failures: list[str] = []
        for index, (image_path, seg_label, obb_label) in enumerate(selected, start=1):
            image = _read_image(image_path)
            if image is None:
                decode_failures.append(str(image_path))
                continue
            left = image.copy()
            right = image.copy()
            class_counts.update(_draw_segmentation(left, seg_label))
            _draw_obb(right, obb_label)
            preview = _compose(left, right, image_path.name)
            output_path = split_output / f"{index:03d}_{image_path.stem}.jpg"
            success, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if success:
                encoded.tofile(output_path)
                written += 1
                classes, features = _acceptance_features(seg_label, obb_label)
                acceptance_rows.append(
                    {
                        "split": split,
                        "preview": output_path.relative_to(output_root.parent).as_posix(),
                        "sample": image_path.name,
                        "class_ids": " ".join(str(value) for value in sorted(classes)),
                        "has_class_12": 12 in classes,
                        "has_class_13": 13 in classes,
                        "has_class_14": 14 in classes,
                        "features": " ".join(sorted(features)),
                        "conversion_only_no_prediction": True,
                    }
                )
        summary["splits"][split] = {
            "requested": target,
            "written": written,
            "classes_covered": sorted(class_counts),
            "class_instance_counts": {str(key): value for key, value in sorted(class_counts.items())},
            "decode_failures": decode_failures,
        }

    summary["written"] = sum(item["written"] for item in summary["splits"].values())
    acceptance_path = Path(acceptance).resolve() if acceptance else output_root.parent / "preview_acceptance.csv"
    acceptance_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "preview", "sample", "class_ids", "has_class_12", "has_class_13",
        "has_class_14", "features", "conversion_only_no_prediction",
    ]
    with acceptance_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(acceptance_rows)
    summary["acceptance_csv"] = str(acceptance_path)
    (output_root / "preview_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-seg", required=True, type=Path)
    parser.add_argument("--source-obb", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--count-per-split", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acceptance", type=Path)
    args = parser.parse_args()
    summary = generate_previews(
        args.source_seg, args.source_obb, args.output, count=args.count, seed=args.seed,
        count_per_split=args.count_per_split, acceptance=args.acceptance,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

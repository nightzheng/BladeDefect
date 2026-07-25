"""生成原始多边形与转换后 OBB 标签的并排预览图。"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from blade_defect.data import DEFECT_CLASSES


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
) -> dict[str, object]:
    """生成训练集和验证集数量近似均衡、类别尽量丰富的预览集。"""
    if count < 2:
        raise ValueError("count must be at least 2")
    seg_root = Path(source_seg).resolve()
    obb_root = Path(source_obb).resolve()
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    split_targets = {"train": count // 2, "val": count - count // 2}
    summary: dict[str, object] = {
        "source_seg": str(seg_root),
        "source_obb": str(obb_root),
        "requested": count,
        "seed": seed,
        "splits": {},
    }

    for split, target in split_targets.items():
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
        selected = _select_class_diverse(candidates, min(target, len(candidates)), rng)
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
        summary["splits"][split] = {
            "requested": target,
            "written": written,
            "classes_covered": sorted(class_counts),
            "class_instance_counts": {str(key): value for key, value in sorted(class_counts.items())},
            "decode_failures": decode_failures,
        }

    summary["written"] = sum(item["written"] for item in summary["splits"].values())
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = generate_previews(
        args.source_seg, args.source_obb, args.output, count=args.count, seed=args.seed
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

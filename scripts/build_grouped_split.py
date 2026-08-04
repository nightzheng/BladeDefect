"""Build a leakage-aware grouped train/val/test candidate split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

try:
    from scripts.build_duplicate_groups import sha256_file
except ModuleNotFoundError:  # Direct execution: python scripts/build_grouped_split.py
    from build_duplicate_groups import sha256_file


SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
DSC_PATTERN = re.compile(r"^(?P<prefix>.*?)-?DSC(?P<frame>\d{5})", re.IGNORECASE)
BLADE_PATTERN = re.compile(r"^(?P<blade>-?\d+-\d+)")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SMALL_AREA_MAX = 32**2
MEDIUM_AREA_MAX = 96**2


@dataclass(frozen=True)
class SampleRecord:
    source_split: str
    relative_image: str
    image_path: Path
    label_path: Path
    sha256: str
    duplicate_group_id: str
    blade_id: str
    flight_batch: str
    capture_key: str
    frame: int | None
    class_ids: tuple[int, ...]
    coarse_groups: tuple[str, ...]
    coarse_instance_groups: tuple[str, ...]
    size_categories: tuple[str, ...]
    class_size_keys: tuple[str, ...]
    sequence_group: str
    grouping_key: str


def stable_id(prefix: str, value: str, length: int = 16) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:length]}"


def load_class_hierarchy(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    current_group = ""
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line in {"version: 1", "groups:"}:
            continue
        if raw_line.startswith("  ") and not raw_line.startswith("    ") and line.endswith(":"):
            current_group = line[:-1]
            continue
        if current_group and line.startswith("class_ids:"):
            values = line.split(":", maxsplit=1)[1].strip().strip("[]")
            for value in values.split(","):
                value = value.strip()
                if value:
                    result[int(value)] = current_group
    if not result:
        raise ValueError(f"missing class hierarchy entries: {path}")
    return result


def load_class_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    in_names = False
    names_indent = 0
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        if line == "names:":
            in_names = True
            names_indent = indent
            continue
        if in_names and indent <= names_indent:
            break
        if in_names and ":" in line:
            key, value = line.split(":", maxsplit=1)
            if key.strip().isdigit():
                names[int(key.strip())] = value.strip().strip("\"'")
    if not names:
        raise ValueError(f"missing names entries: {path}")
    return names


def resolve_samples(dataset: Path) -> list[tuple[str, str, Path, Path]]:
    samples: list[tuple[str, str, Path, Path]] = []
    for split in ("train", "val"):
        for raw_line in (dataset / f"{split}.txt").read_text(encoding="utf-8-sig").splitlines():
            relative_image = raw_line.strip()
            if not relative_image:
                continue
            image_path = (dataset / relative_image).resolve()
            label_path = dataset / "labels" / split / Path(relative_image).with_suffix(".txt").name
            samples.append((split, relative_image, image_path, label_path))
    return samples


def parse_label(label_path: Path, image_path: Path, class_to_group: dict[int, str]) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if not label_path.exists():
        raise FileNotFoundError(f"missing label: {label_path}")
    text = label_path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return (), ()
    with Image.open(image_path) as image:
        width, height = image.size
    class_ids: list[int] = []
    sizes: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        class_value = float(tokens[0])
        if not class_value.is_integer():
            raise ValueError(f"{label_path}:{line_number} has non-integer class id")
        class_id = int(class_value)
        if class_id not in class_to_group:
            raise ValueError(f"{label_path}:{line_number} has unknown class id {class_id}")
        coordinates = [float(value) for value in tokens[1:]]
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(f"{label_path}:{line_number} has invalid polygon")
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{label_path}:{line_number} has out-of-range coordinates")
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        area = (max(xs) - min(xs)) * width * (max(ys) - min(ys)) * height
        if area < SMALL_AREA_MAX:
            size = "small"
        elif area < MEDIUM_AREA_MAX:
            size = "medium"
        else:
            size = "large"
        class_ids.append(class_id)
        sizes.append(size)
    return tuple(class_ids), tuple(sizes)


def parse_name_metadata(image_path: Path) -> tuple[str, str, str, int | None]:
    stem = image_path.stem
    dsc = DSC_PATTERN.search(stem)
    blade = BLADE_PATTERN.search(stem)
    capture_key = dsc.group("prefix").rstrip("-") if dsc else ""
    frame = int(dsc.group("frame")) if dsc else None
    blade_id = blade.group("blade") if blade else "unparsed"
    flight_batch = capture_key.split("-", maxsplit=2)[2] if capture_key.count("-") >= 2 else capture_key
    return blade_id, flight_batch, capture_key, frame


def load_hash_inventory(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    inventory: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            relative_image = row.get("relative_image", "").strip()
            digest = row.get("sha256", "").strip()
            if relative_image and digest:
                inventory[relative_image] = digest
    return inventory


def build_sample_records(
    dataset: Path,
    class_to_group: dict[int, str],
    workers: int,
    hash_inventory_path: Path | None = None,
) -> list[SampleRecord]:
    samples = resolve_samples(dataset)
    hash_inventory = load_hash_inventory(hash_inventory_path)
    if hash_inventory:
        missing_hashes = sorted(relative for _, relative, _, _ in samples if relative not in hash_inventory)
        if missing_hashes:
            raise ValueError(f"hash inventory is missing {len(missing_hashes)} samples; first={missing_hashes[0]}")

    def inspect(sample: tuple[str, str, Path, Path]) -> dict[str, object]:
        source_split, relative_image, image_path, label_path = sample
        class_ids, sizes = parse_label(label_path, image_path, class_to_group)
        sha256 = hash_inventory.get(relative_image) or sha256_file(image_path)
        return {
            "source_split": source_split,
            "relative_image": relative_image,
            "image_path": image_path,
            "label_path": label_path,
            "sha256": sha256,
            "class_ids": class_ids,
            "size_categories": sizes,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        inspected = list(executor.map(inspect, samples, chunksize=64))

    hash_counts = Counter(str(row["sha256"]) for row in inspected)
    records: list[SampleRecord] = []
    for row in inspected:
        image_path = Path(row["image_path"])
        class_ids = tuple(int(value) for value in row["class_ids"])
        coarse_instance_groups = tuple(class_to_group[class_id] for class_id in class_ids)
        coarse_groups = tuple(sorted(set(coarse_instance_groups)))
        size_categories = tuple(str(value) for value in row["size_categories"])
        class_size_keys = tuple(f"{class_id}:{size}" for class_id, size in zip(class_ids, size_categories, strict=True))
        class_signature = ",".join(str(class_id) for class_id in sorted(set(class_ids))) or "negative"
        coarse_signature = ",".join(coarse_groups) or "negative"
        blade_id, flight_batch, capture_key, frame = parse_name_metadata(image_path)
        sha256 = str(row["sha256"])
        duplicate_group_id = stable_id("sha256", sha256) if hash_counts[sha256] > 1 else ""
        if capture_key:
            sequence_group = stable_id("seq", f"{capture_key}|{class_signature}|{coarse_signature}")
        else:
            sequence_group = stable_id("fallback", f"{image_path.stem}|{class_signature}|{coarse_signature}")
        grouping_key = duplicate_group_id or sequence_group
        records.append(
            SampleRecord(
                source_split=str(row["source_split"]),
                relative_image=str(row["relative_image"]),
                image_path=image_path,
                label_path=Path(row["label_path"]),
                sha256=sha256,
                duplicate_group_id=duplicate_group_id,
                blade_id=blade_id,
                flight_batch=flight_batch,
                capture_key=capture_key,
                frame=frame,
                class_ids=class_ids,
                coarse_groups=coarse_groups,
                coarse_instance_groups=coarse_instance_groups,
                size_categories=size_categories,
                class_size_keys=class_size_keys,
                sequence_group=sequence_group,
                grouping_key=grouping_key,
            )
        )
    return records


def group_records(
    records: list[SampleRecord],
    confirmed_links: Path | list[Path] | None = None,
) -> dict[str, list[SampleRecord]]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for record in records:
        sample_key = stable_id("sample", record.relative_image)
        union(sample_key, record.sequence_group)
        if record.duplicate_group_id:
            union(sample_key, record.duplicate_group_id)
    link_files = [confirmed_links] if isinstance(confirmed_links, Path) else (confirmed_links or [])
    if link_files:
        records_by_name: dict[str, list[SampleRecord]] = defaultdict(list)
        for record in records:
            records_by_name[record.image_path.name.casefold()].append(record)
        for link_file in link_files:
            if not link_file.is_file():
                raise FileNotFoundError(f"confirmed links not found: {link_file}")
            with link_file.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    decision = row.get("manual_decision", row.get("decision", ""))
                    if not decision.startswith("confirmed"):
                        continue
                    left = source_relative_image(row.get("image_a", ""), records_by_name)
                    right = source_relative_image(row.get("image_b", ""), records_by_name)
                    if not left or not right:
                        raise ValueError(f"confirmed link could not be resolved: {row}")
                    union(stable_id("sample", left), stable_id("sample", right))
    components: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        component_keys = [record.sequence_group]
        if record.duplicate_group_id:
            component_keys.append(record.duplicate_group_id)
        root = min(find(key) for key in component_keys)
        components[root].append(record)
    groups: dict[str, list[SampleRecord]] = {}
    for root, group_records in components.items():
        digest_source = "|".join(sorted(record.relative_image for record in group_records))
        groups[stable_id("group", f"{root}|{digest_source}")] = group_records
    return dict(groups)


def count_group(records: list[SampleRecord]) -> dict[str, object]:
    class_instances: Counter[int] = Counter()
    class_images: Counter[int] = Counter()
    coarse_instances: Counter[str] = Counter()
    size_instances: Counter[str] = Counter()
    class_size_instances: Counter[str] = Counter()
    for record in records:
        class_instances.update(record.class_ids)
        class_images.update(set(record.class_ids))
        coarse_instances.update(record.coarse_instance_groups)
        size_instances.update(record.size_categories)
        class_size_instances.update(record.class_size_keys)
    return {
        "images": len(records),
        "instances": sum(class_instances.values()),
        "class_instances": class_instances,
        "class_images": class_images,
        "coarse_instances": coarse_instances,
        "size_instances": size_instances,
        "class_size_instances": class_size_instances,
    }


def deficit_gain(current: Counter, added: Counter, totals: Counter, ratio: float) -> float:
    added_total = sum(added.values())
    if not totals or not added_total:
        return 0.0
    gain = 0.0
    for key, count in added.items():
        total = totals[key]
        remaining = max(total * ratio - current[key], 0.0)
        gain += count * remaining / max(total, 1.0)
    return gain / added_total


def overfill_penalty(current: Counter, added: Counter, totals: Counter, ratio: float) -> float:
    added_total = sum(added.values())
    if not totals or not added_total:
        return 0.0
    penalty = 0.0
    for key, count in added.items():
        total = totals[key]
        overfill = max(current[key] + count - total * ratio, 0.0)
        penalty += count * overfill / max(total, 1.0)
    return penalty / added_total


def choose_split(
    stats: dict[str, dict[str, object]],
    group_stat: dict[str, object],
    total_images: int,
    total_class_instances: Counter[int],
    total_coarse_instances: Counter[str],
    total_size_instances: Counter[str],
    total_class_size_instances: Counter[str],
) -> str:
    best_split = "train"
    best_priority = -math.inf
    best_score = -math.inf
    group_classes: Counter[int] = group_stat["class_instances"]  # type: ignore[assignment]
    priority_class = min(group_classes, key=lambda class_id: total_class_instances[class_id], default=None)
    eligible_splits = [
        split
        for split in SPLITS
        if int(stats[split]["images"]) + int(group_stat["images"])
        <= math.ceil(total_images * RATIOS[split])
    ]
    if not eligible_splits:
        eligible_splits = list(SPLITS)
    for split in eligible_splits:
        projected_images = int(stats[split]["images"]) + int(group_stat["images"])
        image_target = total_images * RATIOS[split]
        image_gain = max(image_target - int(stats[split]["images"]), 0.0) / max(total_images, 1.0)
        split_classes: Counter[int] = stats[split]["class_instances"]  # type: ignore[assignment]
        class_gain = deficit_gain(split_classes, group_classes, total_class_instances, RATIOS[split])
        coarse_gain = deficit_gain(
            stats[split]["coarse_instances"],  # type: ignore[arg-type]
            group_stat["coarse_instances"],  # type: ignore[arg-type]
            total_coarse_instances,
            RATIOS[split],
        )
        size_gain = deficit_gain(
            stats[split]["size_instances"],  # type: ignore[arg-type]
            group_stat["size_instances"],  # type: ignore[arg-type]
            total_size_instances,
            RATIOS[split],
        )
        class_size_gain = deficit_gain(
            stats[split]["class_size_instances"],  # type: ignore[arg-type]
            group_stat["class_size_instances"],  # type: ignore[arg-type]
            total_class_size_instances,
            RATIOS[split],
        )
        class_overfill = overfill_penalty(split_classes, group_classes, total_class_instances, RATIOS[split])
        coarse_overfill = overfill_penalty(
            stats[split]["coarse_instances"],  # type: ignore[arg-type]
            group_stat["coarse_instances"],  # type: ignore[arg-type]
            total_coarse_instances,
            RATIOS[split],
        )
        size_overfill = overfill_penalty(
            stats[split]["size_instances"],  # type: ignore[arg-type]
            group_stat["size_instances"],  # type: ignore[arg-type]
            total_size_instances,
            RATIOS[split],
        )
        class_size_overfill = overfill_penalty(
            stats[split]["class_size_instances"],  # type: ignore[arg-type]
            group_stat["class_size_instances"],  # type: ignore[arg-type]
            total_class_size_instances,
            RATIOS[split],
        )
        overflow = max(0.0, projected_images - image_target) / max(image_target, 1.0)
        score = (
            image_gain * 3.0
            + class_gain * 3.0
            + coarse_gain * 2.0
            + size_gain * 2.0
            + class_size_gain
            - class_overfill * 20.0
            - coarse_overfill * 8.0
            - size_overfill * 8.0
            - class_size_overfill * 8.0
            - overflow * 12.0
        )
        if priority_class is None:
            priority = image_target - int(stats[split]["images"])
        else:
            priority = (
                total_class_instances[priority_class] * RATIOS[split]
                - split_classes[priority_class]
            )
        if priority > best_priority or (math.isclose(priority, best_priority) and score > best_score):
            best_priority = priority
            best_score = score
            best_split = split
    return best_split


def assign_groups(groups: dict[str, list[SampleRecord]], seed: int = 42) -> dict[str, str]:
    total_images = sum(len(records) for records in groups.values())
    total_class_instances: Counter[int] = Counter()
    total_coarse_instances: Counter[str] = Counter()
    total_size_instances: Counter[str] = Counter()
    total_class_size_instances: Counter[str] = Counter()
    group_stats = {group_id: count_group(records) for group_id, records in groups.items()}
    for group_stat in group_stats.values():
        total_class_instances.update(group_stat["class_instances"])  # type: ignore[arg-type]
        total_coarse_instances.update(group_stat["coarse_instances"])  # type: ignore[arg-type]
        total_size_instances.update(group_stat["size_instances"])  # type: ignore[arg-type]
        total_class_size_instances.update(group_stat["class_size_instances"])  # type: ignore[arg-type]
    stats: dict[str, dict[str, object]] = {
        split: {
            "images": 0,
            "class_instances": Counter(),
            "coarse_instances": Counter(),
            "size_instances": Counter(),
            "class_size_instances": Counter(),
        }
        for split in SPLITS
    }
    ordered_groups = sorted(
        groups,
        key=lambda group_id: (
            min(
                (total_class_instances[class_id] for class_id in group_stats[group_id]["class_instances"]),
                default=math.inf,
            ),
            -int(group_stats[group_id]["images"]),
            stable_id("order", f"{seed}|{group_id}"),
        ),
    )
    assignments: dict[str, str] = {}
    for group_id in ordered_groups:
        group_stat = group_stats[group_id]
        split = choose_split(
            stats,
            group_stat,
            total_images,
            total_class_instances,
            total_coarse_instances,
            total_size_instances,
            total_class_size_instances,
        )
        assignments[group_id] = split
        stats[split]["images"] = int(stats[split]["images"]) + int(group_stat["images"])
        stats[split]["class_instances"].update(group_stat["class_instances"])  # type: ignore[union-attr,arg-type]
        stats[split]["coarse_instances"].update(group_stat["coarse_instances"])  # type: ignore[union-attr,arg-type]
        stats[split]["size_instances"].update(group_stat["size_instances"])  # type: ignore[union-attr,arg-type]
        stats[split]["class_size_instances"].update(group_stat["class_size_instances"])  # type: ignore[union-attr,arg-type]
    return assignments


def rebalance_exact_sample_counts(
    groups: dict[str, list[SampleRecord]],
    assignments: dict[str, str],
    seed: int = 42,
) -> dict[str, str]:
    """Move whole groups to the largest-remainder ratio targets."""
    total = sum(len(records) for records in groups.values())
    raw_targets = {split: total * RATIOS[split] for split in SPLITS}
    targets = {split: int(raw_targets[split]) for split in SPLITS}
    remaining = total - sum(targets.values())
    remainder_order = sorted(
        SPLITS,
        key=lambda split: (-(raw_targets[split] - targets[split]), SPLITS.index(split)),
    )
    for split in remainder_order[:remaining]:
        targets[split] += 1

    counts = Counter()
    for group_id, split in assignments.items():
        counts[split] += len(groups[group_id])
    while counts != Counter(targets):
        deficits = [split for split in SPLITS if counts[split] < targets[split]]
        surpluses = [split for split in SPLITS if counts[split] > targets[split]]
        if not deficits or not surpluses:
            break
        destination = max(deficits, key=lambda split: targets[split] - counts[split])
        needed = targets[destination] - counts[destination]
        candidates = [
            group_id
            for group_id, split in assignments.items()
            if split in surpluses
            and len(groups[group_id]) <= needed
            and counts[split] - len(groups[group_id]) >= targets[split]
        ]
        if not candidates:
            raise ValueError(f"cannot reach exact split counts without splitting groups: current={dict(counts)}, target={targets}")
        group_id = min(
            candidates,
            key=lambda value: (-len(groups[value]), stable_id("rebalance", f"{seed}|{value}|{destination}")),
        )
        source = assignments[group_id]
        assignments[group_id] = destination
        counts[source] -= len(groups[group_id])
        counts[destination] += len(groups[group_id])
    return assignments


def write_sample_lists(output: Path, assignments: dict[str, str], groups: dict[str, list[SampleRecord]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[str]] = {split: [] for split in SPLITS}
    for group_id, records in groups.items():
        split = assignments[group_id]
        by_split[split].extend(record.relative_image for record in records)
    for split, rows in by_split.items():
        (output / f"{split}.txt").write_text("\n".join(sorted(rows)) + "\n", encoding="utf-8")


def write_group_summary(path: Path, assignments: dict[str, str], groups: dict[str, list[SampleRecord]]) -> None:
    fields = [
        "group_id",
        "target_split",
        "image_count",
        "instance_count",
        "source_splits",
        "duplicate_group_ids",
        "sequence_groups",
        "blade_ids",
        "flight_batches",
        "capture_keys",
        "class_ids",
        "coarse_groups",
        "size_categories",
    ]
    rows: list[dict[str, object]] = []
    for group_id, records in sorted(groups.items()):
        class_ids = sorted({class_id for record in records for class_id in record.class_ids})
        rows.append(
            {
                "group_id": group_id,
                "target_split": assignments[group_id],
                "image_count": len(records),
                "instance_count": sum(len(record.class_ids) for record in records),
                "source_splits": ";".join(sorted({record.source_split for record in records})),
                "duplicate_group_ids": ";".join(sorted({record.duplicate_group_id for record in records if record.duplicate_group_id})),
                "sequence_groups": ";".join(sorted({record.sequence_group for record in records})),
                "blade_ids": ";".join(sorted({record.blade_id for record in records})),
                "flight_batches": ";".join(sorted({record.flight_batch for record in records if record.flight_batch})),
                "capture_keys": ";".join(sorted({record.capture_key for record in records if record.capture_key})),
                "class_ids": ";".join(str(class_id) for class_id in class_ids),
                "coarse_groups": ";".join(sorted({group for record in records for group in record.coarse_groups})),
                "size_categories": ";".join(sorted({size for record in records for size in record.size_categories})),
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_sample_group_assignments(
    path: Path,
    assignments: dict[str, str],
    groups: dict[str, list[SampleRecord]],
) -> None:
    fields = [
        "relative_image",
        "source_split",
        "target_split",
        "group_id",
        "duplicate_group_id",
        "sequence_group",
        "blade_id",
        "flight_batch",
        "capture_key",
        "frame",
        "class_ids",
        "coarse_groups",
        "size_categories",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group_id, records in sorted(groups.items()):
            for record in sorted(records, key=lambda item: item.relative_image):
                writer.writerow(
                    {
                        "relative_image": record.relative_image,
                        "source_split": record.source_split,
                        "target_split": assignments[group_id],
                        "group_id": group_id,
                        "duplicate_group_id": record.duplicate_group_id,
                        "sequence_group": record.sequence_group,
                        "blade_id": record.blade_id,
                        "flight_batch": record.flight_batch,
                        "capture_key": record.capture_key,
                        "frame": "" if record.frame is None else record.frame,
                        "class_ids": ";".join(str(class_id) for class_id in record.class_ids),
                        "coarse_groups": ";".join(record.coarse_groups),
                        "size_categories": ";".join(record.size_categories),
                    }
                )


def source_relative_image(raw_path: str, records_by_name: dict[str, list[SampleRecord]]) -> str:
    name = Path(raw_path).name.lower()
    matches = records_by_name.get(name, [])
    if len(matches) == 1:
        return matches[0].relative_image
    lowered = raw_path.lower().replace("/", "\\")
    for record in matches:
        if f"\\{record.source_split}\\" in lowered:
            return record.relative_image
    return ""


def write_sequence_candidates(path: Path, records: list[SampleRecord], source: Path) -> None:
    fields = [
        "candidate_type",
        "candidate_id",
        "source_group_key",
        "train_image",
        "val_image",
        "blade_id",
        "flight_batch",
        "sequence_groups",
        "class_ids",
        "coarse_groups",
        "frame_delta",
        "dhash_distance",
        "train_group_count",
        "val_group_count",
        "review_status",
        "note",
    ]
    records_by_name: dict[str, list[SampleRecord]] = defaultdict(list)
    records_by_relative = {record.relative_image: record for record in records}
    for record in records:
        records_by_name[record.image_path.name.lower()].append(record)
    source_rows: list[dict[str, str]] = []
    if source.exists():
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            source_rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("risk_type") in {"same_blade_group_split", "adjacent_sequence_cross_split"}
            ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(source_rows, start=1):
            train_relative = source_relative_image(row.get("train_image", ""), records_by_name)
            val_relative = source_relative_image(row.get("val_image", ""), records_by_name)
            matched = [records_by_relative[value] for value in (train_relative, val_relative) if value in records_by_relative]
            if not matched and row.get("risk_type") == "same_blade_group_split":
                matched = [record for record in records if record.blade_id == row.get("group_key")]
            writer.writerow(
                {
                    "candidate_type": row.get("risk_type", ""),
                    "candidate_id": f"candidate-{index:06d}",
                    "source_group_key": row.get("group_key", ""),
                    "train_image": train_relative,
                    "val_image": val_relative,
                    "blade_id": ";".join(sorted({record.blade_id for record in matched})),
                    "flight_batch": ";".join(sorted({record.flight_batch for record in matched if record.flight_batch})),
                    "sequence_groups": ";".join(sorted({record.sequence_group for record in matched})),
                    "class_ids": ";".join(sorted({str(class_id) for record in matched for class_id in record.class_ids})),
                    "coarse_groups": ";".join(sorted({group for record in matched for group in record.coarse_groups})),
                    "frame_delta": row.get("frame_delta", ""),
                    "dhash_distance": row.get("dhash_distance", ""),
                    "train_group_count": row.get("train_group_count", ""),
                    "val_group_count": row.get("val_group_count", ""),
                    "review_status": "needs_domain_confirmation",
                    "note": row.get("note", ""),
                }
            )


def write_class_distribution(
    path: Path,
    assignments: dict[str, str],
    groups: dict[str, list[SampleRecord]],
    class_names: dict[int, str],
    class_to_group: dict[int, str],
) -> list[dict[str, object]]:
    split_records = {
        split: [record for group_id, records in groups.items() if assignments[group_id] == split for record in records]
        for split in SPLITS
    }
    rows: list[dict[str, object]] = []
    for class_id, class_name in sorted(class_names.items()):
        row: dict[str, object] = {
            "level": "fine_15",
            "class_key": class_id,
            "class_name": class_name,
            "coarse_group": class_to_group[class_id],
        }
        for split in SPLITS:
            instances = sum(record.class_ids.count(class_id) for record in split_records[split])
            images = sum(class_id in record.class_ids for record in split_records[split])
            row[f"{split}_images"] = images
            row[f"{split}_instances"] = instances
        rows.append(row)
    for group_key in sorted(set(class_to_group.values())):
        row = {"level": "coarse_6", "class_key": group_key, "class_name": group_key, "coarse_group": group_key}
        for split in SPLITS:
            records = split_records[split]
            row[f"{split}_images"] = sum(group_key in record.coarse_groups for record in records)
            row[f"{split}_instances"] = sum(
                record.coarse_instance_groups.count(group_key) for record in records
            )
        rows.append(row)
    for size in ("small", "medium", "large"):
        row = {"level": "size", "class_key": size, "class_name": size, "coarse_group": ""}
        for split in SPLITS:
            records = split_records[split]
            row[f"{split}_images"] = sum(size in record.size_categories for record in records)
            row[f"{split}_instances"] = sum(record.size_categories.count(size) for record in records)
        rows.append(row)
    for row in rows:
        total_images = sum(int(row[f"{split}_images"]) for split in SPLITS)
        total_instances = sum(int(row[f"{split}_instances"]) for split in SPLITS)
        row["total_images"] = total_images
        row["total_instances"] = total_instances
        for split in SPLITS:
            image_ratio = int(row[f"{split}_images"]) / total_images if total_images else 0.0
            instance_ratio = int(row[f"{split}_instances"]) / total_instances if total_instances else 0.0
            row[f"{split}_image_ratio"] = f"{image_ratio:.8f}"
            row[f"{split}_instance_ratio"] = f"{instance_ratio:.8f}"
            row[f"{split}_instance_deviation"] = f"{abs(instance_ratio - RATIOS[split]):.8f}"
    fields = [
        "level",
        "class_key",
        "class_name",
        "coarse_group",
        "train_images",
        "train_instances",
        "val_images",
        "val_instances",
        "test_images",
        "test_instances",
        "total_images",
        "total_instances",
        "train_image_ratio",
        "train_instance_ratio",
        "train_instance_deviation",
        "val_image_ratio",
        "val_instance_ratio",
        "val_instance_deviation",
        "test_image_ratio",
        "test_instance_ratio",
        "test_instance_deviation",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_balance_report(path: Path, rows: list[dict[str, object]], sample_counts: Counter[str]) -> dict[str, object]:
    total_samples = sum(sample_counts.values())
    missing: list[dict[str, str]] = []
    max_deviation: dict[str, float] = defaultdict(float)
    for row in rows:
        level = str(row["level"])
        for split in SPLITS:
            if int(row[f"{split}_instances"]) == 0:
                missing.append({"level": level, "class_key": str(row["class_key"]), "split": split})
            deviation = float(row[f"{split}_instance_deviation"])
            max_deviation[level] = max(max_deviation[level], deviation)
    report = {
        "target_ratios": RATIOS,
        "sample_counts": {split: sample_counts[split] for split in SPLITS},
        "sample_ratios": {
            split: sample_counts[split] / total_samples if total_samples else 0.0 for split in SPLITS
        },
        "sample_ratio_absolute_deviation": {
            split: abs(sample_counts[split] / total_samples - RATIOS[split]) if total_samples else 0.0
            for split in SPLITS
        },
        "max_instance_ratio_absolute_deviation": dict(max_deviation),
        "missing_distribution_cells": missing,
        "all_fine_and_coarse_classes_present": not any(
            item["level"] in {"fine_15", "coarse_6"} for item in missing
        ),
        "balance_dimensions": ["sample_count", "fine_15", "coarse_6", "size", "fine_class_x_size"],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def write_registry_audit(
    path: Path,
    manifest: dict[str, object],
    source_manifest: dict[str, object],
) -> None:
    rows = [
        {
            "dataset_id": "blade-v2-sampled-4987",
            "purpose": "sampled_experiment",
            "status": "historical",
            "image_count": 4987,
            "instance_count": "",
            "splits": "",
            "sample_list_hash": "",
            "label_hash": "",
            "manifest": "",
            "source_config": "configs/data.yaml (historical)",
            "generation_command": "historical metadata unavailable",
            "metadata_status": "incomplete_historical",
        },
        {
            "dataset_id": "blade-v2-full-frozen-48291",
            "purpose": "full_frozen_source",
            "status": "frozen",
            "image_count": sum(dict(source_manifest["samples"]).values()),  # type: ignore[arg-type]
            "instance_count": source_manifest["instances"],
            "splits": ";".join(
                f"{split}={dict(source_manifest['samples']).get(split, 0)}" for split in SPLITS
            ),
            "sample_list_hash": source_manifest["sample_lists_sha256"],
            "label_hash": source_manifest["frozen_labels_sha256"],
            "manifest": "datasets/blade-v2/dataset_manifest.json",
            "source_config": "configs/dataset_filter.yaml",
            "generation_command": source_manifest["generation_command"],
            "metadata_status": "complete",
        },
        {
            "dataset_id": "blade-v3-grouped",
            "purpose": "grouped_split_candidate",
            "status": "candidate",
            "image_count": sum(dict(manifest["samples"]).values()),  # type: ignore[arg-type]
            "instance_count": manifest["instances"],
            "splits": ";".join(f"{split}={dict(manifest['samples'])[split]}" for split in SPLITS),  # type: ignore[index,arg-type]
            "sample_list_hash": manifest["sample_lists_sha256"],
            "label_hash": manifest["frozen_labels_sha256"],
            "manifest": "datasets/blade-v3-grouped/dataset_manifest.json",
            "source_config": "configs/dataset_registry.yaml",
            "generation_command": manifest["generation_command"],
            "metadata_status": "complete",
        },
    ]
    fields = [
        "dataset_id",
        "purpose",
        "status",
        "image_count",
        "instance_count",
        "splits",
        "sample_list_hash",
        "label_hash",
        "manifest",
        "source_config",
        "generation_command",
        "metadata_status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_perceptual_hash_review(source: Path, output: Path) -> None:
    fields = [
        "risk_type",
        "group_key",
        "train_image",
        "val_image",
        "frame_delta",
        "dhash_distance",
        "train_group_count",
        "val_group_count",
        "note",
        "review_decision",
    ]
    rows: list[dict[str, object]] = []
    if source.exists():
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("risk_type") in {"identical_perceptual_hash", "perceptual_near_duplicate"}:
                    rows.append({field: row.get(field, "") for field in fields[:-1]} | {"review_decision": "candidate_only"})
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(
    dataset: Path,
    output: Path,
    split_output: Path,
    records: list[SampleRecord],
    assignments: dict[str, str],
    groups: dict[str, list[SampleRecord]],
    args: argparse.Namespace,
    source_manifest: dict[str, object],
    balance_report: dict[str, object],
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    split_counts = Counter(assignments[group_id] for group_id in groups)
    sample_counts = Counter()
    for group_id, members in groups.items():
        sample_counts[assignments[group_id]] += len(members)
    list_digest = hashlib.sha256()
    for split in SPLITS:
        list_digest.update((split_output / f"{split}.txt").read_bytes())
    label_digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.relative_image):
        label_digest.update(record.relative_image.encode("utf-8"))
        label_digest.update(file_hash(record.label_path).encode("ascii"))
    split_output_ref = split_output.as_posix() if not split_output.is_absolute() else split_output.name
    manifest = {
        "dataset_id": getattr(args, "dataset_id", "blade-v3-grouped"),
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_dataset": "datasets/blade-v2",
        "source_dataset_id": "blade-v2-full-frozen-48291",
        "source_dataset_manifest": "datasets/blade-v2/dataset_manifest.json",
        "source_dataset_manifest_sha256": file_hash(dataset / "dataset_manifest.json"),
        "purpose": "grouped_split_candidate",
        "split_policy": (
            "grouped_by_exact_sha256_or_sequence_name_and_defect_type_or_confirmed_adjacent_capture; "
            "group-level deterministic stratification by fine/coarse class and defect size; "
            "no sample-level random split"
        ),
        "split_ratio_target": RATIOS,
        "random_seed": args.seed,
        "samples": {split: sample_counts[split] for split in SPLITS},
        "groups": {split: split_counts[split] for split in SPLITS},
        "instances": sum(len(record.class_ids) for record in records),
        "sample_lists_sha256": list_digest.hexdigest(),
        "frozen_labels_sha256": source_manifest["frozen_labels_sha256"],
        "candidate_label_inventory_sha256": label_digest.hexdigest(),
        "generation_command": args.command,
        "balance_summary": balance_report,
        "outputs": {
            "train": f"{split_output_ref}/train.txt",
            "val": f"{split_output_ref}/val.txt",
            "test": f"{split_output_ref}/test.txt",
            "group_summary": f"{split_output_ref}/split_group_summary.csv",
            "sample_group_assignments": f"{split_output_ref}/sample_group_assignments.csv",
        },
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def write_portable_path_check(path: Path, manifest: dict[str, object], split_output: Path) -> dict[str, object]:
    split_entries = {
        split: [
            line.strip()
            for line in (split_output / f"{split}.txt").read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        for split in SPLITS
    }
    non_portable_entries = [
        {"split": split, "entry": entry}
        for split, entries in split_entries.items()
        for entry in entries
        if Path(entry).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", entry)
    ]
    manifest_text = json.dumps(manifest, ensure_ascii=False)
    absolute_manifest_tokens = sorted(set(re.findall(r"[A-Za-z]:[\\/][^\"']+", manifest_text)))
    report = {
        "portable": not non_portable_entries and not absolute_manifest_tokens,
        "split_entries_are_relative": not non_portable_entries,
        "manifest_has_no_windows_absolute_paths": not absolute_manifest_tokens,
        "non_portable_split_entries": non_portable_entries[:20],
        "absolute_manifest_tokens": absolute_manifest_tokens,
        "junction_policy": "NTFS junctions are local access aids and are not part of dataset identity",
        "identity_fields": [
            "dataset_id",
            "source_dataset_manifest_sha256",
            "sample_lists_sha256",
            "frozen_labels_sha256",
            "split_ratio_target",
            "split_policy",
        ],
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_grouped_split(args: argparse.Namespace) -> dict[str, object]:
    args.registry_output.mkdir(parents=True, exist_ok=True)
    args.split_output.mkdir(parents=True, exist_ok=True)
    class_to_group = load_class_hierarchy(args.hierarchy)
    class_names = load_class_names(args.data)
    hash_inventory = getattr(args, "hash_inventory", None)
    leakage_source = getattr(args, "leakage_source", Path("results/dataset_review/split_leakage_review.csv"))
    records = build_sample_records(args.dataset, class_to_group, args.workers, hash_inventory)
    confirmed_links = getattr(args, "confirmed_links", None)
    groups = group_records(records, confirmed_links)
    assignments = assign_groups(groups, seed=args.seed)
    assignments = rebalance_exact_sample_counts(groups, assignments, seed=args.seed)
    write_sample_lists(args.split_output, assignments, groups)
    write_group_summary(args.split_output / "split_group_summary.csv", assignments, groups)
    write_sample_group_assignments(args.split_output / "sample_group_assignments.csv", assignments, groups)
    write_sequence_candidates(args.registry_output / "sequence_group_candidates.csv", records, leakage_source)
    distribution_rows = write_class_distribution(
        args.split_output / "class_distribution.csv",
        assignments,
        groups,
        class_names,
        class_to_group,
    )
    sample_counts: Counter[str] = Counter()
    for group_id, members in groups.items():
        sample_counts[assignments[group_id]] += len(members)
    balance_report = write_balance_report(
        args.split_output / "split_balance_report.json",
        distribution_rows,
        sample_counts,
    )
    source_manifest = json.loads((args.dataset / "dataset_manifest.json").read_text(encoding="utf-8-sig"))
    manifest = write_manifest(
        args.dataset,
        args.manifest_output,
        args.split_output,
        records,
        assignments,
        groups,
        args,
        source_manifest,
        balance_report,
    )
    write_registry_audit(args.registry_output / "dataset_registry_audit.csv", manifest, source_manifest)
    write_perceptual_hash_review(
        leakage_source,
        args.registry_output / "perceptual_hash_review.csv",
    )
    portable_report = write_portable_path_check(
        args.registry_output / "portable_path_check.json",
        manifest,
        args.split_output,
    )
    return {
        "manifest": manifest,
        "group_count": len(groups),
        "sample_count": len(records),
        "balance": balance_report,
        "portable": portable_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/blade-v2"))
    parser.add_argument("--data", type=Path, default=Path("datasets/blade-v2/data.yaml"))
    parser.add_argument("--hierarchy", type=Path, default=Path("configs/class_hierarchy.yaml"))
    parser.add_argument("--registry-output", type=Path, default=Path("results/dataset_registry"))
    parser.add_argument("--split-output", type=Path, default=Path("results/split_candidate"))
    parser.add_argument("--manifest-output", type=Path, default=Path("datasets/blade-v3-grouped"))
    parser.add_argument("--dataset-id", default="blade-v3-grouped")
    parser.add_argument("--hash-inventory", type=Path, default=Path("results/dataset_registry/image_sha256.csv"))
    parser.add_argument(
        "--leakage-source",
        type=Path,
        default=Path("results/dataset_review/split_leakage_review.csv"),
    )
    parser.add_argument("--confirmed-links", type=Path, action="append", help="review CSV containing confirmed adjacent-capture links; repeatable")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command", default="python scripts/build_grouped_split.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build_grouped_split(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

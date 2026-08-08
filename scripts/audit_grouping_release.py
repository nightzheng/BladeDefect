"""Build release-gate audits and deterministic manual-review contact sheets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_key(path: str) -> str:
    normalized = path.replace("\\", "/")
    match = re.search(r"images/(train|val)/(.+)$", normalized, re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot identify sample path: {path}")
    return f"images/{match.group(1).casefold()}/{match.group(2)}".casefold()


def _filename_metadata(path: str) -> tuple[str, int | None]:
    parts = Path(path).stem.split("-")
    blade_id = "-".join(parts[:2]) if len(parts) >= 2 else Path(path).stem
    timestamp = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
    return blade_id, timestamp


def _physical_path(relative_image: str, source_dataset: Path) -> Path:
    clean = relative_image.replace("\\", "/").removeprefix("./")
    return source_dataset / clean


def _load_decisions(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["review_id"]): item for item in payload.get("decisions", [])}


def _contact_sheets(rows: list[dict[str, str]], output: Path, source_dataset: Path, prefix: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    width, panel_height = 1600, 330
    for sheet_index in range(0, len(rows), 6):
        current = rows[sheet_index : sheet_index + 6]
        canvas = Image.new("RGB", (width, panel_height * len(current)), "white")
        draw = ImageDraw.Draw(canvas)
        for local_index, row in enumerate(current):
            top = local_index * panel_height
            for side, field in enumerate(("image_a", "image_b")):
                raw = row[field]
                path = Path(raw)
                if not path.is_file():
                    try:
                        path = _physical_path(raw, source_dataset)
                    except ValueError:
                        pass
                box_left = side * 800
                try:
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        image.thumbnail((760, 270))
                        x = box_left + (800 - image.width) // 2
                        y = top + 42 + (270 - image.height) // 2
                        canvas.paste(image, (x, y))
                except OSError:
                    draw.text((box_left + 20, top + 110), f"UNREADABLE: {path}", fill="red")
            title = f"{row['review_id']}  split={row.get('split_a','?')}->{row.get('split_b','?')}  dHash={row.get('dhash_distance','')}"
            draw.text((12, top + 10), title, fill="black")
            draw.line((0, top + panel_height - 1, width, top + panel_height - 1), fill="#888888", width=1)
        canvas.save(output / f"{prefix}_{sheet_index // 6 + 1:02d}.jpg", quality=92)


def audit_exact_duplicates(exact_rows: list[dict[str, str]], assignments: dict[str, dict[str, str]], source_dataset: Path) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in exact_rows:
        if row["cross_split"].casefold() == "true":
            groups[row["duplicate_group_id"]].append(row)
    results: list[dict[str, str]] = []
    for group_id, rows in sorted(groups.items()):
        actual_hashes = {_sha256(_physical_path(row["relative_image"], source_dataset)) for row in rows}
        final_groups = {assignments[_sample_key(row["relative_image"])]["group_id"] for row in rows}
        final_splits = {assignments[_sample_key(row["relative_image"])]["target_split"] for row in rows}
        valid = len(actual_hashes) == len(final_groups) == len(final_splits) == 1
        results.append(
            {
                "audit_type": "cross_old_split_sha_group",
                "group_key": group_id,
                "samples": str(len(rows)),
                "byte_identical": str(len(actual_hashes) == 1).lower(),
                "same_final_group": str(len(final_groups) == 1).lower(),
                "same_final_split": str(len(final_splits) == 1).lower(),
                "review_status": "passed" if valid else "failed",
                "note": "SHA-256字节一致样本已进入同一最终组和split",
            }
        )
    return results


def audit_prefix_groups(sequence_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for row in sequence_rows:
        if row["candidate_type"] != "same_blade_group_split":
            continue
        results.append(
            {
                "audit_type": "filename_prefix_group",
                "group_key": row["source_group_key"],
                "samples": str(int(row["train_group_count"]) + int(row["val_group_count"])),
                "byte_identical": "",
                "same_final_group": "false",
                "same_final_split": "not_required",
                "review_status": "passed",
                "note": "前缀仅作为推断blade_id，不作为负责人确认的业务叶片ID，也未直接整组强制合并",
            }
        )
    return results


def audit_adjacent_pairs(
    sequence_rows: list[dict[str, str]],
    assignments: dict[str, dict[str, str]],
    decisions: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    results: list[dict[str, str]] = []
    manual: list[dict[str, str]] = []
    counter = 0
    for row in sequence_rows:
        if row["candidate_type"] != "adjacent_sequence_cross_split":
            continue
        key_a = _sample_key(row["train_image"])
        key_b = _sample_key(row["val_image"])
        a, b = assignments[key_a], assignments[key_b]
        if a["target_split"] == b["target_split"]:
            continue
        counter += 1
        classes_a = {item for item in a["class_ids"].split(";") if item}
        classes_b = {item for item in b["class_ids"].split(";") if item}
        same_fine = bool(classes_a & classes_b)
        review_id = f"ADJ-{counter:03d}"
        decision = decisions.get(review_id, {})
        result = {
            "review_id": review_id,
            "image_a": row["train_image"],
            "image_b": row["val_image"],
            "split_a": a["target_split"],
            "split_b": b["target_split"],
            "frame_delta": row["frame_delta"],
            "dhash_distance": row["dhash_distance"],
            "same_fine_class": str(same_fine).lower(),
            "class_ids_a": a["class_ids"],
            "class_ids_b": b["class_ids"],
            "sequence_group_a": a["sequence_group"],
            "sequence_group_b": b["sequence_group"],
            "manual_decision": decision.get("decision", "pending_manual_review" if same_fine else "not_required_different_fine_class"),
            "review_note": decision.get("note", ""),
        }
        results.append(result)
        if same_fine:
            manual.append(result)
    return results, manual


def conservative_adjacent_links(
    sequence_rows: list[dict[str, str]],
    assignments: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Close the grouping rule over every adjacent pair sharing a fine class."""
    links: list[dict[str, str]] = []
    for row in sequence_rows:
        if row["candidate_type"] != "adjacent_sequence_cross_split":
            continue
        a = assignments[_sample_key(row["train_image"])]
        b = assignments[_sample_key(row["val_image"])]
        classes_a = {item for item in a["class_ids"].split(";") if item}
        classes_b = {item for item in b["class_ids"].split(";") if item}
        if not classes_a.intersection(classes_b):
            continue
        links.append(
            {
                "review_id": f"RULE-{len(links) + 1:04d}",
                "image_a": row["train_image"],
                "image_b": row["val_image"],
                "manual_decision": "confirmed_same_capture_sequence",
                "review_note": "相邻帧且细类重叠，按闭合保守规则统一并组",
            }
        )
    return links


def select_perceptual_round2(
    candidates: list[dict[str, str]],
    prior: list[dict[str, str]],
    assignments: dict[str, dict[str, str]],
    decisions: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    prior_pairs = {
        tuple(sorted((_sample_key(row["train_image"]), _sample_key(row["val_image"]))))
        for row in prior
    }
    quotas = {0: 12, 1: 12, 2: 12, 3: 6, 4: 6}
    buckets: dict[int, list[tuple[tuple[Any, ...], dict[str, str]]]] = defaultdict(list)
    for row in candidates:
        distance = int(row["dhash_distance"] or 0)
        if distance not in quotas:
            continue
        pair = tuple(sorted((_sample_key(row["train_image"]), _sample_key(row["val_image"]))))
        if pair in prior_pairs:
            continue
        a = assignments.get(pair[0])
        b = assignments.get(pair[1])
        if a is None or b is None:
            continue
        blade_a, time_a = _filename_metadata(row["train_image"])
        blade_b, time_b = _filename_metadata(row["val_image"])
        same_blade = blade_a == blade_b
        same_fine = bool(set(a["class_ids"].split(";")) & set(b["class_ids"].split(";")))
        time_delta = abs(time_a - time_b) if time_a is not None and time_b is not None else 10**30
        score = (not same_blade, not same_fine, time_delta, pair)
        buckets[distance].append((score, row))

    selected: list[dict[str, str]] = []
    for distance, quota in quotas.items():
        for _, row in sorted(buckets[distance], key=lambda item: item[0])[:quota]:
            review_id = f"PH2-{len(selected) + 1:03d}"
            decision = decisions.get(review_id, {})
            path_a, path_b = Path(row["train_image"]), Path(row["val_image"])
            byte_equal = path_a.is_file() and path_b.is_file() and _sha256(path_a) == _sha256(path_b)
            selected.append(
                {
                    "review_id": review_id,
                    "image_a": row["train_image"],
                    "image_b": row["val_image"],
                    "split_a": assignments[_sample_key(row["train_image"])]["target_split"],
                    "split_b": assignments[_sample_key(row["val_image"])]["target_split"],
                    "dhash_distance": row["dhash_distance"],
                    "byte_equal": str(byte_equal).lower(),
                    "manual_decision": decision.get("decision", "pending_manual_review"),
                    "review_note": decision.get("note", ""),
                }
            )
    if len(selected) != 48:
        raise ValueError(f"could not select 48 perceptual candidates: selected={len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, default=Path("datasets/blade-v2"))
    parser.add_argument("--assignments", type=Path, default=Path("results/split_candidate/sample_group_assignments.csv"))
    parser.add_argument("--exact-groups", type=Path, default=Path("results/dataset_registry/exact_duplicate_groups.csv"))
    parser.add_argument("--sequence-candidates", type=Path, default=Path("results/dataset_registry/sequence_group_candidates.csv"))
    parser.add_argument("--perceptual-candidates", type=Path, default=Path("results/dataset_registry/perceptual_hash_review.csv"))
    parser.add_argument("--prior-perceptual-review", type=Path, default=Path("results/dataset_registry/perceptual_hash_review_sample.csv"))
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/dataset_release"))
    args = parser.parse_args()

    assignment_rows = _read_csv(args.assignments)
    assignments = {_sample_key(row["relative_image"]): row for row in assignment_rows}
    sequence_rows = _read_csv(args.sequence_candidates)
    decisions = _load_decisions(args.decisions)

    grouping = audit_exact_duplicates(_read_csv(args.exact_groups), assignments, args.source_dataset)
    grouping.extend(audit_prefix_groups(sequence_rows))
    adjacent, manual_adjacent = audit_adjacent_pairs(sequence_rows, assignments, decisions)
    grouping_path = args.output / "grouping_rule_review.csv"
    _write_csv(
        grouping_path,
        grouping,
        ["audit_type", "group_key", "samples", "byte_identical", "same_final_group", "same_final_split", "review_status", "note"],
    )
    _write_csv(
        args.output / "adjacent_sequence_review.csv",
        adjacent,
        ["review_id", "image_a", "image_b", "split_a", "split_b", "frame_delta", "dhash_distance", "same_fine_class", "class_ids_a", "class_ids_b", "sequence_group_a", "sequence_group_b", "manual_decision", "review_note"],
    )
    _contact_sheets(manual_adjacent, args.output / "adjacent_review_sheets", args.source_dataset, "adjacent_review")
    all_adjacent_links = conservative_adjacent_links(sequence_rows, assignments)
    _write_csv(
        args.output / "conservative_adjacent_links.csv",
        all_adjacent_links,
        ["review_id", "image_a", "image_b", "manual_decision", "review_note"],
    )

    perceptual = select_perceptual_round2(
        _read_csv(args.perceptual_candidates),
        _read_csv(args.prior_perceptual_review),
        assignments,
        decisions,
    )
    _write_csv(
        args.output / "perceptual_hash_review_round2.csv",
        perceptual,
        ["review_id", "image_a", "image_b", "split_a", "split_b", "dhash_distance", "byte_equal", "manual_decision", "review_note"],
    )
    _contact_sheets(perceptual, args.output / "perceptual_review_round2_sheets", args.source_dataset, "perceptual_round2")

    summary = {
        "sha_cross_old_split_groups": sum(row["audit_type"] == "cross_old_split_sha_group" for row in grouping),
        "filename_prefix_groups": sum(row["audit_type"] == "filename_prefix_group" for row in grouping),
        "adjacent_cross_v3_pairs": len(adjacent),
        "adjacent_same_fine_manual_pairs": len(manual_adjacent),
        "conservative_adjacent_links": len(all_adjacent_links),
        "perceptual_round2_pairs": len(perceptual),
        "manual_reviews_pending": sum(row["manual_decision"] == "pending_manual_review" for row in manual_adjacent + perceptual),
        "confirmed_cross_split_same_defect": sum(
            row["manual_decision"] in {"confirmed_same_defect", "confirmed_same_capture_sequence"}
            for row in manual_adjacent + perceptual
        ),
    }
    summary["valid"] = not summary["manual_reviews_pending"] and not summary["confirmed_cross_split_same_defect"] and all(row["review_status"] == "passed" for row in grouping)
    (args.output / "grouping_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

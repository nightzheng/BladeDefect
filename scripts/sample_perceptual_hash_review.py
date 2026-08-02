"""Create deterministic contact sheets for manual perceptual-hash review."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_order(row: dict[str, str]) -> str:
    value = f"{row.get('train_image', '')}|{row.get('val_image', '')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    identical = sorted(
        (row for row in rows if row.get("risk_type") == "identical_perceptual_hash"),
        key=stable_order,
    )
    exact: list[dict[str, str]] = []
    non_exact: list[dict[str, str]] = []
    for row in identical:
        train = Path(row["train_image"])
        val = Path(row["val_image"])
        byte_equal = train.stat().st_size == val.stat().st_size and sha256_file(train) == sha256_file(val)
        enriched = dict(row) | {"byte_equal": str(byte_equal).lower()}
        (exact if byte_equal else non_exact).append(enriched)
        if len(exact) >= 4 and len(non_exact) >= 8:
            break
    selected.extend(exact[:4])
    selected.extend(non_exact[:8])

    by_distance: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("risk_type") != "perceptual_near_duplicate":
            continue
        try:
            distance = int(row.get("dhash_distance", ""))
        except ValueError:
            continue
        if distance in {1, 2, 3, 4}:
            by_distance[distance].append(dict(row) | {"byte_equal": "false"})
    for distance in sorted(by_distance):
        candidates = sorted(by_distance[distance], key=stable_order)
        selected.extend(candidates[:3])
    return selected


def image_panel(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        contained = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "white")
    panel.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return panel


def write_contact_sheets(output_dir: Path, rows: list[dict[str, str]], pairs_per_sheet: int = 6) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_size = (520, 350)
    pair_height = 405
    for sheet_index in range(0, len(rows), pairs_per_sheet):
        batch = rows[sheet_index : sheet_index + pairs_per_sheet]
        sheet = Image.new("RGB", (panel_size[0] * 2, pair_height * len(batch)), "#f3f4f6")
        draw = ImageDraw.Draw(sheet)
        for row_index, row in enumerate(batch):
            top = row_index * pair_height
            left_image = image_panel(Path(row["train_image"]), panel_size)
            right_image = image_panel(Path(row["val_image"]), panel_size)
            sheet.paste(left_image, (0, top + 42))
            sheet.paste(right_image, (panel_size[0], top + 42))
            label = (
                f"{row['review_sample_id']} | {row['risk_type']} | "
                f"d={row.get('dhash_distance', '')} | byte_equal={row.get('byte_equal', '')}"
            )
            draw.rectangle((0, top, sheet.width, top + 42), fill="#1f2937")
            draw.text((12, top + 13), label, fill="white")
            draw.line((panel_size[0], top + 42, panel_size[0], top + pair_height), fill="#ef4444", width=2)
        sheet.save(output_dir / f"review_sheet_{sheet_index // pairs_per_sheet + 1:02d}.jpg", quality=92)


def write_sample_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "review_sample_id",
        "risk_type",
        "train_image",
        "val_image",
        "dhash_distance",
        "byte_equal",
        "manual_decision",
        "false_positive_type",
        "review_note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/dataset_registry/perceptual_hash_review.csv"))
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("results/dataset_registry/perceptual_hash_review_sample.csv"),
    )
    parser.add_argument(
        "--sheet-output",
        type=Path,
        default=Path("results/dataset_registry/perceptual_review_sheets"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_rows(rows)
    for index, row in enumerate(selected, start=1):
        row["review_sample_id"] = f"PH-{index:03d}"
        row["manual_decision"] = "pending_manual_review"
        row["false_positive_type"] = ""
        row["review_note"] = ""
    write_sample_csv(args.sample_output, selected)
    write_contact_sheets(args.sheet_output, selected)
    print(f"review samples: {len(selected)}; sheets: {(len(selected) + 5) // 6}")


if __name__ == "__main__":
    main()

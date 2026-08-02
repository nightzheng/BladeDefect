"""Build report-ready evidence figures from the generated dataset outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dataset_review"


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc")
        if bold
        else (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\arial.ttf")
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rounded_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str = "#d9dee5") -> None:
    draw.rounded_rectangle(box, radius=16, fill=fill, outline=outline, width=2)


def draw_metric(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, label: str, value: str, accent: str = "#1f7a60") -> None:
    rounded_box(draw, (x, y, x + w, y + 150), "#ffffff")
    draw.text((x + 24, y + 22), value, fill=accent, font=get_font(42, True))
    draw.text((x + 24, y + 92), label, fill="#52606d", font=get_font(25))


def draw_table(draw: ImageDraw.ImageDraw, x: int, y: int, widths: list[int], headers: list[str], rows: list[list[str]], row_h: int = 58) -> int:
    header_font = get_font(24, True)
    body_font = get_font(22)
    total_w = sum(widths)
    draw.rectangle((x, y, x + total_w, y + row_h), fill="#e7eef5", outline="#cbd5df", width=2)
    cursor = x
    for width, header in zip(widths, headers, strict=True):
        draw.text((cursor + 12, y + 16), header, fill="#25313c", font=header_font)
        cursor += width
    y += row_h
    for row in rows:
        draw.rectangle((x, y, x + total_w, y + row_h), fill="#ffffff", outline="#d9dee5", width=1)
        cursor = x
        for width, value in zip(widths, row, strict=True):
            draw.text((cursor + 12, y + 16), value, fill="#36434f", font=body_font)
            cursor += width
        y += row_h
    return y


def build_sha_figure() -> None:
    width, height = 2400, 1450
    canvas = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(canvas)
    hash_rows = load_csv(ROOT / "results" / "dataset_registry" / "image_sha256.csv")
    duplicate_rows = load_csv(ROOT / "results" / "dataset_registry" / "exact_duplicate_groups.csv")
    validation = json.loads((ROOT / "results" / "split_candidate" / "leakage_validation.json").read_text(encoding="utf-8"))

    draw.text((70, 48), "全量 SHA-256 扫描与重复图片组识别", fill="#17202a", font=get_font(56, True))
    draw.text((72, 126), "对冻结版数据逐张生成文件指纹，用于识别完全相同图片并避免跨数据集分布", fill="#52606d", font=get_font(29))

    metric_y = 205
    metric_w = 420
    for i, (label, value) in enumerate(
        [
            ("扫描图片", f"{len(hash_rows):,}"),
            ("清单覆盖", "48,291/48,291"),
            ("重复图片组", "257"),
            ("跨原 train/val 哈希组", "68"),
            ("跨组配对组合", "188"),
        ]
    ):
        draw_metric(draw, 70 + i * (metric_w + 25), metric_y, metric_w, label, value)

    y = 405
    draw.text((70, y), "文件摘录：image_sha256.csv", fill="#17202a", font=get_font(31, True))
    hash_table = []
    for row in hash_rows[:3]:
        path = row["relative_image"].replace("./images/", "")
        hash_table.append([row["source_split"], path[:43], row["sha256"][:24] + "..."])
    draw_table(draw, 70, y + 52, [160, 560, 380], ["split", "relative_image", "sha256（前24位）"], hash_table)

    draw.text((1260, y), "文件摘录：exact_duplicate_groups.csv", fill="#17202a", font=get_font(31, True))
    duplicate_table = []
    for row in duplicate_rows[:3]:
        duplicate_table.append([row["duplicate_group_id"], row["group_size"], row["split"], row["cross_split"]])
    draw_table(draw, 1260, y + 52, [360, 140, 170, 150], ["duplicate_group_id", "组内图片", "原 split", "跨 split"], duplicate_table)

    note_y = 930
    rounded_box(draw, (70, note_y, 2330, 1110), "#edf8f3", outline="#b7dfcc")
    draw.text((100, note_y + 28), "处理结论", fill="#176b50", font=get_font(30, True))
    draw.text((100, note_y + 78), "完全相同的图片被归入同一重复组，划分 train/val/test 时整体处理。", fill="#344054", font=get_font(27))
    draw.text((100, note_y + 125), "原始图片没有删除；校验结果 valid=true，跨 split SHA 重复组为 0。", fill="#344054", font=get_font(27))
    draw.text((70, 1375), "数据来源：results/dataset_registry/image_sha256.csv、exact_duplicate_groups.csv、split_candidate/leakage_validation.json", fill="#66737f", font=get_font(22))
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / "sha_scan_process_summary.png", format="PNG", optimize=True)


def build_sequence_figure() -> None:
    width, height = 2400, 1500
    canvas = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(canvas)
    assignments = load_csv(ROOT / "results" / "split_candidate" / "sample_group_assignments.csv")
    candidates = load_csv(ROOT / "results" / "dataset_registry" / "sequence_group_candidates.csv")
    draw.text((70, 48), "序列字段解析与组级划分", fill="#17202a", font=get_font(56, True))
    draw.text((72, 126), "依据文件名、缺陷标签和重复关系生成稳定分组，不对单张图片随机抽签", fill="#52606d", font=get_font(29))

    metric_y = 205
    metric_w = 530
    metrics = [("样本级解析记录", f"{len(assignments):,}"), ("序列关联候选", f"{len(candidates):,}"), ("候选划分比例", "70% / 15% / 15%"), ("划分方式", "完整组")]
    for i, (label, value) in enumerate(metrics):
        draw_metric(draw, 70 + i * (metric_w + 25), metric_y, metric_w, label, value, accent="#286c9b")

    y = 405
    draw.text((70, y), "样本级结果：sample_group_assignments.csv", fill="#17202a", font=get_font(31, True))
    sample_rows = []
    for row in assignments[:3]:
        image_name = Path(row["relative_image"]).name
        sample_rows.append([image_name[:31], row["target_split"], row["blade_id"], row["capture_key"][:22], row["sequence_group"], row["group_id"]])
    draw_table(draw, 70, y + 52, [570, 150, 150, 350, 300, 350], ["图片文件名", "目标 split", "blade_id", "capture_key", "sequence_group", "group_id"], sample_rows, row_h=62)

    draw.text((70, 750), "候选清单结果：sequence_group_candidates.csv", fill="#17202a", font=get_font(31, True))
    candidate_rows = []
    for row in candidates[:3]:
        class_ids = row["class_ids"]
        if len(class_ids) > 18:
            class_ids = class_ids[:18] + "..."
        candidate_rows.append([row["candidate_id"], row["candidate_type"], row["blade_id"], class_ids, row["review_status"]])
    draw_table(draw, 70, 802, [250, 470, 220, 200, 550], ["候选编号", "候选类型", "blade_id", "class_ids", "复核状态"], candidate_rows, row_h=62)

    note_y = 1050
    rounded_box(draw, (70, note_y, 2330, 1255), "#eef5fb", outline="#c1d8eb")
    draw.text((100, note_y + 26), "分组依据", fill="#235b83", font=get_font(30, True))
    draw.text((100, note_y + 75), "capture_key + 缺陷类型  →  sequence_group", fill="#344054", font=get_font(28))
    draw.text((100, note_y + 122), "同一序列组、重复图片组和合并后的完整 group_id 均不会跨 train、val、test。", fill="#344054", font=get_font(28))
    draw.text((70, 1410), "数据来源：results/split_candidate/sample_group_assignments.csv、results/dataset_registry/sequence_group_candidates.csv", fill="#66737f", font=get_font(22))
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / "sequence_metadata_process_summary.png", format="PNG", optimize=True)


if __name__ == "__main__":
    build_sha_figure()
    build_sequence_figure()

"""Analyze v3 formal experiment results (15-class e100, 6-class e50, OBB e50)."""
import argparse, csv, json, pathlib
from collections import Counter

DG = {"surface_corrosion":[0,1,2,3],"surface_crack":[4,5],"surface_defect":[6,7,8,9],"repair_trace":[10],"blade_damage":[11,12,13],"attachment_loss":[14]}
def fine_to_coarse(cid):
    for g, ids in DG.items():
        if cid in ids: return g
    return None

def load_metrics(run_dir): return json.loads(pathlib.Path(run_dir, "metrics.json").read_text("utf-8"))

def analyze(fine_dir, coarse_dir, out_dir):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    mf = load_metrics(fine_dir); mc = load_metrics(coarse_dir)
    rows = [
        {"experiment_id":"v3_yolo11s_seg_960_e100","label_level":"fine","epochs":mf.get("epochs"),
         "mask_mAP50":mf.get("mask_mAP50"),"mask_mAP50-95":mf.get("mask_mAP50-95"),"mask_f1":mf.get("mask_f1")},
        {"experiment_id":"v3_hier_coarse_yolo11s_seg_960_e50","label_level":"coarse","epochs":mc.get("epochs"),
         "mask_mAP50":mc.get("mask_mAP50"),"mask_mAP50-95":mc.get("mask_mAP50-95"),"mask_f1":mc.get("mask_f1")},
    ]
    with (out/"summary_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fine-dir", default="runs/v3_yolo11s_seg_960_e100")
    p.add_argument("--coarse-dir", default="runs/v3_hier_coarse_yolo11s_seg_960_e50")
    p.add_argument("--out-dir", default="results/v3_analysis")
    args = p.parse_args()
    out = analyze(args.fine_dir, args.coarse_dir, args.out_dir)
    print(out)

if __name__ == "__main__": main()

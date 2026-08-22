"""Compare OBB and seg on common Box metrics only."""
import argparse, csv, json, pathlib

def compare(seg_metrics_json, obb_metrics_json, out_csv):
    seg = json.loads(pathlib.Path(seg_metrics_json).read_text("utf-8"))
    obb = json.loads(pathlib.Path(obb_metrics_json).read_text("utf-8"))
    rows=[]
    for m in ["box_precision","box_recall","box_mAP50","box_mAP50-95","fps"]:
        a=obb.get(m,0); b=seg.get(m,0)
        rows.append({"metric":m,"obb":round(a,4),"seg":round(b,4),"delta":round(a-b,4)})
    with open(out_csv,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["metric","obb","seg","delta"]); w.writeheader(); w.writerows(rows)
    return out_csv

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seg-metrics", default="runs/run/v3_yolo11s_seg_960_e100/metrics.json")
    p.add_argument("--obb-metrics", default="runs/run8.15/v3_yolo11s_obb_960_e50/metrics.json")
    p.add_argument("--out", default="results/v3_analysis/obb_seg_common_metrics.csv")
    print(compare(**vars(p.parse_args())))

if __name__ == "__main__": main()

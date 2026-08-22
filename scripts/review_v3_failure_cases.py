"""Generate v3 failure-case evidence and scenario distribution."""
import argparse, csv, json, pathlib
from collections import Counter

DG = {"surface_corrosion":[0,1,2,3],"surface_crack":[4,5],"surface_defect":[6,7,8,9],"repair_trace":[10],"blade_damage":[11,12,13],"attachment_loss":[14]}
def fine_to_coarse(cid):
    for g, ids in DG.items():
        if cid in ids: return g
    return None

def review(pred_json, out_dir, top_n=60):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pred = json.loads(pathlib.Path(pred_json).read_text("utf-8"))
    cases = []
    for s in pred["samples"]:
        tid = set(s.get("true_classes",[])); pid = set(s.get("predicted_classes",[]))
        img = pathlib.Path(s.get("image_path","")).name
        for cid in tid - pid:
            cases.append({"image":img,"split":s.get("split","val"),"true_class":cid,"predicted_class":"",
                "coarse_true":fine_to_coarse(cid),"coarse_pred":"","candidate_type":"auto_fn",
                "error_type":"localization_miss","review_status":"pending","review_note":"",
                "evidence_path":pred_json,"experiment_id":pred.get("experiment_id",""),"dataset_id":pred.get("dataset_id","")})
        for cid in pid - tid:
            confs = [p.get("confidence",0) for p in s.get("predictions",[]) if p["class_id"]==cid]
            cases.append({"image":img,"split":s.get("split","val"),"true_class":"","predicted_class":cid,
                "coarse_true":"","coarse_pred":fine_to_coarse(cid),"candidate_type":"auto_fp",
                "error_type":"localization_false_positive","review_status":"pending","review_note":"",
                "evidence_path":pred_json,"experiment_id":pred.get("experiment_id",""),"dataset_id":pred.get("dataset_id","")})
    cases = cases[:top_n]
    fields = list(cases[0].keys())
    with (out/"evidence_cases_60.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(cases)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-json", default="runs/run/v3_yolo11s_seg_960_e100/validation_predictions.json")
    p.add_argument("--out-dir", default="results/failure_cases_v3")
    args = p.parse_args()
    print(review(args.pred_json, args.out_dir))

if __name__ == "__main__": main()

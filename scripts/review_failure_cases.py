"""Deep-review failure cases: auto-tag scenarios and error_types."""
import argparse, csv, pathlib

SCENARIO_FIELDS = ["scenario_overexposure","scenario_shadow","scenario_low_contrast","scenario_small_object","scenario_boundary_incomplete","scenario_multiple_defects","scenario_rare_class","evidence_path"]

def review(input_csv, output_csv, evidence_root):
    with open(input_csv, encoding="utf-8-sig") as f:
        cases = list(csv.DictReader(f))
    for r in cases:
        r["review_status"] = "deep_reviewed"
        cid = r.get("true_class","")
        r["scenario_rare_class"] = "1" if cid and int(cid) in [8,13,14] else "0"
        if r.get("candidate_type") == "auto_matched" and r.get("confidence") and float(r["confidence"]) < 0.3:
            r["scenario_small_object"] = "1"; r["scenario_low_contrast"] = "1"
        else:
            r["scenario_small_object"] = "0"; r["scenario_low_contrast"] = "0"
        for fld in ["scenario_overexposure","scenario_shadow","scenario_boundary_incomplete","scenario_multiple_defects"]:
            r[fld] = "?"
        img = pathlib.Path(r.get("image_path","")).name
        exp_id = r.get("experiment_id","")
        r["evidence_path"] = f"{evidence_root}/{exp_id}/validation_predictions.json"
        et = r.get("error_type","")
        if et in ("","missing_detection"): r["error_type"] = "localization_miss"
        elif et == "low_confidence": r["error_type"] = "localization_low_confidence"
    fields = list(cases[0].keys()) if cases else []
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(cases)
    print(f"{len(cases)} cases -> {output_csv}")

def main():
    p = argparse.ArgumentParser(); p.add_argument("--input",default="results/failure_cases/reviewed_cases.csv"); p.add_argument("--output",default="results/failure_cases/reviewed_cases.csv"); p.add_argument("--evidence-root",default="runs/runs"); review(**vars(p.parse_args()))
if __name__ == "__main__": main()
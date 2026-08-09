"""Audit hierarchical error statistics."""
import argparse, csv

def audit(coarse_path, within_path, output):
    coarse = list(csv.DictReader(open(coarse_path, encoding="utf-8-sig")))
    within = list(csv.DictReader(open(within_path, encoding="utf-8-sig")))
    diag = sum(int(r.get("tp",0)) for r in coarse)
    off = sum(int(r.get("fp",0)) + int(r.get("fn",0)) for r in coarse)
    rows = [{"check":"coarse_classes","count":str(len(coarse)),"note":"Expected 6"},{"check":"within_pairs","count":str(len(within)),"note":"Expected >= 750"},{"check":"diagonal_tp","count":str(diag),"note":"Correct coarse"},{"check":"off_diagonal","count":str(off),"note":"Cross-group"}]
    with open(output,"w",newline="",encoding="utf-8-sig") as f:
        w = csv.DictWriter(f,fieldnames=["check","count","note"]); w.writeheader(); w.writerows(rows)
    print(f"diag={diag} off={off} within={len(within)}")

def main():
    p = argparse.ArgumentParser(); p.add_argument("--coarse",default="results/analysis/coarse_class_metrics.csv"); p.add_argument("--within",default="results/analysis/within_group_confusion.csv"); p.add_argument("--output",default="results/error_contract/audit_result.csv"); audit(**vars(p.parse_args()))
if __name__ == "__main__": main()
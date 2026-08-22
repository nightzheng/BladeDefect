"""Compare 15-class aggregation vs independent 6-class retrain."""
import argparse, csv, json, pathlib
from collections import Counter

DG = {"surface_corrosion":[0,1,2,3],"surface_crack":[4,5],"surface_defect":[6,7,8,9],"repair_trace":[10],"blade_damage":[11,12,13],"attachment_loss":[14]}
COARSE_NAME = {0:"surface_corrosion",1:"surface_crack",2:"surface_defect",3:"repair_trace",4:"blade_damage",5:"attachment_loss"}
def f2c(cid):
    for g, ids in DG.items():
        if cid in ids: return g
    return None

def compare(fine_pred_json, coarse_pcm_csv, out_csv):
    pred = json.loads(pathlib.Path(fine_pred_json).read_text("utf-8"))
    cgt=Counter(); cpred=Counter(); ctp=Counter()
    for s in pred["samples"]:
        tcs={f2c(c) for c in s.get("true_classes",[]) if f2c(c)}
        pcs={f2c(c) for c in s.get("predicted_classes",[]) if f2c(c)}
        for g in tcs: cgt[g]+=1
        for g in pcs: cpred[g]+=1
        for g in tcs&pcs: ctp[g]+=1
    rows=[]
    for g in DG:
        gt=cgt.get(g,0); pr=cpred.get(g,0); tp=ctp.get(g,0)
        fn=gt-tp; fp=pr-tp
        prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
        rows.append({"coarse_group":g,"source":"15class_aggregated","gt":gt,"pred":pr,"tp":tp,"fp":fp,"fn":fn,"precision":round(prec,4),"recall":round(rec,4),"f1":round(2*prec*rec/(prec+rec) if prec+rec else 0,4)})
    for r in csv.DictReader(open(coarse_pcm_csv, encoding="utf-8-sig")):
        if r.get("metric_branch")!="mask": continue
        g=COARSE_NAME.get(int(r["class_id"]),"")
        p=float(r.get("precision",0)); rc=float(r.get("recall",0))
        rows.append({"coarse_group":g,"source":"6class_retrained","gt":"","pred":"","tp":"","fp":"","fn":"","precision":round(p,4),"recall":round(rc,4),"f1":round(2*p*rc/(p+rc) if p+rc else 0,4)})
    with open(out_csv,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["coarse_group","source","gt","pred","tp","fp","fn","precision","recall","f1"]); w.writeheader(); w.writerows(rows)
    return out_csv

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fine-pred", default="runs/run/v3_yolo11s_seg_960_e100/validation_predictions.json")
    p.add_argument("--coarse-pcm", default="runs/run/v3_hier_coarse_yolo11s_seg_960_e50/per_class_metrics.csv")
    p.add_argument("--out", default="results/v3_analysis/hierarchy_comparison.csv")
    print(compare(**vars(p.parse_args())))

if __name__ == "__main__": main()

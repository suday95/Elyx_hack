import yaml, os, csv
from datetime import datetime, timedelta
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
CFG = os.path.join(ROOT, "config", "profile.yaml")
RULES = os.path.join(ROOT, "config", "rules.yaml")

def main():
    with open(CFG) as f: prof = yaml.safe_load(f)
    with open(RULES) as f: rules = yaml.safe_load(f)

    daily = pd.read_csv(os.path.join(DATA, "daily.csv"))
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    labs = pd.read_csv(os.path.join(DATA, "labs_quarterly.csv"))
    labs["date"] = pd.to_datetime(labs["date"]).dt.date

    rhr_baseline = prof["baselines"]["rhr_bpm"]

    out = open(os.path.join(DATA, "interventions.csv"), "w", newline="")
    w = csv.writer(out)
    w.writerow(["date","rule_id","trigger_metric","trigger_value","action","owner","follow_up_date","notes"])

    # CV-01: rhr_7d_avg > baseline+5 or hrv drop >15% (approx)
    for i in range(6, len(daily)):
        window = daily.iloc[i-6:i+1]
        rhr7 = window["rhr_bpm"].mean()
        hrv_now = window["hrv_ms"].iloc[-1]
        hrv_prev = window["hrv_ms"].iloc[-2]
        hrv_drop = (hrv_prev - hrv_now) / max(1e-6, hrv_prev)
        if rhr7 > rhr_baseline + 5 or hrv_drop > 0.15:
            d = daily["date"].iloc[i]
            follow = d + timedelta(days=7)
            w.writerow([d.isoformat(),"CV-01","rhr_7d_avg",round(rhr7,1),
                        "deload week; sleep hygiene; -20% intensity","coach",follow.isoformat(),
                        "Auto-trigger based on HR trend"])

    # LIP-02: LDL>130 at quarterly
    for _,r in labs.iterrows():
        if r["ldl_mgdl"] > 130:
            d = r["date"]
            follow = d + timedelta(days=84)
            w.writerow([d.isoformat(),"LIP-02","ldl_mgdl",round(r["ldl_mgdl"],1),
                        "tighten diet; +1 cardio; omega-3","nutritionist",follow.isoformat(),
                        "Triggered by LDL at quarterly"])

    out.close()

if __name__ == "__main__":
    main()

import random, yaml, os, csv
from datetime import datetime, timedelta,date
import pandas as pd
import math

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
CFG = os.path.join(ROOT, "config", "profile.yaml")
RULES = os.path.join(ROOT, "config", "rules.yaml")

def month_diff(d0, d1):
    return (d1.year - d0.year)*12 + (d1.month - d0.month)

def main():
    with open(CFG) as f: prof = yaml.safe_load(f)
    with open(RULES) as f: rules = yaml.safe_load(f)
    random.seed(prof["seed"])

    if isinstance(prof["start_date"], date):
        start = prof["start_date"]
    else:
        start = datetime.strptime(prof["start_date"], "%Y-%m-%d").date()

    end = (start + timedelta(days=30*prof["months"]))
    qweeks = prof["cadence"]["quarterly_labs_weeks"]
    base = prof["baselines"]
    bounds = prof["bounds"]

    daily = pd.read_csv(os.path.join(DATA, "daily.csv"))
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    os.makedirs(DATA, exist_ok=True)
    f = open(os.path.join(DATA, "labs_quarterly.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["date","fasting_glucose_mgdl","ogtt_2h_glucose_mgdl","fasting_insulin_uIUml",
                "total_chol_mgdl","ldl_mgdl","hdl_mgdl","triglycerides_mgdl","apob_mgdl",
                "apoa1_mgdl","lpa_nmoll","crp_mgL","esr_mmhr","alt_uL","ast_uL","creatinine_mgdl",
                "egfr_mlmin","tsh_uIUmL","t3_ngdl","t4_ugdl","cortisol_ugdl","vitd_ngml","b12_pgml",
                "ferritin_ngml","omega3_index_percent"])

    for qw in qweeks:
        qdate = start + timedelta(weeks=qw)
        # adherence last 12 weeks approx
        window_start = qdate - timedelta(days=84)
        dfw = daily[(daily["date"]>=window_start) & (daily["date"]<=qdate)]
        adh = dfw["adherence"].mean() if len(dfw)>0 else 0.75

        # glycemic
        fpg = base["fasting_glucose_mgdl"]
        ogtt2 = base["ogtt_2h_glucose_mgdl"]
        fpg -= random.uniform(*rules["glycemic_monthly"]["fasting_drop_if_good"])*(adh*2)  # scaled
        ogtt2 -= random.uniform(*rules["glycemic_monthly"]["ogtt2h_drop_if_good"])*(adh*2)
        fpg += random.gauss(0, rules["glycemic_monthly"]["noise_std"])
        ogtt2 += random.gauss(0, rules["glycemic_monthly"]["noise_std"])

        # lipids
        months = max(1, month_diff(start, qdate))
        ldl = base["ldl_mgdl"] - months*random.uniform(*rules["lipids_monthly"]["ldl_drop_if_good"])*adh/2
        hdl = base["hdl_mgdl"] + months*random.uniform(*rules["lipids_monthly"]["hdl_gain_if_good"])*adh/2
        tg  = base["triglycerides_mgdl"] - months*random.uniform(*rules["lipids_monthly"]["tg_drop_if_good"])*adh/2
        ldl += random.gauss(0, rules["lipids_monthly"]["noise_std"])
        hdl += random.gauss(0, rules["lipids_monthly"]["noise_std"])
        tg  += random.gauss(0, rules["lipids_monthly"]["noise_std"])
        total = ldl + hdl + tg/5.0

        apob = base["apob_mgdl"] - (ldl - base["ldl_mgdl"])*0.3
        apoa1 = 140 + (hdl- base["hdl_mgdl"])*1.0

        # CRP
        crp = base["crp_mgL"] + random.gauss(0, rules["inflammation"]["noise_std"])
        # mean revert gently
        crp -= (crp - base["crp_mgL"])*rules["inflammation"]["mean_revert_rate"]

        # rest fairly stable
        row = [
            qdate.isoformat(), round(fpg,1), round(ogtt2,1), round(base["fasting_insulin_uIUml"],1),
            round(total,1), round(ldl,1), round(hdl,1), round(tg,1), round(apob,1), round(apoa1,1),
            base["lpa_nmoll"], round(crp,2), 9, 30, 27, 1.0, 95, base["tsh_uIUmL"], base["t3_ngdl"],
            base["t4_ugdl"], base["cortisol_ugdl"], base["vitd_ngml"], base["b12_pgml"],
            base["ferritin_ngml"], base["omega3_index_percent"]
        ]
        w.writerow(row)

    f.close()

if __name__ == "__main__":
    main()

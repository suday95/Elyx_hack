#!/usr/bin/env python3
import random
import yaml
import os
import csv
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
CFG = os.path.join(ROOT, "config", "profile.yaml")
RULES = os.path.join(ROOT, "config", "rules.yaml")

def month_diff(d0, d1):
    return (d1.year - d0.year) * 12 + (d1.month - d0.month)

def clip(x, lo, hi):
    return max(lo, min(hi, x))

def get_bounds(bounds_dict, key, default_lo, default_hi):
    v = bounds_dict.get(key)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    try:
        # single value, treat as hi bound
        return float(default_lo), float(default_hi)
    except Exception:
        return default_lo, default_hi

def main():
    # load config
    with open(CFG) as f:
        prof = yaml.safe_load(f)
    with open(RULES) as f:
        rules = yaml.safe_load(f)

    seed = int(prof.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    # parse start / end properly (use relativedelta for months)
    start = pd.to_datetime(str(prof.get("start_date", "2025-01-01"))).date()
    months = int(prof.get("months", 8))
    end = (pd.Timestamp(start) + relativedelta(months=months)).date() - timedelta(days=1)

    # quarterly weeks array (validate)
    qweeks_raw = prof.get("cadence", {}).get("quarterly_labs_weeks", [])
    # accept numeric weeks like [4,12,24] or month offsets (if you accidentally passed months, handle gracefully)
    qweeks = []
    for v in qweeks_raw:
        try:
            qweeks.append(int(v))
        except Exception:
            continue
    # keep only qweeks that fall within [start, end]
    qweeks = sorted(list(set(qweeks)))
    qdates = []
    for qw in qweeks:
        qdate = start + timedelta(weeks=qw)
        if qdate <= end:
            qdates.append(qdate)
    if len(qdates) == 0:
        # fallback: schedule at ~0, ~12, ~24, ~36 weeks depending on months
        if months >= 6:
            qdates = [start + timedelta(weeks=w) for w in (0, 12, 24) if start + timedelta(weeks=w) <= end]
        else:
            qdates = [start + timedelta(weeks=4)]  # minimal fallback

    # load daily to compute recent adherence windows
    daily_path = os.path.join(DATA, "daily.csv")
    if not os.path.exists(daily_path):
        raise FileNotFoundError(f"{daily_path} not found. Generate daily.csv first.")
    daily = pd.read_csv(daily_path)
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    os.makedirs(DATA, exist_ok=True)
    out_path = os.path.join(DATA, "labs_quarterly.csv")
    f = open(out_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["date","fasting_glucose_mgdl","ogtt_2h_glucose_mgdl","fasting_insulin_uIUml",
                "total_chol_mgdl","ldl_mgdl","hdl_mgdl","triglycerides_mgdl","apob_mgdl",
                "apoa1_mgdl","lpa_nmoll","crp_mgL","esr_mmhr","alt_uL","ast_uL","creatinine_mgdl",
                "egfr_mlmin","tsh_uIUmL","t3_ngdl","t4_ugdl","cortisol_ugdl","vitd_ngml","b12_pgml",
                "ferritin_ngml","omega3_index_percent"])

    # baseline & bounds
    base = prof.get("baselines", {})
    bounds = prof.get("bounds", {})

    # sensible defaults for lab bounds (if not provided)
    fpg_lo, fpg_hi = 50.0, 200.0
    ldl_lo, ldl_hi = 40.0, 300.0
    hdl_lo, hdl_hi = 20.0, 120.0
    tg_lo, tg_hi = 30.0, 1000.0
    crp_lo, crp_hi = 0.0, 20.0

    # override from profile.bounds where present
    if "fasting_glucose_mgdl" in bounds:
        fpg_lo, fpg_hi = bounds.get("fasting_glucose_mgdl", [fpg_lo, fpg_hi])
    if "ldl_mgdl" in bounds:
        ldl_lo, ldl_hi = bounds.get("ldl_mgdl", [ldl_lo, ldl_hi])
    if "hdl_mgdl" in bounds:
        hdl_lo, hdl_hi = bounds.get("hdl_mgdl", [hdl_lo, hdl_hi])
    if "triglycerides_mgdl" in bounds:
        tg_lo, tg_hi = bounds.get("triglycerides_mgdl", [tg_lo, tg_hi])
    if "crp_mgL" in bounds:
        crp_lo, crp_hi = bounds.get("crp_mgL", [crp_lo, crp_hi])

    # helper: get adherence default from profile (fallback to 0.5)
    adh_default = prof.get("adherence", {}).get("base", 0.5)

    # iterate each quarterly date and generate labs
    for qdate in qdates:
        # use recent 12 weeks adherence or fallback
        window_start = qdate - timedelta(days=84)
        dfw = daily[(daily["date"] >= window_start) & (daily["date"] <= qdate)]
        adh = float(dfw["adherence"].mean()) if len(dfw) > 0 else float(adh_default)

        # months since baseline
        months_since = max(1, month_diff(start, qdate))

        # ---------- Glycemic ----------
        fpg = float(base.get("fasting_glucose_mgdl", 100.0))
        ogtt2 = float(base.get("ogtt_2h_glucose_mgdl", 140.0))
        # per-quarter improvement if adherence good: use per-quarter ranges from rules (safer than "*2")
        fpg_drop_range = rules.get("glycemic_monthly", {}).get("fasting_drop_if_good", [1, 3])
        ogtt_drop_range = rules.get("glycemic_monthly", {}).get("ogtt2h_drop_if_good", [2, 6])
        # scale: improvement proportional to adherence and number of quarters passed (months_since/3)
        quarters_passed = max(1, months_since // 3)
        fpg -= (random.uniform(fpg_drop_range[0], fpg_drop_range[1]) * adh * quarters_passed)
        ogtt2 -= (random.uniform(ogtt_drop_range[0], ogtt_drop_range[1]) * adh * quarters_passed)
        # illness spikes: check if any illness in the recent 2-week window
        recent_events = daily[(daily["date"] >= (qdate - timedelta(days=14))) & (daily["date"] <= qdate)]
        # we don't have an events file here; if illness is simulated in daily via higher stress/adherence dips, we let noise capture that
        fpg += random.gauss(0, rules.get("glycemic_monthly", {}).get("noise_std", 0.8))
        ogtt2 += random.gauss(0, rules.get("glycemic_monthly", {}).get("noise_std", 0.8))

        # ---------- Lipids ----------
        ldl = float(base.get("ldl_mgdl", 120.0))
        hdl = float(base.get("hdl_mgdl", 45.0))
        tg = float(base.get("triglycerides_mgdl", 150.0))
        ldl_drop_range = rules.get("lipids_monthly", {}).get("ldl_drop_if_good", [2, 5])
        hdl_gain_range = rules.get("lipids_monthly", {}).get("hdl_gain_if_good", [0.5, 1.5])
        tg_drop_range = rules.get("lipids_monthly", {}).get("tg_drop_if_good", [4, 10])
        # scale change by months_since (conservative: per-month change)
        ldl -= months_since * random.uniform(ldl_drop_range[0], ldl_drop_range[1]) * (adh * 0.33)
        hdl += months_since * random.uniform(hdl_gain_range[0], hdl_gain_range[1]) * (adh * 0.33)
        tg -= months_since * random.uniform(tg_drop_range[0], tg_drop_range[1]) * (adh * 0.33)
        # noise
        ldl += random.gauss(0, rules.get("lipids_monthly", {}).get("noise_std", 1.0))
        hdl += random.gauss(0, rules.get("lipids_monthly", {}).get("noise_std", 1.0))
        tg += random.gauss(0, rules.get("lipids_monthly", {}).get("noise_std", 1.0))
        # total cholesterol approx (Friedewald-ish)
        total = ldl + hdl + tg / 5.0

        # ---------- apolipoproteins & others ----------
        apob = float(base.get("apob_mgdl", 100.0))
        # derive apob change modestly from LDL change if baseline exists
        apob += (ldl - float(base.get("ldl_mgdl", ldl))) * 0.3
        apoa1 = float(base.get("apoa1_mgdl", 140.0))
        apoa1 += (hdl - float(base.get("hdl_mgdl", hdl))) * 0.8

        # ---------- CRP / inflammation ----------
        crp = float(base.get("crp_mgL", 1.0)) + random.gauss(0, rules.get("inflammation", {}).get("noise_std", 0.1))
        # mean revert toward baseline
        mr = float(rules.get("inflammation", {}).get("mean_revert_rate", 0.05))
        crp = float(crp - (crp - float(base.get("crp_mgL", crp))) * mr)
        # small spike for any simulated illness days (if you had events.csv, you'd use it here)

        # ---------- fixed-ish labs (or low-variance) ----------
        fasting_insulin = float(base.get("fasting_insulin_uIUml", 10.0))  # keep baseline; changes are complex
        lpa = float(base.get("lpa_nmoll", 60))
        esr = 9
        alt = 30
        ast = 27
        creatinine = 1.0
        egfr = 95

        tsh = float(base.get("tsh_uIUmL", 2.0))
        t3 = float(base.get("t3_ngdl", 120.0))
        t4 = float(base.get("t4_ugdl", 8.2))
        cortisol = float(base.get("cortisol_ugdl", 16.0))
        vitd = float(base.get("vitd_ngml", 24.0))
        b12 = float(base.get("b12_pgml", 420.0))
        ferritin = float(base.get("ferritin_ngml", 120.0))
        omega3 = float(base.get("omega3_index_percent", 5.0))

        # Clip labs to plausible bounds
        fpg = clip(round(fpg, 1), fpg_lo, fpg_hi)
        ogtt2 = clip(round(ogtt2, 1), 70.0, 300.0)
        ldl = clip(round(ldl, 1), ldl_lo, ldl_hi)
        hdl = clip(round(hdl, 1), hdl_lo, hdl_hi)
        tg = clip(round(tg, 1), tg_lo, tg_hi)
        total = clip(round(total, 1), 80.0, 800.0)
        apob = clip(round(apob, 1), 40.0, 250.0)
        apoa1 = clip(round(apoa1, 1), 60.0, 250.0)
        crp = clip(round(crp, 2), crp_lo, crp_hi)
        vitd = clip(round(vitd, 1), 5.0, 80.0)
        b12 = clip(round(b12, 1), 100.0, 2000.0)
        ferritin = clip(round(ferritin, 1), 10.0, 1000.0)
        omega3 = clip(round(omega3, 2), 1.0, 14.0)

        row = [
            qdate.isoformat(), fpg, ogtt2, round(fasting_insulin,1),
            total, ldl, hdl, tg, apob, apoa1,
            lpa, crp, esr, alt, ast, creatinine,
            egfr, tsh, t3, t4, cortisol, vitd, b12,
            ferritin, omega3
        ]
        w.writerow(row)

    f.close()
    print(f"Wrote labs to {out_path} for {len(qdates)} quarterly dates: {[d.isoformat() for d in qdates]}")

if __name__ == "__main__":
    main()

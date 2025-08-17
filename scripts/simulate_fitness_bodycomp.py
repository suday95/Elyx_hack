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

def get_section(rules, prof, key, subkey, default=None):
    """Prefer rules[key][subkey], fallback to prof[key][subkey], else default."""
    return rules.get(key, {}).get(subkey,
           prof.get(key, {}).get(subkey, default))

def clip(x, lo, hi):
    return max(lo, min(hi, x))

def main():
    # load config
    with open(CFG) as f:
        prof = yaml.safe_load(f)
    with open(RULES) as f:
        rules = yaml.safe_load(f)

    seed = int(prof.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    # parse start / end using exact months
    start = pd.to_datetime(str(prof.get("start_date", "2025-01-01"))).date()
    months = int(prof.get("months", 8))
    end = (pd.Timestamp(start) + relativedelta(months=months)).date() - timedelta(days=1)

    # ensure data dir exists
    os.makedirs(DATA, exist_ok=True)

    # load daily.csv (expected produced earlier)
    daily = pd.read_csv(os.path.join(DATA, "daily.csv"))
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    # open outputs
    ffit = open(os.path.join(DATA, "fitness.csv"), "w", newline="")
    wf = csv.writer(ffit)
    wf.writerow(["date", "vo2max_est", "5km_time_min", "1rm_deadlift_kg", "1rm_squat_kg", "grip_strength_kg", "fms_score", "spirometry_fev1_L"])

    fbc = open(os.path.join(DATA, "body_comp.csv"), "w", newline="")
    wb = csv.writer(fbc)
    wb.writerow(["date", "dexa_bodyfat_percent", "dexa_lean_mass_kg", "bone_density_tscore"])

    # initial states from profile
    vo2 = float(prof["baselines"].get("vo2max_est", 38.0))
    grip = float(prof["baselines"].get("grip_strength_kg", 40.0))
    fms = float(prof["baselines"].get("fms_score", 12))
    fev1 = float(prof["baselines"].get("spirometry_fev1_L", 3.6))
    bodyfat = float(prof["baselines"].get("dexa_bodyfat_percent", 26.0))
    lean = float(prof["baselines"].get("dexa_lean_mass_kg", 60.5))
    bdt = float(prof["baselines"].get("bone_density_tscore", 0.2))

    # bounds fallback helpers
    vo2_bounds = prof.get("bounds", {}).get("vo2max_est", [30, 50])
    bf_bounds = prof.get("bounds", {}).get("dexa_bodyfat_percent", [18, 28])
    # coerce to floats
    vo2_min, vo2_max = float(vo2_bounds[0]), float(vo2_bounds[1])
    bf_min, bf_max = float(bf_bounds[0]), float(bf_bounds[1])
    grip_min, grip_max = 20.0, 80.0
    fms_min, fms_max = 0.0, 21.0
    fev1_min, fev1_max = 1.0, 6.0

    day = start
    weeks = 0

    # iterate week-by-week
    while day <= end:
        week_start = day
        week_end = day + timedelta(days=6)
        week_df = daily[(daily["date"] >= week_start) & (daily["date"] <= week_end)]
        adh = float(week_df["adherence"].mean()) if len(week_df) > 0 else float(prof.get("adherence", {}).get("base", 0.5))

        # compute sessions: count days with active_minutes>35 as session days
        cardio_sessions = int((week_df["active_minutes"] > 35).sum()) if len(week_df) > 0 else 0
        # strength proxy: days with soreness>3 (not perfect but simple)
        strength_sessions = int((week_df["soreness"] > 3).sum()) if len(week_df) > 0 else 0

        # VO2 logic: use rules; vo2_weekly_gain_if_cardio3 may be a range
        vo2_gain_rule = get_section(rules, prof, "fitness", "vo2_weekly_gain_if_cardio3", [0.3, 0.5])
        vo2_loss = float(get_section(rules, prof, "fitness", "vo2_weekly_loss_if_low", 0.2))
        # apply gains or loss
        if cardio_sessions >= 3 and adh > 0.7:
            if isinstance(vo2_gain_rule, (list, tuple)) and len(vo2_gain_rule) == 2:
                delta = random.uniform(vo2_gain_rule[0], vo2_gain_rule[1])
            else:
                delta = float(vo2_gain_rule)
            vo2 += delta
        else:
            vo2 -= vo2_loss

        vo2 = clip(round(vo2, 1), vo2_min, vo2_max)

        # Grip (strength)
        grip_gain_rule = get_section(rules, prof, "fitness", "grip_weekly_gain_if_strength2", [0.2, 0.4])
        if strength_sessions >= 2 and adh > 0.7:
            if isinstance(grip_gain_rule, (list, tuple)) and len(grip_gain_rule) == 2:
                grip += random.uniform(grip_gain_rule[0], grip_gain_rule[1])
            else:
                grip += float(grip_gain_rule)
        grip = clip(round(grip, 1), grip_min, grip_max)

        # FMS: apply every 4 weeks, but skip at week 0 to avoid immediate change
        if weeks > 0 and weeks % 4 == 0 and adh > 0.7:
            fms += float(get_section(rules, prof, "fitness", "fms_gain_per_4w_if_mobility2", 1))
        fms = clip(round(fms, 0), fms_min, fms_max)

        # Body composition: apply monthly-ish (every 4 weeks) AFTER first month (weeks>0)
        if weeks > 0 and weeks % 4 == 0:
            bf_drop_rule = get_section(rules, prof, "body_comp_monthly", "bodyfat_drop", [0.3, 0.6])
            lean_gain = float(get_section(rules, prof, "body_comp_monthly", "lean_mass_gain", 0.2))
            if isinstance(bf_drop_rule, (list, tuple)) and len(bf_drop_rule) == 2:
                bodyfat -= random.uniform(bf_drop_rule[0], bf_drop_rule[1]) * adh
            else:
                bodyfat -= float(bf_drop_rule) * adh
            lean += lean_gain * adh

        # Spirometry monthly-ish
        if weeks > 0 and weeks % 4 == 0:
            spi_gain_rule = get_section(rules, prof, "fitness", "spirometry_monthly_gain", [0.02, 0.05])
            if isinstance(spi_gain_rule, (list, tuple)) and len(spi_gain_rule) == 2:
                fev1 += random.uniform(spi_gain_rule[0], spi_gain_rule[1])
            else:
                fev1 += float(spi_gain_rule)

        # Clip body comp & related metrics
        bodyfat = clip(round(bodyfat, 1), bf_min, bf_max)
        lean = clip(round(lean, 1), 10.0, 200.0)
        bdt = round(float(bdt), 2)
        fev1 = clip(round(fev1, 2), fev1_min, fev1_max)

        # map vo2 to rough 5km time (simple heuristic): lower vo2 -> slower
        # keep output reasonable
        five_k_time = round(30 + max(0, (55 - vo2) * 0.5), 1)  # minutes

        # 1RM approximations (prototype): scale with grip as proxy
        one_rm_dead = int(110 + grip * 0.5)
        one_rm_squat = int(90 + grip * 0.3)

        # write week-end row (use week_end date)
        row_date = (week_start + timedelta(days=6)).isoformat()
        wf.writerow([row_date, vo2, five_k_time, one_rm_dead, one_rm_squat, grip, int(fms), fev1])
        wb.writerow([row_date, bodyfat, lean, bdt])

        # advance one week
        day += timedelta(days=7)
        weeks += 1

    ffit.close()
    fbc.close()
    print(f"Wrote fitness.csv and body_comp.csv to {DATA}")

if __name__ == "__main__":
    main()

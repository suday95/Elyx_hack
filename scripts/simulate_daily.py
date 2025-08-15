#!/usr/bin/env python3
import os
import csv
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import yaml
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
CFG = os.path.join(ROOT, "config", "profile.yaml")
RULES = os.path.join(ROOT, "config", "rules.yaml")


def read_events():
    p = os.path.join(DATA, "events.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    events = {}
    for _, r in df.iterrows():
        try:
            d = pd.to_datetime(r["date"]).date().isoformat()
        except Exception:
            # skip malformed row
            continue
        etype = str(r.get("event_type", "")).strip()
        try:
            intensity = float(r.get("intensity", 1.0))
        except Exception:
            intensity = 1.0
        events.setdefault(d, []).append((etype, intensity))
    return events


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    # load configs
    with open(CFG) as f:
        prof = yaml.safe_load(f)
    with open(RULES) as f:
        rules = yaml.safe_load(f)

    # seeds for reproducibility
    seed = int(prof.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    # helper to pick dynamic params from rules first, then profile
    def get_section(key, subkey, default=None):
        return rules.get(key, {}).get(subkey,
               prof.get(key, {}).get(subkey, default))

    # parse start/end reliably
    start = pd.to_datetime(str(prof.get("start_date", "2025-01-01"))).date()
    months = int(prof.get("months", 8))
    end = (pd.Timestamp(start) + relativedelta(months=months)).date() - timedelta(days=1)

    # ensure data directory
    os.makedirs(DATA, exist_ok=True)

    # load events (travel/illness etc.)
    events = read_events()

    # baseline & bounds
    base = prof.get("baselines", {})
    bounds = prof.get("bounds", {})

    # open CSV writer (schema unchanged)
    out_path = os.path.join(DATA, "daily.csv")
    f = open(out_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["date", "adherence", "steps", "active_minutes", "weight_kg", "rhr_bpm", "hrv_ms",
                "sleep_hours", "sleep_quality", "stress_score", "soreness", "caloric_balance_kcal"])

    # initial states (floats)
    weight = float(base.get("weight_kg", 75.0))
    rhr = float(base.get("rhr_bpm", 65.0))
    hrv = float(base.get("hrv_ms", 40.0))
    sleep_hours = float(base.get("sleep_hours", 6.5))
    sleep_quality = float(base.get("sleep_quality", 3))
    stress = float(base.get("stress_score", 3))

    # counters for plateau and weekly effects
    no_weight_loss_days = 0
    weekly_cardio = 0
    weekly_strength = 0

    day = start
    # helper to detect week boundary (Sunday end)
    def is_week_end(d):
        # choose Sunday as week end (weekday()==6)
        return d.weekday() == 6

    while day <= end:
        key = day.isoformat()
        ev = events.get(key, [])
        # parse travel and illness events (as lists of (type,intensity))
        travel_days = [e for e in ev if e[0] == "travel"]
        illness_days = [e for e in ev if e[0] == "illness"]

        # adherence: from rules/profile with fallback
        adh_base = get_section("adherence", "base", 0.5)
        adh = float(adh_base)
        # apply travel & illness penalties (sum intensities)
        for item in travel_days:
            try:
                _, intn = item
                adh -= float(get_section("adherence", "travel_penalty_per_day", 0.15)) * float(intn)
            except Exception:
                adh -= float(get_section("adherence", "travel_penalty_per_day", 0.15))
        for item in illness_days:
            try:
                _, intn = item
                adh -= float(get_section("adherence", "illness_penalty_per_day", 0.25)) * float(intn)
            except Exception:
                adh -= float(get_section("adherence", "illness_penalty_per_day", 0.25))
        # noise + recovery bonus if no recent issues (simple)
        adh += random.gauss(0, float(get_section("adherence", "noise_std", 0.05)))
        # clamp
        adh = clip(adh, 0.0, 1.0)

        # steps & active_minutes roughly derived from adherence
        steps = int(4000 + adh * 6000 + random.gauss(0, 500))
        active = int(adh * 60 + random.gauss(0, 5))
        active = max(0, active)

        # simple classifier: classify as cardio session if active>30 with probability ~adh
        did_cardio = False
        did_strength = False
        if active > 30 and random.random() < min(1.0, adh + 0.1):
            did_cardio = True
            weekly_cardio += 1
        # small chance of strength session
        if active > 20 and random.random() < 0.2:
            did_strength = True
            weekly_strength += 1

        # sleep: base from rules/profile, travel reduces, noise applies
        sh_base = float(get_section("sleep", "base", base.get("sleep_hours", 6.5)))
        if travel_days:
            drop = get_section("sleep", "travel_drop", [0.5, 1.0])
            # drop can be a range list -> pick uniform
            if isinstance(drop, (list, tuple)) and len(drop) == 2:
                sh_base -= random.uniform(drop[0], drop[1])
            else:
                sh_base -= float(drop)
        sleep_hours = clip(sh_base + random.gauss(0, float(get_section("sleep", "noise_std", 0.4))),
                           float(bounds.get("sleep_hours", 4.5)),
                           float(bounds.get("sleep_hours", 8.5)))
        # sleep_quality heuristic
        sleep_quality = clip(3 + (sleep_hours - 6.5) * 0.4 + random.gauss(0, 0.4), 1, 5)

        # stress influenced by travel/illness + noise
        stress = clip(3 + (1 if travel_days else 0) + (1 if illness_days else 0) + random.gauss(0, 0.5), 1, 5)

        # soreness (0-10)
        soreness = clip((1 if random.random() < 0.3 else 0) + random.gauss(0.5, 1.0), 0, 10)

        # caloric balance & weight dynamics
        caloric_balance = int(-300 * adh + random.gauss(0, 100))

        # weekly_loss default
        weekly_loss_if_high = float(get_section("weight", "weekly_loss_if_high_adherence", 0.25))
        weekly_loss = weekly_loss_if_high * adh if caloric_balance < 0 else 0.0

        # plateau handling
        plateau_days = int(get_section("weight", "plateau_after_days", 14))
        if no_weight_loss_days >= plateau_days:
            weekly_loss = float(get_section("weight", "plateau_weekly_loss", weekly_loss))

        # daily weight change approx
        d_weight = -(weekly_loss / 7.0) + random.gauss(0, float(get_section("weight", "noise_std", 0.2)) / 7.0)
        # travel water gain
        if travel_days:
            d_weight += float(get_section("weight", "travel_water_gain", 0.3)) / 7.0

        prev_weight = weight
        weight = clip(weight + d_weight, float(bounds.get("weight_kg", [50, 200])[0]) if isinstance(bounds.get("weight_kg"), list) else float(bounds.get("weight_kg", 35)),
                      float(bounds.get("weight_kg", [30, 300])[1]) if isinstance(bounds.get("weight_kg"), list) else float(bounds.get("weight_kg", 300)))
        # track stall days
        if weight >= prev_weight - 0.01:
            no_weight_loss_days += 1
        else:
            no_weight_loss_days = 0

        # RHR & HRV dynamics: noise, travel/illness bumps and small improvements with good adherence & sleep
        rhr += random.gauss(0, float(get_section("rhr", "noise_std", 1.0)))
        hrv += random.gauss(0, float(get_section("hrv", "noise_std", 2.0)))
        if travel_days:
            tb = get_section("rhr", "travel_bump", [2, 5])
            if isinstance(tb, (list, tuple)) and len(tb) == 2:
                rhr += random.uniform(tb[0], tb[1])
            else:
                rhr += float(tb)
            hd = get_section("hrv", "travel_drop", [3, 8])
            if isinstance(hd, (list, tuple)) and len(hd) == 2:
                hrv -= random.uniform(hd[0], hd[1])
            else:
                hrv -= float(hd)
        if illness_days:
            ib = get_section("rhr", "illness_bump", [5, 10])
            if isinstance(ib, (list, tuple)) and len(ib) == 2:
                rhr += random.uniform(ib[0], ib[1])
            else:
                rhr += float(ib)
            idrop = get_section("hrv", "illness_drop", [5, 15])
            if isinstance(idrop, (list, tuple)) and len(idrop) == 2:
                hrv -= random.uniform(idrop[0], idrop[1])
            else:
                hrv -= float(idrop)

        # gentle improvements if high adherence & decent sleep
        if adh > 0.75 and sleep_hours > 6.8:
            rhr -= float(get_section("rhr", "weekly_drop_if_cardio3", 0.5)) / 7.0
            gain = get_section("hrv", "weekly_gain_if_good_sleep", [1, 2.5])
            if isinstance(gain, (list, tuple)) and len(gain) == 2:
                hrv += random.uniform(gain[0], gain[1]) / 7.0
            else:
                hrv += float(gain) / 7.0

        # clip to bounds (use keys from profile.bounds)
        rhr_min, rhr_max = bounds.get("rhr_bpm", [40, 120]) if isinstance(bounds.get("rhr_bpm"), list) else (float(bounds.get("rhr_bpm", 40)), float(bounds.get("rhr_bpm", 120)))
        hrv_min, hrv_max = bounds.get("hrv_ms", [10, 200]) if isinstance(bounds.get("hrv_ms"), list) else (float(bounds.get("hrv_ms", 10)), float(bounds.get("hrv_ms", 200)))
        rhr = clip(round(rhr), rhr_min, rhr_max)
        hrv = clip(round(hrv, 1), hrv_min, hrv_max)

        # write today's row (schema unchanged)
        w.writerow([day.isoformat(), round(adh, 3), int(steps), int(active), round(weight, 2),
                    int(round(rhr)), round(hrv, 1), round(sleep_hours, 2),
                    int(round(sleep_quality)), int(round(stress)), int(round(soreness)),
                    int(caloric_balance)])

        # if end of week, reset weekly counters and optionally apply weekly effects (VO2/strength)
        if is_week_end(pd.Timestamp(day)):
            # Example: stub to apply weekly fitness effects later
            # weekly_cardio, weekly_strength can be used to adjust VO2/strength state
            weekly_cardio = 0
            weekly_strength = 0

        day += timedelta(days=1)

    f.close()
    print(f"Wrote daily data to {out_path}")


if __name__ == "__main__":
    main()

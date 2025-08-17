#!/usr/bin/env python3
import random
import yaml
import os
import csv
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
CFG = os.path.join(ROOT, "config", "profile.yaml")

def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def main():
    # load profile
    with open(CFG) as f:
        prof = yaml.safe_load(f)

    seed = int(prof.get("seed", 42))
    random.seed(seed)

    # parse start reliably
    start = pd.to_datetime(str(prof.get("start_date", "2025-01-01"))).date()
    months = int(prof.get("months", 8))
    end = (pd.Timestamp(start) + relativedelta(months=months)).date() - timedelta(days=1)

    # cadence params with safe defaults
    freq = prof.get("cadence", {}).get("frequencies", {})
    travel_every = int(freq.get("travel_every_n_weeks", 4))
    if travel_every <= 0:
        travel_every = 4
    illness_p = float(freq.get("illness_probability_weekly", 0.1))
    illness_p = min(max(illness_p, 0.0), 1.0)

    os.makedirs(DATA, exist_ok=True)
    out_path = os.path.join(DATA, "events.csv")
    out = open(out_path, "w", newline="")
    w = csv.writer(out)
    w.writerow(["date", "event_type", "intensity", "notes"])

    # Travel every n weeks and weekly illness sampling
    cur = start
    week = 0
    trip_count = 0
    travel_days_total = 0
    illness_days_total = 0

    while cur <= end:
        # schedule travel on every nth week (skip week 0 so signup week is not necessarily travel)
        if week % travel_every == 0 and week > 0:
            trip_len = 7  # enforce 1-week business trips per PS
            trip_count += 1
            for i in range(trip_len):
                d = cur + timedelta(days=i)
                if d > end:
                    break
                w.writerow([d.isoformat(), "travel", int(random.randint(1, 3)), "Work travel"])
                travel_days_total += 1

        # illness weekly roll
        if random.random() < illness_p:
            ill_len = random.randint(3, 5)
            start_ill = cur + timedelta(days=random.randint(0, 6))
            for i in range(ill_len):
                d = start_ill + timedelta(days=i)
                if d > end:
                    break
                # allow overlap with travel (realistic), but you can skip if not desired
                w.writerow([d.isoformat(), "illness", int(random.randint(1, 2)), "Viral symptoms"])
                illness_days_total += 1

        cur += timedelta(days=7)
        week += 1

    out.close()
    print(f"Wrote events to {out_path}. Trips: {trip_count}, travel_days: {travel_days_total}, illness_days: {illness_days_total}")

if __name__ == "__main__":
    main()

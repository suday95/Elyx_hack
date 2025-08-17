import random, yaml, os, csv
from datetime import datetime, timedelta,date
import math
import pandas as pd

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
    for _,r in df.iterrows():
        events.setdefault(r["date"], []).append((r["event_type"], r["intensity"]))
    return events

def clip(x, lo, hi):
    return max(lo, min(hi, x))

def main():
    with open(CFG) as f: prof = yaml.safe_load(f)
    with open(RULES) as f: rules = yaml.safe_load(f)
    random.seed(prof["seed"])

    if isinstance(prof["start_date"], date):
        start = prof["start_date"]
    else:
        start = datetime.strptime(prof["start_date"], "%Y-%m-%d").date()

    end = (start + timedelta(days=30*prof["months"]))
    bounds = prof["bounds"]
    base = prof["baselines"]

    events = read_events()

    os.makedirs(DATA, exist_ok=True)
    f = open(os.path.join(DATA, "daily.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["date","adherence","steps","active_minutes","weight_kg","rhr_bpm","hrv_ms",
                "sleep_hours","sleep_quality","stress_score","soreness","caloric_balance_kcal"])

    weight = base["weight_kg"]
    rhr = base["rhr_bpm"]
    hrv = base["hrv_ms"]
    sleep_hours = base["sleep_hours"]
    sleep_quality = base["sleep_quality"]
    stress = base["stress_score"]

    day = start
    # counters for plateaus and sessions
    no_weight_loss_days = 0
    weekly_cardio = 0
    weekly_strength = 0

    while day <= end:
        ev = events.get(day.isoformat(), [])
        travel_days = [e for e in ev if e[0]=="travel"]
        illness_days = [e for e in ev if e=="illness"]

        # adherence
        adh = rules["adherence"]["base"]
        for _,intn in travel_days:
            adh -= rules["adherence"]["travel_penalty_per_day"]*intn
        for _,intn in illness_days:
            adh -= rules["adherence"]["illness_penalty_per_day"]*intn
        adh += random.gauss(0, rules["adherence"]["noise_std"])
        adh = clip(adh, 0.0, 1.0)

        # steps & active minutes roughly from adherence
        steps = int(4000 + adh*6000 + random.gauss(0,500))
        active = int(adh*60 + random.gauss(0,5))
        active = max(0, active)

        # sleep
        sh = rules["sleep"]["base"]
        if travel_days:
            sh -= random.uniform(*rules["sleep"]["travel_drop"])
        sleep_hours = clip(sh + random.gauss(0, rules["sleep"]["noise_std"]),
                           *bounds["sleep_hours"])
        sleep_quality = clip(3 + (sleep_hours-6.5)*0.4 + random.gauss(0,0.4), 1, 5)

        # stress
        stress = clip(3 + (1 if travel_days else 0) + (1 if illness_days else 0) + random.gauss(0,0.5), 1, 5)

        # soreness from random + adherence/strength placeholder
        soreness = clip((1 if random.random()<0.3 else 0) + random.gauss(1,1), 0, 10)

        # weight weekly trend
        # approximate deficit proportional to adherence
        caloric_balance = int(-300*adh + random.gauss(0,100))
        if caloric_balance < 0:
            weekly_loss = rules["weight"]["weekly_loss_if_high_adherence"]*adh
        else:
            weekly_loss = 0.0

        # daily weight change approx
        d_weight = -(weekly_loss/7.0) + random.gauss(0, rules["weight"]["noise_std"]/7.0)
        if travel_days:
            d_weight += rules["weight"]["travel_water_gain"]/7.0
        prev_weight = weight
        weight = clip(weight + d_weight, *bounds["weight_kg"])
        if weight >= prev_weight - 0.01:
            no_weight_loss_days += 1
        else:
            no_weight_loss_days = 0

        # RHR & HRV
        rhr += random.gauss(0, rules["rhr"]["noise_std"])
        hrv += random.gauss(0, rules["hrv"]["noise_std"])
        if travel_days:
            rhr += random.uniform(*rules["rhr"]["travel_bump"])
            hrv -= random.uniform(*rules["hrv"]["travel_drop"])
        if illness_days:
            rhr += random.uniform(*rules["rhr"]["illness_bump"])
            hrv -= random.uniform(*rules["hrv"]["illness_drop"])
        # gentle improvement with adherence & sleep
        if adh > 0.75 and sleep_hours > 6.8:
            rhr -= rules["rhr"]["weekly_drop_if_cardio3"]/7.0
            hrv += random.uniform(*rules["hrv"]["weekly_gain_if_good_sleep"])/7.0

        rhr = clip(rhr, *bounds["rhr_bpm"])
        hrv = clip(hrv, *bounds["hrv_ms"])

        w.writerow([day.isoformat(), round(adh,3), steps, active, round(weight,2),
                    int(round(rhr)), round(hrv,1), round(sleep_hours,2),
                    int(round(sleep_quality)), int(round(stress)), int(round(soreness)),
                    caloric_balance])

        day += timedelta(days=1)

    f.close()

if __name__ == "__main__":
    main()

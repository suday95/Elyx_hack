import random, yaml, os, csv
from datetime import date, datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
CFG = os.path.join(ROOT, "config", "profile.yaml")

def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def main():
    with open(CFG) as f:
        prof = yaml.safe_load(f)
    random.seed(prof["seed"])

    if isinstance(prof["start_date"], date):
        start = prof["start_date"]
    else:
        start = datetime.strptime(prof["start_date"], "%Y-%m-%d").date()

    end = (start + timedelta(days=30*prof["months"]))  # approx 8 months
    freq = prof["cadence"]["frequencies"]
    travel_every = freq["travel_every_n_weeks"]
    illness_p = freq["illness_probability_weekly"]

    os.makedirs(DATA, exist_ok=True)
    out = open(os.path.join(DATA, "events.csv"), "w", newline="")
    w = csv.writer(out)
    w.writerow(["date","event_type","intensity","notes"])

    # Travel every n weeks
    cur = start
    week = 0
    while cur <= end:
        if week % travel_every == 0 and week>0:
            trip_len = random.randint(5,7)
            for i in range(trip_len):
                d = cur + timedelta(days=i)
                if d > end: break
                w.writerow([d.isoformat(),"travel", random.randint(1,3), "Work travel"])
        # Illness weekly roll
        if random.random() < illness_p:
            ill_len = random.randint(3,5)
            start_ill = cur + timedelta(days=random.randint(0,6))
            for i in range(ill_len):
                d = start_ill + timedelta(days=i)
                if d> end: break
                w.writerow([d.isoformat(),"illness", random.randint(1,2), "Viral symptoms"])
        cur += timedelta(days=7)
        week += 1
    out.close()

if __name__ == "__main__":
    main()

import random, yaml, os, csv
from datetime import datetime, timedelta,date
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
CFG = os.path.join(ROOT, "config", "profile.yaml")
RULES = os.path.join(ROOT, "config", "rules.yaml")

def main():
    with open(CFG) as f: prof = yaml.safe_load(f)
    with open(RULES) as f: rules = yaml.safe_load(f)
    random.seed(prof["seed"])

    if isinstance(prof["start_date"], date):
        start = prof["start_date"]
    else:
        start = datetime.strptime(prof["start_date"], "%Y-%m-%d").date()

    end = (start + timedelta(days=30*prof["months"]))

    daily = pd.read_csv(os.path.join(DATA, "daily.csv"))
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    os.makedirs(DATA, exist_ok=True)

    # fitness.csv
    ffit = open(os.path.join(DATA, "fitness.csv"), "w", newline="")
    wf = csv.writer(ffit)
    wf.writerow(["date","vo2max_est","5km_time_min","1rm_deadlift_kg","1rm_squat_kg","grip_strength_kg","fms_score","spirometry_fev1_L"])

    # body_comp.csv
    fbc = open(os.path.join(DATA, "body_comp.csv"), "w", newline="")
    wb = csv.writer(fbc)
    wb.writerow(["date","dexa_bodyfat_percent","dexa_lean_mass_kg","bone_density_tscore"])

    vo2 = prof["baselines"]["vo2max_est"]
    grip = prof["baselines"]["grip_strength_kg"]
    fms  = prof["baselines"]["fms_score"]
    fev1 = prof["baselines"]["spirometry_fev1_L"]
    bodyfat = prof["baselines"]["dexa_bodyfat_percent"]
    lean = prof["baselines"]["dexa_lean_mass_kg"]
    bdt = prof["baselines"]["bone_density_tscore"]

    day = start
    weeks = 0
    while day <= end:
        week_df = daily[(daily["date"]>=day) & (daily["date"]<=(day+timedelta(days=6)))]
        adh = week_df["adherence"].mean() if len(week_df)>0 else 0.75
        # approximate sessions from active_minutes
        cardio_sessions = int((week_df["active_minutes"]>35).sum()/3) if len(week_df)>0 else 0
        strength_sessions = int((week_df["soreness"]>3).sum()/3) if len(week_df)>0 else 0

        if cardio_sessions>=3 and adh>0.7:
            vo2 += random.uniform(*rules["fitness"]["vo2_weekly_gain_if_cardio3"])
        else:
            vo2 -= rules["fitness"]["vo2_weekly_loss_if_low"]
        vo2 = max(prof["bounds"]["vo2max_est"][0], min(prof["bounds"]["vo2max_est"][1], vo2))

        if strength_sessions>=2 and adh>0.7:
            grip += random.uniform(*rules["fitness"]["grip_weekly_gain_if_strength2"])
        grip = max(30, min(70, grip))

        if weeks%4==0 and adh>0.7:
            fms = min(21, fms + rules["fitness"]["fms_gain_per_4w_if_mobility2"])

        if weeks%4==0:
            # body comp monthly
            bodyfat -= random.uniform(*rules["body_comp_monthly"]["bodyfat_drop"])*adh
            lean += rules["body_comp_monthly"]["lean_mass_gain"]*adh

        # Spirometry monthly-ish
        if weeks%4==0:
            fev1 += random.uniform(*rules["fitness"]["spirometry_monthly_gain"])

        wf.writerow([(day+timedelta(days=6)).isoformat(), round(vo2,1), round(30 + (55-vo2)*0.5,1),
                     int(110 + grip*0.5), int(90 + grip*0.3), round(grip,1), int(fms), round(fev1,2)])

        wb.writerow([(day+timedelta(days=6)).isoformat(), round(bodyfat,1), round(lean,1), round(bdt,2)])

        day += timedelta(days=7)
        weeks += 1

    ffit.close(); fbc.close()

if __name__ == "__main__":
    main()

import os, csv
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")

def main():
    daily = pd.read_csv(os.path.join(DATA, "daily.csv"))
    daily["date"] = pd.to_datetime(daily["date"])
    chats = pd.read_csv(os.path.join(DATA, "chats.csv"))
    labs = pd.read_csv(os.path.join(DATA, "labs_quarterly.csv"))
    fitness = pd.read_csv(os.path.join(DATA, "fitness.csv"))

    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    chats["month"] = pd.to_datetime(chats["datetime"]).dt.to_period("M").astype(str)

    kpi = (daily.groupby("month")
           .agg(adherence_avg=("adherence","mean"),
                weight_change_kg=("weight_kg", lambda s: s.iloc[-1]-s.iloc[-2]),
                sleep_avg=("sleep_hours","mean"),
                stress_avg=("stress_score","mean"))
           .reset_index())

    # simplistic session count proxy
    sessions = (daily.assign(sesh=(daily["active_minutes"]>35).astype(int))
                .groupby("month")["sesh"].sum().reset_index(name="sessions_total"))

    # consults approximations from chats
    consults = (chats.groupby("month")["sender"]
                .agg(consults_attended=lambda s: (s!="member").sum(),
                     consults_missed=lambda s: 0) # placeholder
                .reset_index())

    # LDL changes per month approximate from quarterly (forward-fill)
    labs["month"] = pd.to_datetime(labs["date"]).dt.to_period("M").astype(str)
    ldl_m = labs[["month","ldl_mgdl"]].copy()
    ldl_m = ldl_m.set_index("month").reindex(kpi["month"]).ffill().reset_index()
    ldl_m["ldl_change_mgdl"] = ldl_m["ldl_mgdl"].diff().fillna(0)

    # VO2 change per month
    fitness["month"] = pd.to_datetime(fitness["date"]).dt.to_period("M").astype(str)
    vo2_m = fitness.groupby("month")["vo2max_est"].mean().reindex(kpi["month"]).ffill().reset_index()
    vo2_m["vo2max_change"] = vo2_m["vo2max_est"].diff().fillna(0)

    # Rationale coverage: share of interventions with chats referencing rule_id (approx here as 85-95%)
    kpi["rationale_coverage_percent"] = 90

    out = kpi.merge(sessions, on="month", how="left").merge(consults, on="month", how="left") \
             .merge(ldl_m[["month","ldl_change_mgdl"]], on="month", how="left") \
             .merge(vo2_m[["month","vo2max_change"]], on="month", how="left")

    out = out[["month","adherence_avg","sessions_total","consults_attended","consults_missed",
               "weight_change_kg","sleep_avg","stress_avg","ldl_change_mgdl","vo2max_change",
               "rationale_coverage_percent"]]

    out = out.fillna(0)
    out["adherence_avg"] = out["adherence_avg"].round(2)
    out["sleep_avg"] = out["sleep_avg"].round(1)
    out["stress_avg"] = out["stress_avg"].round(1)
    out["weight_change_kg"] = out["weight_change_kg"].round(1)
    out["ldl_change_mgdl"] = out["ldl_change_mgdl"].round(1)
    out["vo2max_change"] = out["vo2max_change"].round(1)

    out.to_csv(os.path.join(DATA, "kpis_monthly.csv"), index=False)

if __name__ == "__main__":
    main()

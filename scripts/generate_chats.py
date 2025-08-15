import random, yaml, os, csv
from datetime import datetime, timedelta,date
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
CFG = os.path.join(ROOT, "config", "profile.yaml")

def main():
    with open(CFG) as f: prof = yaml.safe_load(f)
    random.seed(prof["seed"])
    if isinstance(prof["start_date"], date):
        start = prof["start_date"]
    else:
        start = datetime.strptime(prof["start_date"], "%Y-%m-%d") #may get error check 

    end = start + timedelta(days=30*prof["months"])

    os.makedirs(DATA, exist_ok=True)
    out = open(os.path.join(DATA, "chats.csv"), "w", newline="")
    w = csv.writer(out)
    w.writerow(["datetime","sender","role","message","tags","linked_intervention_id"])

    # Basic weekly pattern: member update + role reply
    cur = start
    while cur <= end:
        # Member
        w.writerow([(cur+timedelta(hours=9)).strftime("%Y-%m-%d %H:%M"),
                    "member","Member",
                    "Weekly update: travel light, stayed on plan about 70%. Sleep was okay.",
                    "progress;weekly",""])
        # Coach reply
        w.writerow([(cur+timedelta(hours=11)).strftime("%Y-%m-%d %H:%M"),
                    "coach","Coach",
                    "Thanks for the update. Keep hydration up on travel days; we will hold volume this week.",
                    "plan;advice",""])
        cur += timedelta(days=7)

    # Interventions anchored messages
    inter_path = os.path.join(DATA, "interventions.csv")
    if os.path.exists(inter_path):
        df = pd.read_csv(inter_path)
        for _,r in df.iterrows():
            t = datetime.strptime(r["date"], "%Y-%m-%d")
            msg = f"Triggered {r['rule_id']} due to {r['trigger_metric']}={r['trigger_value']}. Action: {r['action']}."
            role = "Physician" if r["owner"]=="nutritionist" else "Coach"
            sender = "physician" if r["owner"]=="nutritionist" else "coach"
            w.writerow([(t+timedelta(hours=10)).strftime("%Y-%m-%d %H:%M"),
                        sender, role, msg, "intervention;labs" if r["rule_id"]=="LIP-02" else "intervention",
                        r["rule_id"]])

    out.close()

if __name__ == "__main__":
    main()

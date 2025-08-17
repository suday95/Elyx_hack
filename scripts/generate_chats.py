#!/usr/bin/env python3
import os
import csv
import random
import yaml
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import pytz

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
CFG = os.path.join(ROOT, "config", "profile.yaml")

SGT = pytz.timezone("Asia/Singapore")

def random_time_for_day(base_date):
    """Return timezone-aware datetime in Singapore between 07:00-21:59."""
    hour = random.randint(7, 21)
    minute = random.choice([0, 5, 10, 15, 20, 30, 45])
    dt = datetime(base_date.year, base_date.month, base_date.day, hour, minute)
    return SGT.localize(dt)

def pick_member_message(prof):
    templates = [
        "Quick check-in: travel this week, managed ~{adh}% of plan. Sleep {sleep}h.",
        "Question: Is it ok to do high-intensity on travel days? Also slept {sleep}h last night.",
        "Progress: followed diet ~{adh}%. Steps are down due to travel.",
        "Curious about {topic}. Also, energy low this morning.",
        "Small heads-up — had one high reading on glucose this morning."
    ]
    topics = ["keto vs low-carb", "timing of cardio", "supplements", "sleep supplements"]
    tpl = random.choice(templates)
    msg = tpl.format(
        adh=int(100 * prof.get("adherence", {}).get("base", 0.5)),
        sleep=round(prof.get("baselines", {}).get("sleep_hours", 6.5) + random.uniform(-0.8, 0.8), 1),
        topic=random.choice(topics)
    )
    return msg

def load_interventions(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            return df
        except Exception:
            try:
                return pd.read_csv(path)
            except Exception:
                return None
    return None

def find_linked_intervention(interventions_df, msg_dt):
    """Find an intervention within +/-1 day of msg_dt and return its rule id."""
    if interventions_df is None or interventions_df.empty:
        return ""
    # compare using dates (ignore time)
    msg_date = pd.Timestamp(msg_dt.date())
    candidates = interventions_df[
        (pd.to_datetime(interventions_df["date"]).dt.date >= (msg_date - pd.Timedelta(days=1)).date()) &
        (pd.to_datetime(interventions_df["date"]).dt.date <= (msg_date + pd.Timedelta(days=1)).date())
    ]
    if not candidates.empty:
        # return first matching rule id (string)
        return str(candidates.iloc[0].get("rule_id", ""))
    return ""

def main():
    # load config
    with open(CFG) as f:
        prof = yaml.safe_load(f)

    # seeds for reproducibility
    seed = int(prof.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    # parse start/end
    start = datetime.strptime(str(prof["start_date"]), "%Y-%m-%d")
    # use relativedelta for months
    months = int(prof.get("months", 8))
    end = start + relativedelta(months=months)
    # ensure end is inclusive of last day
    end = end - timedelta(days=1)

    os.makedirs(DATA, exist_ok=True)
    out_path = os.path.join(DATA, "chats.csv")
    interventions_path = os.path.join(DATA, "interventions.csv")

    interventions_df = load_interventions(interventions_path)

    # prepare CSV
    out = open(out_path, "w", newline="")
    w = csv.writer(out)
    w.writerow(["datetime", "sender", "role", "message", "tags", "linked_intervention_id"])

    # weekly window generator: iterate by week blocks
    cur = start
    while cur <= end:
        week_start = cur
        # sample messages in the week: Poisson with lambda ~5
        msgs_this_week = np.random.poisson(lam=5)
        msgs_this_week = max(0, min(int(msgs_this_week), 12))

        for _ in range(msgs_this_week):
            day_offset = random.randint(0, 6)
            day = week_start + timedelta(days=day_offset)
            msg_dt = random_time_for_day(day)

            msg_text = pick_member_message(prof)
            linked_id = find_linked_intervention(interventions_df, msg_dt)

            w.writerow([
                msg_dt.strftime("%Y-%m-%d %H:%M %z"),
                "member",
                "Member",
                msg_text,
                "member-initiated",
                linked_id
            ])

            # team reply sometimes (60% chance), with small delay
            if random.random() < 0.6:
                # reply by coach most of the time; sometimes nutritionist/concierge/physician
                role_choice = random.random()
                if role_choice < 0.75:
                    sender = "coach"; role = "Coach"
                    reply_tag = "reply;coach"
                    reply_texts = [
                        "Got it — noted. Quick tip: prioritize sleep + hydration. We'll follow up.",
                        "Thanks for updating. Keep volume light during travel and track sleep.",
                        "Noted. If glucose spiked, log meals and I'll review trends."
                    ]
                elif role_choice < 0.9:
                    sender = "nutritionist"; role = "Nutritionist"
                    reply_tag = "reply;nutritionist"
                    reply_texts = [
                        "Quick nutrition tip: prioritize lean protein at meals and avoid late carbs.",
                        "I can send a 3-day travel meal plan if you'd like."
                    ]
                else:
                    sender = "concierge"; role = "Concierge"
                    reply_tag = "reply;concierge"
                    reply_texts = [
                        "Noted — I can adjust your schedule and send a hotel workout.",
                        "I'll log this for the coach and book a follow-up if needed."
                    ]
                reply_dt = msg_dt + timedelta(hours=random.randint(1, 6))
                reply_dt = SGT.normalize(reply_dt)
                w.writerow([
                    reply_dt.strftime("%Y-%m-%d %H:%M %z"),
                    sender,
                    role,
                    random.choice(reply_texts),
                    reply_tag,
                    linked_id
                ])

        # advance one week
        cur = week_start + timedelta(days=7)

    # append interventions as messages (owner->sender mapping, attach linked_intervention_id)
    owner_role_map = {
        "coach": ("coach", "Coach"),
        "nutritionist": ("nutritionist", "Nutritionist"),
        "physician": ("physician", "Physician"),
        "doctor": ("physician", "Physician"),
        "concierge": ("concierge", "Concierge")
    }

    if interventions_df is not None and not interventions_df.empty:
        for _, r in interventions_df.iterrows():
            # robust date parse
            try:
                t = pd.to_datetime(r["date"])
            except Exception:
                try:
                    t = datetime.strptime(str(r["date"]), "%Y-%m-%d")
                except Exception:
                    # skip invalid
                    continue
            # localize at 10:00 SGT
            t_local = SGT.localize(datetime(t.year, t.month, t.day, 10, 0))
            msg = (f"Triggered {r.get('rule_id','')} due to {r.get('trigger_metric','')}="
                   f"{r.get('trigger_value','')}. Action: {r.get('action','')}. "
                   f"Follow-up: {r.get('follow_up_date','') or ''}")
            owner = str(r.get("owner", "coach")).lower()
            sender, role = owner_role_map.get(owner, ("coach", "Coach"))
            tags = "intervention;labs" if str(r.get("rule_id","")).startswith("LIP") else "intervention"

            w.writerow([
                t_local.strftime("%Y-%m-%d %H:%M %z"),
                sender,
                role,
                msg,
                tags,
                str(r.get("rule_id",""))
            ])

    out.close()
    print(f"Wrote chats to {out_path}")

if __name__ == "__main__":
    main()

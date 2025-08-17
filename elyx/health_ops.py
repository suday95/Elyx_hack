# streamlit run health-ops-streamlit-app.py
# ──────────────────────────────────────────────────────────────────────────────
# Health Ops Console — Streamlit
# Implements: data ingestion/normalization, caching, and pages:
# Overview • Journey Timeline • Diagnostics & Labs • Daily/Weekly Trends •
# Fitness & Body Composition • Interventions & Rationale • Chats • KPIs & Reports • Screening
#
# Drop your CSVs in ./data with these filenames:
#   member_profile.csv, events.csv, daily.csv, weekly.csv, labs_quarterly.csv,
#   fitness.csv, body_comp.csv, screening_static.csv, interventions.csv,
#   chats.csv, kpis_monthly.csv
#
# Column name expectations are documented per loader; if yours differ, either
# rename columns in CSVs or tweak the mapping dicts below.
# ──────────────────────────────────────────────────────────────────────────────

import io
import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional but nice: Altair for interactive charts
import altair as alt

# ──────────────────────────────────────────────────────────────────────────────
# Config & Theme
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Health Ops Console", layout="wide")
st.title("🏥 Health Ops Console")

# === UI Theme: dark, modern, vivid accents ===================================
# Inject custom CSS for a sleek, black background UI and vivid accent colors.
DARK_CSS = """
<style>
:root{
  --bg:#060607;
  --panel:#071018;
  --muted:#9ca3af;
  --card:#0f1720;
  --accent-1:#00FF6A; /* neon green */
  --accent-2:#00D1FF; /* electric cyan */
  --accent-3:#FFA500; /* vivid orange */
  --accent-4:#7c3aed; /* vivid purple (fallback) */
}
html, body, [data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, var(--bg) 0%, #030305 100% ) !important;
  color: #e6eef8;
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #040406, var(--panel)) !important;
  color: #e6eef8;
}
header, footer {visibility: hidden}
/* Hide Streamlit warning and info boxes */
.stAlert {display: none !important;}
.stAlert > div {display: none !important;}
div[data-testid="stAlert"] {display: none !important;}
.stButton>button { background: linear-gradient(90deg,var(--accent-1), var(--accent-2)); border: none; color: #041022; }
.stDownloadButton>button{ background: linear-gradient(90deg,var(--accent-3), var(--accent-2)); border:none; color:#041022 }
.css-1d391kg { background: rgba(255,255,255,0.02); }
[data-testid="stMetric"] .css-1vg6q84 { color: var(--muted); }
/* Card-like panels */
div.stExplainer, .stMarkdown, .stFrame { background: rgba(255,255,255,0.02); border-radius: 12px; padding: 8px;}
/* Tweak dataframe header */
.stDataFrame table thead th { background: rgba(255,255,255,0.03) !important; }
/* Tooltip tweaks for Altair */
.vega-tooltip { background: #08101a; color: #e6eef8; border-radius:6px }
/* Accent badges */
.badge-accent { background: linear-gradient(90deg,var(--accent-1),var(--accent-2)); color: #041022; padding:4px 8px; border-radius:999px; font-weight:600 }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# Vivid color palette used across charts and tags
VIVID_PALETTE = ["#00FF6A", "#00D1FF", "#FFA500", "#7C3AED", "#06D6A0", "#FF7B00"]  # neon green, electric cyan, vivid orange, purple, mint, coral

# Register an Altair theme that uses vivid palette
def altair_vivid_theme():
    return {
        "config": {
            "range": {"category": VIVID_PALETTE},
            "title": {"color": "#e6eef8"},
            "axis": {"labelColor": "#cbd5e1", "titleColor": "#cbd5e1"},
            "legend": {"labelColor": "#cbd5e1", "titleColor": "#cbd5e1"},
            "view": {"background":"#061018"}
        }
    }
alt.themes.register("vivid_theme", altair_vivid_theme)
alt.themes.enable("vivid_theme")

# Ensure Altair charts use dark background by default
alt.renderers.set_embed_options(actions=False)
alt.data_transformers.disable_max_rows()

# Make Altair charts render with higher contrast tooltips where possible
alt.data_transformers.disable_max_rows()


# === Column mapping config (adapt to your exact headers) ======================
# Provide a mapping of **standard names -> your actual CSV column names**.
# We'll rename your columns to the standard names used across the app.
# You can also override this at runtime in the sidebar by pasting JSON.
DEFAULT_COLUMN_MAP = {
    "member_profile": {
        "member_id": "member_id",
        "name": "name",
        "age": "age",
        "goals": "goals",
        "height_cm": "height_cm",
        "weight_kg": "weight_kg",
        "vo2_baseline": "vo2_baseline",
    },
    "events": {
        "member_id": "member_id",
        "date": "date",
        "event_type": "event_type",
        "label": "notes",
        # extra: intensity
    },
    "daily": {
        "member_id": "member_id",
        "date": "date",
        "RHR": "rhr_bpm",
        "HRV": "hrv_ms",
        "sleep_hours": "sleep_hours",
        "sleep_quality": "sleep_quality",
        "weight": "weight_kg",
        "steps": "steps",
        "active_minutes": "active_minutes",
        "adherence": "adherence",
        # extras: stress_score, soreness, caloric_balance_kcal
    },
    "weekly": {
        "member_id": "member_id",
        "week_start": "week_start",
        "cardio_sessions": "cardio_sessions",
        "strength_sessions": "strength_sessions",
        "adherence": "adherence",
        "stress": "stress",
    },
    "labs_quarterly": {
        "member_id": "member_id",
        "date": "date",
        "LDL": "ldl_mgdl",
        "ApoB": "apob_mgdl",
        "FPG": "fasting_glucose_mgdl",
        "OGTT_2h": "ogtt_2h_glucose_mgdl",
        "CRP": "crp_mgL",
        # many extras preserved: fasting_insulin_uIUml, total_chol_mgdl, hdl_mgdl, triglycerides_mgdl, apoa1_mgdl, lpa_nmoll, esr_mmhr, alt_uL, ast_uL, creatinine_mgdl, egfr_mlmin, tsh_uIUmL, t3_ngdl, t4_ugdl, cortisol_ugdl, vitd_ngml, b12_pgml, ferritin_ngml, omega3_index_percent
    },
    "fitness": {
        "member_id": "member_id",
        "date": "date",
        "VO2max": "vo2max_est",
        "grip_strength": "grip_strength_kg",
        "FMS": "fms_score",
        "cardio_sessions": "cardio_sessions",
        # extras: 5km_time_min, 1rm_deadlift_kg, 1rm_squat_kg, spirometry_fev1_L
    },
    "body_comp": {
        "member_id": "member_id",
        "date": "date",
        "body_fat_pct": "dexa_bodyfat_percent",
        "lean_mass": "dexa_lean_mass_kg",
        # extra: bone_density_tscore kept as-is
    },
    "screening_static": {
        "member_id": "member_id",
        "item": "item",
        "result": "result",
        "date": "date",
        "next_due": "next_due",
    },
    "interventions": {
        "member_id": "member_id",
        "date": "date",
        "rule_id": "rule_id",
        "trigger_metric": "trigger_metric",
        "trigger_value": "trigger_value",
        "action": "action",
        "owner": "owner",
        "follow_up_date": "follow_up_date",
        "notes": "notes",
    },
    "chats": {
        "member_id": "member_id",
        "timestamp": "datetime",
        "text": "message",
        "rule_id": "linked_intervention_id",
        "thread_id": "thread_id",
        "sender": "sender",
        "topic": "topic",
        "tag": "tags",
        # extra: role
    },
    "kpis_monthly": {
        "member_id": "member_id",
        "month": "month",
        "adherence": "adherence_avg",
        "sessions": "sessions_total",
        "LDL_delta": "ldl_change_mgdl",
        "VO2max_delta": "vo2max_change",
        # extras: consults_attended, consults_missed, weight_change_kg, sleep_avg, stress_avg, rationale_coverage_percent
    },
}

# In‑app override container
if "column_map_override" not in st.session_state:
    st.session_state["column_map_override"] = None

def get_column_map() -> dict:
    return st.session_state.get("column_map_override") or DEFAULT_COLUMN_MAP


# Color coding (consistent across pages)
ROLE_COLORS = {
    "physician": "#2563eb",  # blue
    "coach": "#16a34a",      # green
    "nutritionist": "#fb923c",  # orange
    "concierge": "#7c3aed",  # purple
}
EVENT_COLORS = {
    "travel": "#14b8a6",  # teal
    "illness": "#ef4444",  # red
}
INTERVENTION_COLOR = "#f59e0b"  # gold

DATA_DIR = os.path.join(os.getcwd(), "data")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _parse_date(s):
    if pd.isna(s):
        return pd.NaT
    # Allow multiple formats gracefully
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(s), fmt)
        except Exception:
            continue
    # Fall back to pandas
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def _ensure_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def _exists(fname: str) -> bool:
    return os.path.exists(os.path.join(DATA_DIR, fname))


def _csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Loaders with normalization + helper indices
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_all() -> Dict[str, pd.DataFrame]:
    _ensure_dir()
    loaded = {}
    column_map = get_column_map()

    def apply_mapping(df: pd.DataFrame, csv_key: str) -> pd.DataFrame:
        mapping = (column_map or {}).get(csv_key, {})
        inv = {v: k for k, v in mapping.items() if v in df.columns}
        if inv:
            df = df.rename(columns=inv)
        return df

    def load(name: str, csv_key: str, required_cols=None):
        """Load a CSV, apply column mapping, parse dates, and auto-fill member_id for single-member datasets.
        Returns a dataframe.
        """
        fpath = os.path.join(DATA_DIR, name)
        if not os.path.exists(fpath):
            return pd.DataFrame()
        df = _csv(fpath)
        df = apply_mapping(df, csv_key)
        # Date/time parsing
        for col in [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "timestamp"]) ]:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
        # Warn if required columns missing (but don't fail)
        if required_cols:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.warning(f"{name}: missing expected columns {missing}. Present: {list(df.columns)}")
        # Auto-fill member_id for single-member workflows
        if "member_id" not in df.columns:
            # prefer member_profile if present
            if "member_profile" in loaded and not loaded["member_profile"].empty and "member_id" in loaded["member_profile"].columns:
                default_mid = loaded["member_profile"].member_id.iloc[0]
                df["member_id"] = default_mid
                st.info(f"Auto-filled member_id for {name} from member_profile: {default_mid}")
            else:
                df["member_id"] = "member_1"
                st.info(f"Auto-filled member_id for {name} as 'member_1' (single-member)")
        return df

    loaded["member_profile"] = load("member_profile.csv", "member_profile", ["member_id", "name"])
    loaded["events"] = load("events.csv", "events", ["member_id", "date", "event_type", "label"])
    loaded["daily"] = load("daily.csv", "daily", ["member_id", "date"])
    loaded["weekly"] = load("weekly.csv", "weekly", ["member_id"])  # week_start may be derived
    labs_name = "labs_quarterly.csv" if os.path.exists(os.path.join(DATA_DIR, "labs_quarterly.csv")) else ("lab_quarterly.csv" if os.path.exists(os.path.join(DATA_DIR, "lab_quarterly.csv")) else "labs_quarterly.csv")
    loaded["labs_quarterly"] = load(labs_name, "labs_quarterly", ["member_id", "date"])  # LDL/ApoB optional but recommended
    loaded["fitness"] = load("fitness.csv", "fitness", ["member_id", "date"])
    loaded["body_comp"] = load("body_comp.csv", "body_comp", ["member_id", "date"])
    loaded["screening_static"] = load("screening_static.csv", "screening_static", ["member_id", "item", "result", "date", "next_due"]) 
    loaded["interventions"] = load("interventions.csv", "interventions", ["member_id", "date", "rule_id"]) 
    loaded["chats"] = load("chats.csv", "chats", ["member_id", "timestamp", "text"]) 
    loaded["kpis_monthly"] = load("kpis_monthly.csv", "kpis_monthly", ["member_id", "month"]) 

    loaded = normalize_and_link(loaded)
    return loaded


def normalize_and_link(d: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    # Member index
    for key, df in d.items():
        if not df.empty and "member_id" not in df.columns:
            # Try to recover member_id if a single member exists in profile
            if not d["member_profile"].empty and "member_id" in d["member_profile"].columns:
                mid = d["member_profile"].member_id.iloc[0]
                df["member_id"] = mid
                d[key] = df

    # Month/week indices
    if not d["daily"].empty:
        daily = d["daily"].copy()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()
        # ISO week start (Monday)
        daily["week_start"] = daily["date"] - pd.to_timedelta(daily["date"].dt.weekday, unit="D")
        d["daily"] = daily

    if not d["weekly"].empty:
        weekly = d["weekly"].copy()
        if "week_start" not in weekly.columns and "date" in weekly.columns:
            weekly["week_start"] = weekly["date"] - pd.to_timedelta(pd.to_datetime(weekly["date"]).dt.weekday, unit="D")
        d["weekly"] = weekly

    # Fitness & body comp join by date/week (outer join for completeness)
    if not d["fitness"].empty and not d["body_comp"].empty:
        f = d["fitness"][['member_id','date'] + [c for c in d['fitness'].columns if c not in ('member_id','date')]].copy()
        b = d["body_comp"][['member_id','date'] + [c for c in d['body_comp'].columns if c not in ('member_id','date')]].copy()
        joined = pd.merge_asof(
            f.sort_values("date"),
            b.sort_values("date"),
            by="member_id",
            on="date",
            direction="nearest",
            tolerance=pd.Timedelta(days=7),
            suffixes=("_fit", "_body"),
        )
        d["fitness_body"] = joined
    else:
        d["fitness_body"] = pd.DataFrame()

    # Common joins
    # daily ↔ events by date
    if not d["daily"].empty and not d["events"].empty:
        de = pd.merge(d["daily"], d["events"], on=["member_id", "date"], how="left", suffixes=("", "_event"))
        d["daily_events"] = de
    else:
        d["daily_events"] = pd.DataFrame()

    # labs_quarterly ←→ interventions (approx by date via LDL triggers)
    if not d["labs_quarterly"].empty and not d["interventions"].empty:
        labs = d["labs_quarterly"].sort_values("date")
        iv = d["interventions"].sort_values("date")
        link = pd.merge_asof(iv, labs, by="member_id", on="date", direction="nearest", tolerance=pd.Timedelta(days=14), suffixes=("_iv", "_lab"))
        d["interventions_labs"] = link
    else:
        d["interventions_labs"] = pd.DataFrame()

    # interventions ↔ chats via rule_id
    if not d["interventions"].empty and not d["chats"].empty and "rule_id" in d["chats"].columns:
        d["chats_interventions"] = pd.merge(d["chats"], d["interventions"], on=["member_id", "rule_id"], how="left", suffixes=("_chat", "_iv"))
    else:
        d["chats_interventions"] = pd.DataFrame()

    # daily aggregated → weekly (validation)
    if not d["daily"].empty:
        agg = d["daily"].copy()
        num_cols = [c for c in agg.columns if agg[c].dtype != 'O' and c not in ("member_id")]
        weekly_from_daily = agg.groupby(["member_id", "week_start"], as_index=False)[num_cols].mean(numeric_only=True)
        d["weekly_from_daily"] = weekly_from_daily
    else:
        d["weekly_from_daily"] = pd.DataFrame()

    return d


# ──────────────────────────────────────────────────────────────────────────────
# UI Helpers
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def member_header(profile: pd.DataFrame):
    if profile.empty:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, var(--accent-3) 0%, var(--accent-2) 100%);
            color: #041022;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            📁 Upload member_profile.csv to unlock member header.
        </div>
        """, unsafe_allow_html=True)
        return
    row = profile.iloc[0]
    
    # Create a modern header with gradient background
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h2 style="margin: 0; color: white;">👤 {name}</h2>
        <p style="margin: 5px 0; opacity: 0.9;">Age: {age} • Goals: {goals}</p>
    </div>
    """.format(
        name=row.get('name', 'Member'),
        age=int(row.get('age', 0)) if pd.notna(row.get('age')) else '—',
        goals=row.get('goals', '—') if pd.notna(row.get('goals')) else '—'
    ), unsafe_allow_html=True)
    
    # Metrics in a clean row
    cols = st.columns(4)
    with cols[0]:
        height = row.get('height_cm', None)
        if pd.notna(height):
            st.metric("📏 Height", f"{height:.0f} cm")
        else:
            st.metric("📏 Height", "—")
    with cols[1]:
        weight = row.get('weight_kg', None)
        if pd.notna(weight):
            st.metric("⚖️ Weight", f"{weight:.1f} kg")
        else:
            st.metric("⚖️ Weight", "—")
    with cols[2]:
        vo2_base = row.get('vo2_baseline', None)
        if pd.notna(vo2_base):
            st.metric("🫁 Baseline VO₂max", f"{vo2_base:.1f}")
        else:
            st.metric("🫁 Baseline VO₂max", "—")
    with cols[3]:
        # Calculate BMI if both height and weight are available
        if pd.notna(height) and pd.notna(weight) and height > 0:
            bmi = weight / ((height/100) ** 2)
            bmi_status = "Normal" if 18.5 <= bmi < 25 else "Overweight" if 25 <= bmi < 30 else "Obese" if bmi >= 30 else "Underweight"
            st.metric("📊 BMI", f"{bmi:.1f}", help=f"Status: {bmi_status}")
        else:
            st.metric("📊 BMI", "—")


def _latest_value(df: pd.DataFrame, col: str) -> Optional[float]:
    if df.empty or col not in df.columns:
        return None
    s = df.dropna(subset=[col]).sort_values("date")
    if s.empty:
        return None
    return s[col].iloc[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────

def page_overview(d: Dict[str, pd.DataFrame]):
    member_header(d["member_profile"])

    # KPI Cards
    c1, c2, c3, c4, c5 = st.columns(5)

    # Current weight trend (30-day slope)
    wt_slope = "—"
    if not d["daily"].empty and "weight" in d["daily"].columns:
        today = pd.Timestamp.today()
        thirty_days_ago = today - pd.Timedelta(days=30)
        recent = d["daily"].sort_values("date").dropna(subset=["weight"]) \
            .query("date >= @thirty_days_ago")
        if len(recent) >= 2:
            x = (recent["date"] - recent["date"].min()).dt.days.values.reshape(-1,1)
            y = recent["weight"].values
            # Simple slope via polyfit
            slope = np.polyfit(x.ravel(), y.astype(float), 1)[0]
            wt_slope = f"{slope:.2f} kg/30d"
    with c1:
        st.metric("Weight trend (30d slope)", wt_slope)

    # Latest LDL / ApoB
    ldl = _latest_value(d["labs_quarterly"], "LDL")
    apob = _latest_value(d["labs_quarterly"], "ApoB")
    with c2:
        st.metric("Latest LDL", f"{ldl:.0f} mg/dL" if ldl is not None else "—")
    with c3:
        st.metric("Latest ApoB", f"{apob:.0f} mg/dL" if apob is not None else "—")

    # VO2max current vs baseline
    vo2_curr = _latest_value(d["fitness"], "VO2max")
    vo2_base = None
    if not d["member_profile"].empty and "vo2_baseline" in d["member_profile"].columns:
        vo2_baseline_series = d["member_profile"].get("vo2_baseline")
        if vo2_baseline_series is not None and not vo2_baseline_series.empty:
            vo2_base = vo2_baseline_series.iloc[0]
    with c4:
        delta = (vo2_curr - vo2_base) if (vo2_curr is not None and vo2_base is not None) else None
        st.metric("VO₂max (vs baseline)", f"{vo2_curr:.1f}" if vo2_curr is not None else "—", \
                  f"{delta:+.1f}" if delta is not None else None)

    # Adherence last 30 days
    adhere = "—"
    if not d["daily"].empty and "adherence" in d["daily"].columns:
        today = pd.Timestamp.today()
        thirty_days_ago = today - pd.Timedelta(days=30)
        recent = d["daily"].query("date >= @thirty_days_ago")
        if not recent.empty:
            adhere = f"{recent['adherence'].mean():.0f}%"
    with c5:
        st.metric("Adherence (30d)", adhere)

    st.divider()

    # Multi-metric trend selector
    st.subheader("Multi‑metric trends")
    metrics = [c for c in ("RHR", "HRV", "sleep_hours") if c in d["daily"].columns]
    if metrics:
        timeframe = st.selectbox("Timeframe", ["30d", "90d", "180d", "All"], index=0)
        df = d["daily"].sort_values("date").copy()
        if timeframe != "All" and timeframe is not None:
            days = int(timeframe.rstrip("d"))
            today = pd.Timestamp.today()
            days_ago = today - pd.Timedelta(days=days)
            df = df.query("date >= @days_ago")
        long = df.melt(id_vars=["date"], value_vars=metrics, var_name="metric", value_name="value").dropna()

        # Interactive selection for highlighting
        nearest = alt.selection(type="single", nearest=True, on="mouseover",
                                fields=["date"], empty="none")

        base = alt.Chart(long).mark_line(interpolate='monotone', point=False).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color("metric:N", scale=alt.Scale(range=VIVID_PALETTE), legend=alt.Legend(title="Metric")),
            tooltip=["date:T", "metric:N", "value:Q"],
        ).properties(height=340)

        # area under line for visual weight
        area = alt.Chart(long).mark_area(opacity=0.06, interpolate='monotone').encode(
            x="date:T",
            y=alt.Y("value:Q"),
            color=alt.Color("metric:N", scale=alt.Scale(range=VIVID_PALETTE)),
        )

        points = base.mark_point(size=45).encode(opacity=alt.value(0))
        selectors = alt.Chart(long).mark_point().encode(x='date:T', opacity=alt.value(0)).add_selection(nearest)

        labels = base.mark_text(align='left', dx=7, dy=-7).encode(text=alt.condition(nearest, 'value:Q', alt.value(' ')))

        chart = (area + base + points + labels + selectors).interactive()
        st.altair_chart(chart, use_container_width=True)

        # Small summary cards under chart: latest, delta 30d
        latest_date = df['date'].max()
        latest_vals = df[df['date'] == latest_date][metrics].iloc[0] if latest_date is not pd.NaT else None
        cols = st.columns(len(metrics))
        for i, m in enumerate(metrics):
            with cols[i]:
                val = latest_vals.get(m) if latest_vals is not None and m in latest_vals.index else None
                delta = "—"
                try:
                    past = df.query("date >= @latest_date - pd.Timedelta(days=30)").dropna(subset=[m])[m]
                    if len(past) >= 2:
                        delta = f"{(past.iloc[-1] - past.iloc[0]):+.2f}"
                except Exception:
                    delta = "—"
                st.metric(m, f"{val:.2f}" if val is not None else "—", delta)
    else:
        st.info("daily.csv needs columns: RHR, HRV, sleep_hours for this chart.")

    st.divider()
    st.subheader("📅 Timeline: Events • Interventions • Labs")
    build_timeline(d, clickable=True)


def build_timeline(d: Dict[str, pd.DataFrame], clickable: bool=False, filters: Optional[Dict]=None, on_select_key: str="timeline_selection"):
    # Improved timeline: stacked lanes, color-coded badges, and clickable selection
    ev = d["events"][['date','event_type','label','intensity']].copy() if not d["events"].empty else pd.DataFrame(columns=['date','event_type','label','intensity'])
    ev["type"] = ev["event_type"].str.lower()

    iv = d["interventions"][['date','rule_id','owner']].copy() if not d["interventions"].empty else pd.DataFrame(columns=['date','rule_id','owner'])
    iv["type"] = "intervention"
    iv["label"] = iv["rule_id"].astype(str)

    labs = d["labs_quarterly"][['date']].copy() if not d["labs_quarterly"].empty else pd.DataFrame(columns=['date'])
    if not labs.empty:
        labs["type"] = "lab"
        labs["label"] = "Lab"

    tl = pd.concat([
        ev[['date','type','label','intensity']],
        iv[['date','type','label',]],
        labs[['date','type','label']]
    ], ignore_index=True)
    tl = tl.dropna(subset=["date"]).sort_values("date")

    if tl.empty:
        st.info("No timeline data available.")
        return

    # Create a lane index so items don't overlap visually
    tl = tl.reset_index(drop=True)
    tl['lane'] = (tl.index % 3)  # simple lane assignment for visual stacking

    selection = alt.selection_single(fields=['date'], nearest=True, on='click', empty='none')

    chart = alt.Chart(tl).mark_point(filled=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('lane:O', title=None, axis=None),
        color=alt.Color('type:N', scale=alt.Scale(domain=['travel','illness','intervention','lab'], range=[EVENT_COLORS.get('travel'), EVENT_COLORS.get('illness'), INTERVENTION_COLOR, '#6b7280'])),
        shape='type:N',
        size=alt.Size('intensity:Q', legend=None, scale=alt.Scale(range=[80,300])),
        tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('type:N', title='Type'), alt.Tooltip('label:N', title='Label'), alt.Tooltip('intensity:Q', title='Intensity')],
    ).add_selection(selection).properties(height=140)

    st.altair_chart(chart, use_container_width=True)

    if clickable:
        sel = selection
        # If selection made, show details for nearest date
        try:
            # fallback: let user pick from available dates
            dates = tl['date'].sort_values().dt.date.unique()
            selected = st.selectbox('Inspect date', options=dates)
            if selected is not None:
                selected_date = pd.to_datetime(selected)
            else:
                selected_date = pd.Timestamp.today()
            st.session_state[on_select_key] = selected_date
            details_panel(d, selected_date)
        except Exception:
            pass



def details_panel(d: Dict[str, pd.DataFrame], date: pd.Timestamp):
    st.subheader(f"📋 Details for {date.strftime('%B %d, %Y')}")
    left, right = st.columns(2)

    with left:
        st.markdown("### 📊 Daily Metrics")
        if not d["daily"].empty:
            day = d["daily"].query("date == @date")
            if not day.empty:
                # Create daily metrics table
                day_data = day.drop(columns=["member_id", "date"]).iloc[0]
                daily_metrics = []
                
                for metric, value in day_data.items():
                    if pd.notna(value):
                        if metric in ["RHR", "HRV", "sleep_hours", "sleep_quality", "weight", "steps", "active_minutes", "adherence"]:
                            # Format based on metric type
                            if metric == "RHR":
                                daily_metrics.append(["❤️ Resting Heart Rate", f"{value:.0f} bpm", "💓"])
                            elif metric == "HRV":
                                daily_metrics.append(["💓 Heart Rate Variability", f"{value:.0f} ms", "📊"])
                            elif metric == "sleep_hours":
                                sleep_status = "🟢" if value >= 7 else "🟡" if value >= 6 else "🔴"
                                daily_metrics.append(["😴 Sleep Hours", f"{value:.1f} hrs", sleep_status])
                            elif metric == "sleep_quality":
                                quality_status = "🟢" if value >= 7 else "🟡" if value >= 5 else "🔴"
                                daily_metrics.append(["🌙 Sleep Quality", f"{value:.0f}/10", quality_status])
                            elif metric == "weight":
                                daily_metrics.append(["⚖️ Weight", f"{value:.1f} kg", "📏"])
                            elif metric == "steps":
                                steps_status = "🟢" if value >= 10000 else "🟡" if value >= 7500 else "🔴"
                                daily_metrics.append(["👟 Steps", f"{value:,.0f}", steps_status])
                            elif metric == "active_minutes":
                                active_status = "🟢" if value >= 30 else "🟡" if value >= 20 else "🔴"
                                daily_metrics.append(["🏃 Active Minutes", f"{value:.0f} min", active_status])
                            elif metric == "adherence":
                                adherence_status = "🟢" if value >= 80 else "🟡" if value >= 60 else "🔴"
                                daily_metrics.append(["✅ Adherence", f"{value:.0f}%", adherence_status])
                
                if daily_metrics:
                    # Style the table with custom CSS
                    st.markdown("""
                    <style>
                    .daily-table {
                        background: linear-gradient(135deg, var(--accent-2) 0%, var(--accent-4) 100%);
                        border-radius: 10px;
                        padding: 15px;
                        margin: 10px 0;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }
                    .daily-table table {
                        width: 100%;
                        border-collapse: collapse;
                    }
                    .daily-table th, .daily-table td {
                        padding: 8px 12px;
                        text-align: left;
                        border-bottom: 1px solid rgba(255,255,255,0.1);
                    }
                    .daily-table th {
                        background-color: rgba(255,255,255,0.1);
                        font-weight: 600;
                        color: #041022;
                    }
                    .daily-table td {
                        color: #041022;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Display as HTML table
                    html_table = """
                    <div class="daily-table">
                    <table>
                    <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                    """
                    for metric, value, status in daily_metrics:
                        html_table += f"<tr><td>{metric}</td><td>{value}</td><td>{status}</td></tr>"
                    html_table += "</table></div>"
                    
                    st.markdown(html_table, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, var(--accent-2) 0%, var(--accent-4) 100%);
                    color: #041022;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 10px 0;
                    font-weight: 600;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                ">
                    No daily data for this date
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, var(--accent-2) 0%, var(--accent-4) 100%);
                color: #041022;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                font-weight: 600;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
                No daily data available
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📅 Events")
        ev = d["events"].query("date == @date") if not d["events"].empty else pd.DataFrame()
        if not ev.empty:
            for _, event in ev.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        event_icon = "✈️" if event.get('event_type', '').lower() == 'travel' else "🤒" if event.get('event_type', '').lower() == 'illness' else "📝"
                        st.markdown(f"**{event_icon} {event.get('event_type', 'Event')}**")
                    with col2:
                        if pd.notna(event.get('label')):
                            st.markdown(f"*{event['label']}*")
                        if pd.notna(event.get('intensity')):
                            st.caption(f"Intensity: {event['intensity']}")
                    st.divider()
        else:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, var(--accent-2) 0%, var(--accent-4) 100%);
                color: #041022;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                font-weight: 600;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
                No events for this date
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("### 🎯 Interventions & Actions")
        iv = d["interventions"].query("date == @date") if not d["interventions"].empty else pd.DataFrame()
        if not iv.empty:
            for _, intervention in iv.iterrows():
                with st.container():
                    st.markdown(f"**🔧 {intervention.get('rule_id', 'Rule')}**")
                    if pd.notna(intervention.get('action')):
                        st.markdown(f"*Action: {intervention['action']}*")
                    if pd.notna(intervention.get('owner')):
                        st.caption(f"👤 {intervention['owner']}")
                    if pd.notna(intervention.get('notes')):
                        st.markdown(f"📝 {intervention['notes']}")
                    st.divider()
            
            # Related chats
            if not d["chats"].empty and "rule_id" in d["chats"].columns:
                rids = iv["rule_id"].dropna().unique().tolist()
                ch = d["chats"].query("rule_id in @rids")
                if not ch.empty:
                    st.markdown("### 💬 Related Messages")
                    for _, chat in ch.iterrows():
                        with st.container():
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                sender_icon = "👤" if str(chat.get('sender', '')).lower() in ('member', 'client', 'you') else "👨‍⚕️"
                                st.markdown(f"**{sender_icon} {chat.get('sender', 'Unknown')}**")
                            with col2:
                                st.markdown(f"*{chat.get('text', '')[:100]}{'...' if len(str(chat.get('text', ''))) > 100 else ''}*")
                                st.caption(f"📅 {pd.to_datetime(chat['timestamp']).strftime('%H:%M')}")
                            st.divider()
        else:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, var(--accent-3) 0%, var(--accent-2) 100%);
                color: #041022;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                font-weight: 600;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
                No interventions for this date
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics & Labs Page
# ──────────────────────────────────────────────────────────────────────────────

def page_diagnostics(d: Dict[str, pd.DataFrame]):
    st.subheader("📊 Lab Results Dashboard")
    if d["labs_quarterly"].empty:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%);
            color: #041022;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            📁 Upload labs_quarterly.csv to view lab results
        </div>
        """, unsafe_allow_html=True)
    else:
        labs = d["labs_quarterly"].sort_values("date", ascending=False)
        
        # Latest lab results summary table
        latest_labs = labs.iloc[0] if not labs.empty else None
        if latest_labs is not None:
            # Create a table-like display for lab metrics
            lab_metrics = []
            if "LDL" in latest_labs and pd.notna(latest_labs["LDL"]):
                ldl_status = "🟢" if latest_labs["LDL"] < 100 else "🟡" if latest_labs["LDL"] < 130 else "🔴"
                lab_metrics.append(["LDL", f"{latest_labs['LDL']:.0f} mg/dL", ldl_status])
            if "ApoB" in latest_labs and pd.notna(latest_labs["ApoB"]):
                apob_status = "🟢" if latest_labs["ApoB"] < 80 else "🟡" if latest_labs["ApoB"] < 100 else "🔴"
                lab_metrics.append(["ApoB", f"{latest_labs['ApoB']:.0f} mg/dL", apob_status])
            if "FPG" in latest_labs and pd.notna(latest_labs["FPG"]):
                fpg_status = "🟢" if latest_labs["FPG"] < 100 else "🟡" if latest_labs["FPG"] < 126 else "🔴"
                lab_metrics.append(["Fasting Glucose", f"{latest_labs['FPG']:.0f} mg/dL", fpg_status])
            if "CRP" in latest_labs and pd.notna(latest_labs["CRP"]):
                crp_status = "🟢" if latest_labs["CRP"] < 1 else "🟡" if latest_labs["CRP"] < 3 else "🔴"
                lab_metrics.append(["CRP", f"{latest_labs['CRP']:.1f} mg/L", crp_status])
            
            if lab_metrics:
                # Create a styled table
                lab_df = pd.DataFrame(lab_metrics, columns=["Metric", "Value", "Status"])
                st.markdown("#### 📊 Latest Lab Results")
                
                # Style the table with custom CSS
                st.markdown("""
                <style>
                .lab-table {
                    background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%);
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .lab-table table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .lab-table th, .lab-table td {
                    padding: 8px 12px;
                    text-align: left;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }
                .lab-table th {
                    background-color: rgba(255,255,255,0.1);
                    font-weight: 600;
                    color: #041022;
                }
                .lab-table td {
                    color: #041022;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display as HTML table
                html_table = """
                <div class="lab-table">
                <table>
                <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                """
                for metric, value, status in lab_metrics:
                    html_table += f"<tr><td>{metric}</td><td>{value}</td><td>{status}</td></tr>"
                html_table += "</table></div>"
                
                st.markdown(html_table, unsafe_allow_html=True)
        
        st.divider()
        
        # Lab history table with better styling
        st.subheader("📈 Lab History")
        if not labs.empty:
            # Format the dataframe for display
            display_labs = labs.copy()
            display_labs['date'] = display_labs['date'].dt.strftime('%Y-%m-%d')
            
            # Select only numeric columns for display
            numeric_cols = ['LDL', 'ApoB', 'FPG', 'OGTT_2h', 'CRP']
            available_cols = [col for col in numeric_cols if col in display_labs.columns]
            
            if available_cols:
                display_df = display_labs[['date'] + available_cols].copy()
                
                # Add color coding for values
                def color_cells(val, col):
                    if pd.isna(val):
                        return ''
                    if col == 'LDL':
                        if val < 100: return 'background-color: #d4edda; color: #155724'
                        elif val < 130: return 'background-color: #fff3cd; color: #856404'
                        else: return 'background-color: #f8d7da; color: #721c24'
                    elif col == 'ApoB':
                        if val < 80: return 'background-color: #d4edda; color: #155724'
                        elif val < 100: return 'background-color: #fff3cd; color: #856404'
                        else: return 'background-color: #f8d7da; color: #721c24'
                    elif col == 'FPG':
                        if val < 100: return 'background-color: #d4edda; color: #155724'
                        elif val < 126: return 'background-color: #fff3cd; color: #856404'
                        else: return 'background-color: #f8d7da; color: #721c24'
                    elif col == 'CRP':
                        if val < 1: return 'background-color: #d4edda; color: #155724'
                        elif val < 3: return 'background-color: #fff3cd; color: #856404'
                        else: return 'background-color: #f8d7da; color: #721c24'
                    return ''
                
                styled_df = display_df.style.apply(lambda x: [color_cells(val, col) for val, col in zip(x, x.index)], axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("LDL & ApoB over time")
    labs = d["labs_quarterly"].copy()
    if not labs.empty and {"LDL","ApoB","date"}.issubset(labs.columns):
        long = labs.melt(id_vars=["date"], value_vars=["LDL","ApoB"], var_name="metric", value_name="value").dropna()
        chart = alt.Chart(long).mark_line(point=True).encode(
            x="date:T", y="value:Q", color="metric:N", tooltip=["date:T","metric:N","value:Q"]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Need LDL and ApoB columns.")

    st.subheader("Glucose (fasting) & OGTT 2h")
    if not labs.empty and {"FPG","OGTT_2h"}.issubset(labs.columns):
        long = labs.melt(id_vars=["date"], value_vars=["FPG","OGTT_2h"], var_name="metric", value_name="value").dropna()
        st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
            x="date:T", y="value:Q", color="metric:N", tooltip=["date:T","metric:N","value:Q"]
        ).properties(height=300), use_container_width=True)
    else:
        st.info("Add FPG and OGTT_2h columns to labs_quarterly.csv if available.")

    st.subheader("CRP with illness markers")
    if not labs.empty and "CRP" in labs.columns:
        base = alt.Chart(labs).mark_line(point=True).encode(x="date:T", y="CRP:Q")
        # Illness overlays from events
        ev = d["events"]
        if not ev.empty:
            ill = ev.query("event_type.str.lower() == 'illness'", engine='python')
            if not ill.empty:
                overlay = alt.Chart(ill).mark_rule().encode(x="date:T", color=alt.value(EVENT_COLORS['illness']))
                chart = alt.layer(base.properties(height=250), overlay)
                st.altair_chart(chart, use_container_width=True)  # type: ignore
            else:
                st.altair_chart(base.properties(height=250), use_container_width=True)
        else:
            st.altair_chart(base.properties(height=250), use_container_width=True)
    else:
        st.info("Add CRP column to labs_quarterly.csv for this chart.")

    st.divider()
    st.subheader("🔬 Lab Detail Panel")
    if not labs.empty:
        pick = st.selectbox("Select lab run by date", options=labs.sort_values("date")["date"].dt.date.unique())
        if pick is not None:
            sel_date = pd.to_datetime(pick)
        else:
            sel_date = pd.Timestamp.today()
        row = labs.query("date == @sel_date")
        
        if not row.empty:
            lab_data = row.iloc[0]
            st.markdown(f"### 📊 Lab Results from {lab_data['date'].strftime('%B %d, %Y')}")
            
            # Display lab values in metric cards
            col1, col2 = st.columns(2)
            with col1:
                if "LDL" in lab_data and pd.notna(lab_data["LDL"]):
                    ldl_status = "🟢" if lab_data["LDL"] < 100 else "🟡" if lab_data["LDL"] < 130 else "🔴"
                    st.metric("LDL", f"{lab_data['LDL']:.0f} mg/dL", help=f"{ldl_status} {lab_data['LDL']:.0f} mg/dL")
                if "ApoB" in lab_data and pd.notna(lab_data["ApoB"]):
                    apob_status = "🟢" if lab_data["ApoB"] < 80 else "🟡" if lab_data["ApoB"] < 100 else "🔴"
                    st.metric("ApoB", f"{lab_data['ApoB']:.0f} mg/dL", help=f"{apob_status} {lab_data['ApoB']:.0f} mg/dL")
                if "FPG" in lab_data and pd.notna(lab_data["FPG"]):
                    fpg_status = "🟢" if lab_data["FPG"] < 100 else "🟡" if lab_data["FPG"] < 126 else "🔴"
                    st.metric("Fasting Glucose", f"{lab_data['FPG']:.0f} mg/dL", help=f"{fpg_status} {lab_data['FPG']:.0f} mg/dL")
            with col2:
                if "OGTT_2h" in lab_data and pd.notna(lab_data["OGTT_2h"]):
                    ogtt_status = "🟢" if lab_data["OGTT_2h"] < 140 else "🟡" if lab_data["OGTT_2h"] < 200 else "🔴"
                    st.metric("OGTT 2h", f"{lab_data['OGTT_2h']:.0f} mg/dL", help=f"{ogtt_status} {lab_data['OGTT_2h']:.0f} mg/dL")
                if "CRP" in lab_data and pd.notna(lab_data["CRP"]):
                    crp_status = "🟢" if lab_data["CRP"] < 1 else "🟡" if lab_data["CRP"] < 3 else "🔴"
                    st.metric("CRP", f"{lab_data['CRP']:.1f} mg/L", help=f"{crp_status} {lab_data['CRP']:.1f} mg/L")
                if "HDL" in lab_data and pd.notna(lab_data["HDL"]):
                    hdl_status = "🟢" if lab_data["HDL"] >= 40 else "🔴"
                    st.metric("HDL", f"{lab_data['HDL']:.0f} mg/dL", help=f"{hdl_status} {lab_data['HDL']:.0f} mg/dL")
        
        # Interventions triggered near this lab
        iv = d["interventions_labs"]
        iv_near = iv.query("date == @sel_date") if not iv.empty else pd.DataFrame()
        if not iv_near.empty:
            st.markdown("### 🎯 Interventions Triggered (±14d)")
            for _, intervention in iv_near.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"**{intervention['date'].strftime('%Y-%m-%d')}**")
                    with col2:
                        st.markdown(f"**{intervention.get('rule_id', 'Rule')}**")
                        if pd.notna(intervention.get('action')):
                            st.markdown(f"*{intervention['action']}*")
                        if pd.notna(intervention.get('owner')):
                            st.caption(f"👤 {intervention['owner']}")
                    st.divider()
        
        # Linked chats
        if not d["chats"].empty and "rule_id" in d["chats"].columns and not iv_near.empty:
            rids = iv_near["rule_id"].dropna().unique().tolist()
            ch = d["chats"].query("rule_id in @rids")
            if not ch.empty:
                st.markdown("### 💬 Related Communications")
                for _, chat in ch.iterrows():
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            sender_icon = "👤" if str(chat.get('sender', '')).lower() in ('member', 'client', 'you') else "👨‍⚕️"
                            st.markdown(f"**{sender_icon} {chat.get('sender', 'Unknown')}**")
                        with col2:
                            st.markdown(f"*{chat.get('text', '')[:150]}{'...' if len(str(chat.get('text', ''))) > 150 else ''}*")
                            st.caption(f"📅 {pd.to_datetime(chat['timestamp']).strftime('%Y-%m-%d %H:%M')}")
                        st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# Daily / Weekly Trends
# ──────────────────────────────────────────────────────────────────────────────

def page_trends(d: Dict[str, pd.DataFrame]):
    st.subheader("RHR & HRV (dual axis approximation)")
    daily = d["daily"].copy()
    if daily.empty or not {"RHR","HRV","date"}.issubset(daily.columns):
        st.info("daily.csv needs RHR and HRV columns.")
    else:
        base = alt.Chart(daily).transform_fold(["RHR","HRV"], as_=["metric","value"]).mark_line().encode(
            x="date:T", y="value:Q", color="metric:N", tooltip=["date:T","metric:N","value:Q"]
        ).properties(height=280)
        # Shade travel periods
        ev = d["events"]
        if not ev.empty:
            trav = ev.query("event_type.str.lower() == 'travel'", engine='python')
            if not trav.empty:
                rules = alt.Chart(trav).mark_rule().encode(x="date:T", color=alt.value(EVENT_COLORS['travel']))
                chart = alt.layer(base, rules)
                st.altair_chart(chart, use_container_width=True)  # type: ignore
            else:
                st.altair_chart(base, use_container_width=True)
        else:
            st.altair_chart(base, use_container_width=True)

    st.subheader("Sleep: hours & quality")
    if daily.empty or not {"sleep_hours","sleep_quality"}.issubset(daily.columns):
        st.info("daily.csv needs sleep_hours & sleep_quality.")
    else:
        long = daily.melt(id_vars=["date"], value_vars=["sleep_hours","sleep_quality"], var_name="metric", value_name="value").dropna()
        st.altair_chart(alt.Chart(long).mark_line().encode(x="date:T", y="value:Q", color="metric:N").properties(height=250), use_container_width=True)

    st.subheader("Weight trend (7‑day MA)")
    if daily.empty or "weight" not in daily.columns:
        st.info("daily.csv needs weight column.")
    else:
        df = daily.sort_values("date").copy()
        df["weight_ma7"] = df["weight"].rolling(7, min_periods=1).mean()
        base = alt.Chart(df).mark_line().encode(x="date:T", y="weight:Q")
        ma = alt.Chart(df).mark_line(strokeDash=[4,4]).encode(x="date:T", y="weight_ma7:Q")
        chart = alt.layer(base.properties(height=250), ma)
        st.altair_chart(chart, use_container_width=True)  # type: ignore

    st.subheader("Steps / Active minutes")
    if daily.empty or not {"steps","active_minutes"}.issubset(daily.columns):
        st.info("daily.csv needs steps & active_minutes.")
    else:
        long = daily.melt(id_vars=["date"], value_vars=["steps","active_minutes"], var_name="metric", value_name="value").dropna()
        st.altair_chart(alt.Chart(long).mark_bar().encode(x="date:T", y="value:Q", color="metric:N").properties(height=220), use_container_width=True)

    st.divider()
    st.subheader("Weekly aggregation & toggles")
    mode = st.radio("View", ["Daily","Weekly"], horizontal=True)
    if mode == "Weekly":
        wk = d["weekly_from_daily"] if d["weekly_from_daily"].empty else d["weekly_from_daily"].copy()
        if wk.empty:
            st.info("No weekly aggregation available.")
        else:
            # Example: sessions breakdown
            for cols, title in [(["cardio_sessions","strength_sessions"], "Cardio vs Strength sessions/week"),
                                (["adherence","stress"], "Average adherence & stress")]:
                present = [c for c in cols if c in wk.columns]
                if present:
                    long = wk.melt(id_vars=["week_start"], value_vars=present, var_name="metric", value_name="value").dropna()
                    st.altair_chart(alt.Chart(long).mark_line(point=True).encode(x="week_start:T", y="value:Q", color="metric:N").properties(height=240), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Fitness & Body Composition
# ──────────────────────────────────────────────────────────────────────────────

def page_fitness(d: Dict[str, pd.DataFrame]):
    st.subheader("💪 Fitness & Body Composition")
    f = d["fitness"].copy()
    if f.empty:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-4) 100%);
            color: #041022;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            📁 Upload fitness.csv to view fitness progress
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fitness summary table
        latest_fitness = f.sort_values("date", ascending=False).iloc[0] if not f.empty else None
        if latest_fitness is not None:
            # Create fitness metrics table
            fitness_metrics = []
            if "VO2max" in latest_fitness and pd.notna(latest_fitness["VO2max"]):
                fitness_metrics.append(["VO₂max", f"{latest_fitness['VO2max']:.1f} ml/kg/min", "🫁"])
            if "grip_strength" in latest_fitness and pd.notna(latest_fitness["grip_strength"]):
                fitness_metrics.append(["Grip Strength", f"{latest_fitness['grip_strength']:.1f} kg", "💪"])
            if "FMS" in latest_fitness and pd.notna(latest_fitness["FMS"]):
                fms_score = latest_fitness["FMS"]
                fms_status = "🟢" if fms_score >= 14 else "🔴"
                fitness_metrics.append(["FMS Score", f"{fms_score:.0f}", fms_status])
            if "cardio_sessions" in latest_fitness and pd.notna(latest_fitness["cardio_sessions"]):
                fitness_metrics.append(["Cardio Sessions", f"{latest_fitness['cardio_sessions']:.0f}/week", "🏃"])
            
            if fitness_metrics:
                st.markdown("#### 📊 Latest Fitness Metrics")
                
                # Style the table with custom CSS
                st.markdown("""
                <style>
                .fitness-table {
                    background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-4) 100%);
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .fitness-table table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .fitness-table th, .fitness-table td {
                    padding: 8px 12px;
                    text-align: left;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }
                .fitness-table th {
                    background-color: rgba(255,255,255,0.1);
                    font-weight: 600;
                    color: #041022;
                }
                .fitness-table td {
                    color: #041022;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display as HTML table
                html_table = """
                <div class="fitness-table">
                <table>
                <tr><th>Metric</th><th>Value</th><th>Icon</th></tr>
                """
                for metric, value, icon in fitness_metrics:
                    html_table += f"<tr><td>{metric}</td><td>{value}</td><td>{icon}</td></tr>"
                html_table += "</table></div>"
                
                st.markdown(html_table, unsafe_allow_html=True)
        
        st.divider()
        
        # Fitness progress charts
        st.subheader("📈 Fitness Progress")
        for col, title in [("VO2max","VO₂max Estimate"), ("grip_strength","Grip Strength"), ("FMS","FMS Score" )]:
            if col in f.columns:
                chart = alt.Chart(f.dropna(subset=[col])).mark_line(point=True).encode(
                    x="date:T", 
                    y=alt.Y(f"{col}:Q", title=title), 
                    tooltip=["date:T", col]
                )
                st.altair_chart(chart.properties(height=260), use_container_width=True)

    st.subheader("🏃‍♂️ Body Composition")
    b = d["body_comp"].copy()
    if b.empty:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, var(--accent-3) 0%, var(--accent-1) 100%);
            color: #041022;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            📁 Upload body_comp.csv to view body composition data
        </div>
        """, unsafe_allow_html=True)
    else:
        # Body composition summary table
        latest_body = b.sort_values("date", ascending=False).iloc[0] if not b.empty else None
        if latest_body is not None:
            # Create body composition metrics table
            body_metrics = []
            if "body_fat_pct" in latest_body and pd.notna(latest_body["body_fat_pct"]):
                bf_pct = latest_body["body_fat_pct"]
                bf_status = "🟢" if bf_pct < 20 else "🟡" if bf_pct < 25 else "🔴"
                body_metrics.append(["Body Fat %", f"{bf_pct:.1f}%", bf_status])
            if "lean_mass" in latest_body and pd.notna(latest_body["lean_mass"]):
                body_metrics.append(["Lean Mass", f"{latest_body['lean_mass']:.1f} kg", "💪"])
            
            if body_metrics:
                st.markdown("#### 📊 Body Composition")
                
                # Style the table with custom CSS
                st.markdown("""
                <style>
                .body-table {
                    background: linear-gradient(135deg, var(--accent-3) 0%, var(--accent-1) 100%);
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .body-table table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .body-table th, .body-table td {
                    padding: 8px 12px;
                    text-align: left;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }
                .body-table th {
                    background-color: rgba(255,255,255,0.1);
                    font-weight: 600;
                    color: #041022;
                }
                .body-table td {
                    color: #041022;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display as HTML table
                html_table = """
                <div class="body-table">
                <table>
                <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                """
                for metric, value, status in body_metrics:
                    html_table += f"<tr><td>{metric}</td><td>{value}</td><td>{status}</td></tr>"
                html_table += "</table></div>"
                
                st.markdown(html_table, unsafe_allow_html=True)
        
        # Body composition chart
        present = [c for c in ["body_fat_pct","lean_mass"] if c in b.columns]
        if present:
            long = b.melt(id_vars=["date"], value_vars=present, var_name="metric", value_name="value").dropna()
            st.altair_chart(alt.Chart(long).mark_line(point=True).encode(x="date:T", y="value:Q", color="metric:N"), use_container_width=True)

    # Insights box (simple heuristic summary)
    st.subheader("Insights")
    insight = []
    if not f.empty and "VO2max" in f.columns and f["VO2max"].notna().sum() >= 2:
        f2 = f.dropna(subset=["VO2max"]).sort_values("date")
        delta = f2["VO2max"].iloc[-1] - f2["VO2max"].iloc[0]
        weeks = max(1, (f2["date"].iloc[-1] - f2["date"].iloc[0]).days/7)
        insight.append(f"VO₂max {delta:+.1f} over {weeks:.0f} weeks")
    if not f.empty and {"cardio_sessions","date"}.issubset(f.columns):
        avg_cardio = f["cardio_sessions"].tail(12).mean()
        if not np.isnan(avg_cardio):
            insight.append(f"~{avg_cardio:.1f} cardio sessions/week recently")
    st.info("; ".join(insight) if insight else "Add more fitness/body comp data to unlock insights.")


# ──────────────────────────────────────────────────────────────────────────────
# Interventions & Rationale
# ──────────────────────────────────────────────────────────────────────────────

def page_interventions(d: Dict[str, pd.DataFrame]):
    st.subheader("🎯 Interventions & Clinical Actions")
    iv = d["interventions"].copy()
    if iv.empty:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, var(--accent-3) 0%, var(--accent-2) 100%);
            color: #041022;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            📁 Upload interventions.csv to view clinical interventions
        </div>
        """, unsafe_allow_html=True)
        return

    # Summary metrics table
    if not iv.empty:
        # Calculate metrics
        total_interventions = len(iv)
        recent_interventions = len(iv[iv['date'] >= pd.Timestamp.today() - pd.Timedelta(days=30)])
        unique_owners = iv['owner'].nunique() if 'owner' in iv.columns else 0
        unique_rules = iv['rule_id'].nunique() if 'rule_id' in iv.columns else 0
        
        # Create metrics table
        metrics_data = [
            ["Total Interventions", total_interventions, "🔧"],
            ["Last 30 Days", recent_interventions, "📅"],
            ["Care Team Members", unique_owners, "👥"],
            ["Active Rules", unique_rules, "📋"]
        ]
        
        st.markdown("#### 📊 Intervention Summary")
        
        # Style the table with custom CSS
        st.markdown("""
        <style>
        .metrics-table {
            background: linear-gradient(135deg, var(--accent-3) 0%, var(--accent-2) 100%);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metrics-table table {
            width: 100%;
            border-collapse: collapse;
        }
        .metrics-table th, .metrics-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metrics-table th {
            background-color: rgba(255,255,255,0.1);
            font-weight: 600;
            color: #041022;
        }
        .metrics-table td {
            color: #041022;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display as HTML table
        html_table = """
        <div class="metrics-table">
        <table>
        <tr><th>Metric</th><th>Value</th><th>Icon</th></tr>
        """
        for metric, value, icon in metrics_data:
            html_table += f"<tr><td>{metric}</td><td>{value}</td><td>{icon}</td></tr>"
        html_table += "</table></div>"
        
        st.markdown(html_table, unsafe_allow_html=True)
    
    st.divider()
    
    # Interventions timeline with better formatting
    st.subheader("📅 Recent Interventions")
    if not iv.empty:
        recent_iv = iv.sort_values("date", ascending=False).head(10)
        
        # Create a more readable display
        for _, intervention in recent_iv.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**{intervention['date'].strftime('%Y-%m-%d')}**")
                    if 'owner' in intervention and pd.notna(intervention['owner']):
                        st.caption(f"👤 {intervention['owner']}")
                with col2:
                    if 'rule_id' in intervention and pd.notna(intervention['rule_id']):
                        st.markdown(f"**Rule:** {intervention['rule_id']}")
                    if 'action' in intervention and pd.notna(intervention['action']):
                        st.markdown(f"**Action:** {intervention['action']}")
                    if 'notes' in intervention and pd.notna(intervention['notes']):
                        st.markdown(f"*{intervention['notes']}*")
                st.divider()

    st.divider()
    st.subheader("Why chain")
    rid = st.selectbox("Pick rule_id", options=iv["rule_id"].dropna().unique())
    row = iv.query("rule_id == @rid").sort_values("date").iloc[-1]
    st.write(row)

    # Trigger metric trend before intervention
    trig = row.get("trigger_metric")
    trig_val = row.get("trigger_value")
    if pd.notna(trig) and trig in d["daily"].columns:
        df = d["daily"][['date', trig]].dropna().sort_values("date")
        # window: 60d before intervention
        t0 = pd.to_datetime(row["date"]) - pd.Timedelta(days=60)
        dfw = df.query("date >= @t0 and date <= @row.date")
        line = alt.Chart(dfw).mark_line().encode(x="date:T", y=alt.Y(f"{trig}:Q", title=trig))
        if pd.notna(trig_val):
            thr = alt.Chart(pd.DataFrame({"y":[trig_val]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
            chart = alt.layer(line.properties(height=260), thr)
            st.altair_chart(chart, use_container_width=True) # type: ignore
        else:
            st.altair_chart(line.properties(height=260), use_container_width=True)
    else:
        st.info("Trigger metric not found in daily.csv")

    # Related chats
    if not d["chats"].empty and "rule_id" in d["chats"].columns:
        ch = d["chats"].query("rule_id == @rid")
        st.subheader("Related chats")
        st.dataframe(ch)

    # Post-intervention changes (HRV 7‑day avg)
    if not d["daily"].empty and "HRV" in d["daily"].columns:
        df = d["daily"][['date','HRV']].dropna().sort_values("date")
        df["HRV_ma7"] = df["HRV"].rolling(7, min_periods=1).mean()
        t0 = pd.to_datetime(row["date"]) - pd.Timedelta(days=30)
        t1 = pd.to_datetime(row["date"]) + pd.Timedelta(days=30)
        win = df.query("date >= @t0 and date <= @t1")
        mark = alt.Chart(pd.DataFrame({"date":[pd.to_datetime(row["date"])]})).mark_rule(color=INTERVENTION_COLOR)
        st.subheader("HRV 7‑day average around intervention")
        chart = alt.layer(alt.Chart(win).mark_line().encode(x="date:T", y="HRV_ma7:Q"), mark.encode(x="date:T"))
        st.altair_chart(chart, use_container_width=True)  # type: ignore

    # Export bundle (CSV subset + rationale markdown)
    st.divider()
    if st.button("Export Audit Bundle (.zip)"):
        buf = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(buf, 'w') as z:
            # Minimal bundle
            z.writestr('intervention_row.csv', iv.query("rule_id == @rid").to_csv(index=False))
            if not d["chats"].empty:
                z.writestr('related_chats.csv', d["chats"].query("rule_id == @rid").to_csv(index=False))
            md = f"""# Audit — {rid}\n\n**Trigger:** {trig} → threshold {trig_val}\n\n**Date:** {row['date']}\n\n**Owner:** {row.get('owner','—')}\n\n**Action:** {row.get('action','—')}\n\n"""
            z.writestr('rationale.md', md)
        st.download_button("Download Bundle", data=buf.getvalue(), file_name=f"audit_{rid}.zip", mime="application/zip")


# ──────────────────────────────────────────────────────────────────────────────
# Chats Page
# ──────────────────────────────────────────────────────────────────────────────

def page_chats(d: Dict[str, pd.DataFrame]):
    st.subheader("💬 Communication Hub")
    ch = d["chats"].copy()
    if ch.empty:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, var(--accent-2) 0%, var(--accent-4) 100%);
            color: #041022;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            📁 Upload chats.csv to view communication history
        </div>
        """, unsafe_allow_html=True)
        return

    # Normalize timestamp and sort
    ch['timestamp'] = pd.to_datetime(ch['timestamp'], errors='coerce')
    ch = ch.sort_values('timestamp')

    # Filters
    if 'tag' in ch.columns:
        tags = sorted([t for t in ch['tag'].dropna().unique().tolist() if t != ''])
        chosen = st.multiselect('Filter by tag', options=tags, default=tags)
        if chosen:
            ch = ch[ch['tag'].isin(chosen)]

    mode = st.radio('Group by', ['Week','Topic'], horizontal=True)
    if mode == 'Week':
        ch['week_start'] = pd.to_datetime(ch['timestamp']).dt.to_period('W').apply(lambda r: r.start_time)
        weeks = sorted(ch['week_start'].unique())
        wk = st.selectbox('Week', options=weeks)
        df = ch.query('week_start == @wk')
    else:
        if 'topic' in ch.columns:
            topics = sorted(ch['topic'].dropna().unique().tolist())
            topic = st.selectbox('Topic', options=topics)
            df = ch.query('topic == @topic')
        else:
            st.info("No 'topic' column; showing all chats.")
            df = ch

    left, mid, right = st.columns([2,5,3])

    with left:
        st.markdown("### 💬 Threads")
        if 'thread_id' in df.columns:
            threads = df.groupby('thread_id')['timestamp'].max().sort_values(ascending=False).index.tolist()
            tid = st.selectbox('Select Thread', options=threads)
            thread = df.query('thread_id == @tid').sort_values('timestamp')
        else:
            tid = None
            thread = df
        
        # Show thread summary instead of raw dataframe
        st.markdown("#### 📋 Thread Summary")
        if 'thread_id' in df.columns:
            thread_summary = df.groupby('thread_id').agg({
                'timestamp': ['min', 'max', 'count'],
                'sender': 'nunique'
            }).reset_index()
            thread_summary.columns = ['Thread ID', 'First Message', 'Last Message', 'Message Count', 'Participants']
            
            for _, row in thread_summary.head(5).iterrows():
                with st.container():
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**Thread {row['Thread ID'][:8]}...**")
                        st.caption(f"📅 {row['First Message'].strftime('%m/%d')} - {row['Last Message'].strftime('%m/%d')}")
                    with col2:
                        st.markdown(f"💬 {row['Message Count']}")
                        st.caption(f"👥 {row['Participants']}")
                    st.divider()
        else:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, var(--accent-2) 0%, var(--accent-4) 100%);
                color: #041022;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                font-weight: 600;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
                No thread information available
            </div>
            """, unsafe_allow_html=True)

    with mid:
        st.markdown("### 💬 Messages")
        if not thread.empty:
            for _, r in thread.iterrows():
                side = 'member' if str(r.get('sender','')).lower() in ('member','client','you') else 'provider'
                bubble_color = 'linear-gradient(90deg,var(--accent-1),var(--accent-2))' if side=='member' else 'rgba(255,255,255,0.04)'
                align = 'right' if side=='member' else 'left'
                
                # Enhanced chat bubble with better styling
                st.markdown(f"""
                <div style='display:flex; justify-content:{'flex-end' if align=='right' else 'flex-start'}; padding:8px; margin:4px 0;'>
                    <div style='
                        max-width:75%; 
                        background:{bubble_color}; 
                        padding:12px 16px; 
                        border-radius:18px; 
                        color:#061018;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        position: relative;
                    '>
                        <div style='font-weight:600; font-size:0.9em; margin-bottom:4px;'>
                            {r.get('sender','Unknown')}
                        </div>
                        <div style='font-size:0.95em; line-height:1.4;'>
                            {r.get('text','')}
                        </div>
                        <div style='font-size:0.75em; color:#9ca3af; margin-top:6px; text-align:right;'>
                            {pd.to_datetime(r['timestamp']).strftime('%m/%d %H:%M')}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, var(--accent-2) 0%, var(--accent-4) 100%);
                color: #041022;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                font-weight: 600;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
                No messages in this thread
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("### 📊 Context")
        if not d['daily'].empty and not thread.empty:
            tmin, tmax = thread['timestamp'].min(), thread['timestamp'].max()
            tmin = pd.to_datetime(tmin) - pd.Timedelta(days=3)
            tmax = pd.to_datetime(tmax) + pd.Timedelta(days=3)
            snap = d['daily'].query('date >= @tmin and date <= @tmax')
            
            if not snap.empty:
                st.markdown("#### 📈 Health Snapshot")
                latest_snap = snap.tail(1).iloc[0] if not snap.empty else None
                if latest_snap is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        if pd.notna(latest_snap.get('RHR')):
                            st.metric("❤️ RHR", f"{latest_snap['RHR']:.0f} bpm")
                        if pd.notna(latest_snap.get('HRV')):
                            st.metric("💓 HRV", f"{latest_snap['HRV']:.0f} ms")
                    with col2:
                        if pd.notna(latest_snap.get('sleep_hours')):
                            st.metric("😴 Sleep", f"{latest_snap['sleep_hours']:.1f} hrs")
                        if pd.notna(latest_snap.get('adherence')):
                            st.metric("✅ Adherence", f"{latest_snap['adherence']:.0f}%")
        
        if 'rule_id' in thread.columns:
            rids = [x for x in thread['rule_id'].dropna().unique().tolist() if x != '']
            if rids and not d['interventions'].empty:
                st.markdown("#### 🎯 Related Interventions")
                related_interventions = d['interventions'].query('rule_id in @rids')
                if not related_interventions.empty:
                    for _, intervention in related_interventions.iterrows():
                        with st.container():
                            st.markdown(f"**🔧 {intervention.get('rule_id', 'Rule')}**")
                            if pd.notna(intervention.get('action')):
                                st.markdown(f"*{intervention['action']}*")
                            st.caption(f"📅 {intervention['date'].strftime('%Y-%m-%d')}")
                            st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# KPIs & Reports Page
# ──────────────────────────────────────────────────────────────────────────────

def page_kpis(d: Dict[str, pd.DataFrame]):
    st.subheader("📊 KPI Dashboard")
    km = d["kpis_monthly"].copy()
    if km.empty:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, var(--accent-4) 0%, var(--accent-1) 100%);
            color: #041022;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            📁 Upload kpis_monthly.csv to view performance metrics
        </div>
        """, unsafe_allow_html=True)
    else:
        # KPI Summary Table
        latest_kpis = km.sort_values("month", ascending=False).iloc[0] if not km.empty else None
        if latest_kpis is not None:
            # Create KPI metrics table
            kpi_metrics = []
            if "adherence" in latest_kpis and pd.notna(latest_kpis["adherence"]):
                adherence_status = "🟢" if latest_kpis["adherence"] >= 80 else "🟡" if latest_kpis["adherence"] >= 60 else "🔴"
                kpi_metrics.append(["Adherence", f"{latest_kpis['adherence']:.0f}%", adherence_status])
            if "sessions" in latest_kpis and pd.notna(latest_kpis["sessions"]):
                kpi_metrics.append(["Total Sessions", f"{latest_kpis['sessions']:.0f}", "📊"])
            if "LDL_delta" in latest_kpis and pd.notna(latest_kpis["LDL_delta"]):
                ldl_delta = latest_kpis["LDL_delta"]
                ldl_status = "🟢" if ldl_delta < 0 else "🔴"
                kpi_metrics.append(["LDL Change", f"{ldl_delta:+.1f} mg/dL", ldl_status])
            if "VO2max_delta" in latest_kpis and pd.notna(latest_kpis["VO2max_delta"]):
                vo2_delta = latest_kpis["VO2max_delta"]
                vo2_status = "🟢" if vo2_delta > 0 else "🔴"
                kpi_metrics.append(["VO₂max Change", f"{vo2_delta:+.1f}", vo2_status])
            
            if kpi_metrics:
                st.markdown("#### 📊 Latest KPI Summary")
                
                # Style the table with custom CSS
                st.markdown("""
                <style>
                .kpi-table {
                    background: linear-gradient(135deg, var(--accent-4) 0%, var(--accent-1) 100%);
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .kpi-table table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .kpi-table th, .kpi-table td {
                    padding: 8px 12px;
                    text-align: left;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }
                .kpi-table th {
                    background-color: rgba(255,255,255,0.1);
                    font-weight: 600;
                    color: #041022;
                }
                .kpi-table td {
                    color: #041022;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display as HTML table
                html_table = """
                <div class="kpi-table">
                <table>
                <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                """
                for metric, value, status in kpi_metrics:
                    html_table += f"<tr><td>{metric}</td><td>{value}</td><td>{status}</td></tr>"
                html_table += "</table></div>"
                
                st.markdown(html_table, unsafe_allow_html=True)
        
        st.divider()
        
        # KPI History Table
        st.subheader("📈 Monthly Performance History")
        if not km.empty:
            display_kpis = km.sort_values("month", ascending=False).copy()
            display_kpis['month'] = display_kpis['month'].dt.strftime('%Y-%m')
            
            # Select key metrics for display
            key_metrics = ['adherence', 'sessions', 'LDL_delta', 'VO2max_delta']
            available_metrics = [col for col in key_metrics if col in display_kpis.columns]
            
            if available_metrics:
                display_df = display_kpis[['month'] + available_metrics].copy()
                
                # Add color coding for performance
                def color_kpi_cells(val, col):
                    if pd.isna(val):
                        return ''
                    if col == 'adherence':
                        if val >= 80: return 'background-color: #d4edda; color: #155724'
                        elif val >= 60: return 'background-color: #fff3cd; color: #856404'
                        else: return 'background-color: #f8d7da; color: #721c24'
                    elif col == 'LDL_delta':
                        if val < 0: return 'background-color: #d4edda; color: #155724'
                        else: return 'background-color: #f8d7da; color: #721c24'
                    elif col == 'VO2max_delta':
                        if val > 0: return 'background-color: #d4edda; color: #155724'
                        else: return 'background-color: #f8d7da; color: #721c24'
                    return ''
                
                styled_df = display_df.style.apply(lambda x: [color_kpi_cells(val, col) for val, col in zip(x, x.index)], axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
        if {"adherence","sessions"}.issubset(km.columns):
            long = km.melt(id_vars=["month"], value_vars=["adherence","sessions"], var_name="metric", value_name="value").dropna()
            st.altair_chart(alt.Chart(long).mark_line(point=True).encode(x="month:T", y="value:Q", color="metric:N").properties(height=260), use_container_width=True)
        if {"LDL_delta","VO2max_delta"}.issubset(km.columns):
            long = km.melt(id_vars=["month"], value_vars=["LDL_delta","VO2max_delta"], var_name="metric", value_name="value").dropna()
            st.altair_chart(alt.Chart(long).mark_bar().encode(x="month:T", y="value:Q", color="metric:N").properties(height=220), use_container_width=True)

    st.divider()
    st.subheader("Generate monthly report")
    if not km.empty:
        mon = st.selectbox("Month", options=km.sort_values("month")["month"].dt.date.unique())
        if mon is not None:
            selected_month = pd.to_datetime(mon)
        else:
            selected_month = pd.Timestamp.today()
        row = km.query("month == @selected_month").iloc[0]
        md = f"""# Monthly Report — {mon}\n\n- **Adherence:** {row.get('adherence','—')}\n- **Sessions:** {row.get('sessions','—')}\n- **LDL Δ:** {row.get('LDL_delta','—')}\n- **VO₂max Δ:** {row.get('VO2max_delta','—')}\n\n## Notes\nAdd narrative here.\n"""
        st.download_button("Download Markdown", data=md.encode('utf-8'), file_name=f"report_{mon}.md", mime="text/markdown")


# ──────────────────────────────────────────────────────────────────────────────
# Screening Page
# ──────────────────────────────────────────────────────────────────────────────

def page_screening(d: Dict[str, pd.DataFrame]):
    st.subheader("🔍 Screening & Prevention Checklist")
    s = d["screening_static"].copy()
    if s.empty:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, var(--accent-2) 0%, var(--accent-3) 100%);
            color: #041022;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            📁 Upload screening_static.csv to view screening schedule
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Summary metrics table
    today = pd.Timestamp.today().normalize()
    s["overdue"] = pd.to_datetime(s["next_due"], errors='coerce') < today
    s["due_soon"] = (pd.to_datetime(s["next_due"], errors='coerce') >= today) & (pd.to_datetime(s["next_due"], errors='coerce') <= today + pd.Timedelta(days=30))
    
    total_screenings = len(s)
    overdue_count = s["overdue"].sum()
    due_soon_count = s["due_soon"].sum()
    completed_count = total_screenings - overdue_count - due_soon_count
    
    # Create screening metrics table
    screening_metrics = [
        ["Total Screenings", total_screenings, "📋"],
        ["Overdue", overdue_count, "⚠️" if overdue_count > 0 else "✅"],
        ["Due Soon (30d)", due_soon_count, "⏰" if due_soon_count > 0 else "✅"],
        ["Up to Date", completed_count, "✅"]
    ]
    
    st.markdown("#### 📊 Screening Summary")
    
    # Style the table with custom CSS
    st.markdown("""
    <style>
    .screening-table {
        background: linear-gradient(135deg, var(--accent-2) 0%, var(--accent-3) 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .screening-table table {
        width: 100%;
        border-collapse: collapse;
    }
    .screening-table th, .screening-table td {
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .screening-table th {
        background-color: rgba(255,255,255,0.1);
        font-weight: 600;
        color: #041022;
    }
    .screening-table td {
        color: #041022;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display as HTML table
    html_table = """
    <div class="screening-table">
    <table>
    <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
    """
    for metric, value, status in screening_metrics:
        html_table += f"<tr><td>{metric}</td><td>{value}</td><td>{status}</td></tr>"
    html_table += "</table></div>"
    
    st.markdown(html_table, unsafe_allow_html=True)
    
    st.divider()
    
    # Screening items with better formatting
    st.subheader("📋 Screening Items")
    
    # Overdue items first
    overdue_items = s[s["overdue"]].sort_values("next_due")
    if not overdue_items.empty:
        st.markdown("### ⚠️ **Overdue Items**")
        for _, item in overdue_items.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.markdown(f"**{item['item']}**")
                with col2:
                    if pd.notna(item['result']):
                        st.markdown(f"Last: {item['result']}")
                with col3:
                    st.markdown(f"**Due:** {item['next_due'].strftime('%Y-%m-%d')}")
                st.divider()
    
    # Due soon items
    due_soon_items = s[s["due_soon"]].sort_values("next_due")
    if not due_soon_items.empty:
        st.markdown("### ⏰ **Due Soon**")
        for _, item in due_soon_items.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.markdown(f"**{item['item']}**")
                with col2:
                    if pd.notna(item['result']):
                        st.markdown(f"Last: {item['result']}")
                with col3:
                    st.markdown(f"**Due:** {item['next_due'].strftime('%Y-%m-%d')}")
                st.divider()
    
    # Up to date items
    up_to_date_items = s[~(s["overdue"] | s["due_soon"])].sort_values("next_due")
    if not up_to_date_items.empty:
        st.markdown("### ✅ **Up to Date**")
        for _, item in up_to_date_items.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.markdown(f"**{item['item']}**")
                with col2:
                    if pd.notna(item['result']):
                        st.markdown(f"Last: {item['result']}")
                with col3:
                    st.markdown(f"**Next:** {item['next_due'].strftime('%Y-%m-%d')}")
                st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────────────────────────────────────

def sidebar_nav() -> str:
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Journey Timeline",
            "Diagnostics & Labs",
            "Daily/Weekly Trends",
            "Fitness & Body Composition",
            "Interventions & Rationale",
            "Chats",
            "KPIs & Reports",
            "Screening",
        ],
        index=0,
    )
    st.sidebar.caption("Data folder: ./data")

    with st.sidebar.expander("Column Mapping (paste JSON)"):
        st.caption("Map standard keys to your actual column names. Example: {\"daily\": {\"RHR\": \"RestingHR\"}}")
        default_json = io.StringIO()
        import json
        print(json.dumps(DEFAULT_COLUMN_MAP, indent=2), file=default_json)
        txt = st.text_area("Overrides (JSON)", value=default_json.getvalue(), height=220)
        try:
            overrides = json.loads(txt)
            if st.button("Apply Mapping"):
                st.session_state["column_map_override"] = overrides
                st.success("Applied. Use the 'R' icon to rerun if needed.")
        except Exception:
            st.info("JSON is invalid — using defaults.")

    return page or "Overview"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.sidebar.success("Cached loads are enabled.")
    data = load_all()
    page = sidebar_nav()

    if page == "Overview":
        page_overview(data)
    elif page == "Journey Timeline":
        st.subheader("Journey timeline (interactive)")
        build_timeline(data, clickable=True)
    elif page == "Diagnostics & Labs":
        page_diagnostics(data)
    elif page == "Daily/Weekly Trends":
        page_trends(data)
    elif page == "Fitness & Body Composition":
        page_fitness(data)
    elif page == "Interventions & Rationale":
        page_interventions(data)
    elif page == "Chats":
        page_chats(data)
    elif page == "KPIs & Reports":
        page_kpis(data)
    elif page == "Screening":
        page_screening(data)


if __name__ == "__main__":
    main()

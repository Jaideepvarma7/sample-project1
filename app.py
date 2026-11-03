"""PhoneWorth - smartphone fair-price predictor.

Run with:  streamlit run app.py

Pages (top tabs):
  1. Predict price   - spec form -> fair price, 10-90% range, top reasons
  2. Deal checker    - compare a shop price against the fair price
  3. Compare         - 2-3 phones side by side
  4. Similar phones  - 5 closest phones by specs
  5. Model report    - metrics vs baseline, scatter, feature importance
"""

from __future__ import annotations

import json
import math
from string import Template

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features.build import (
    CATEGORICAL_FEATURES,
    FEATURES,
    FEATURE_LABELS,
    NUMERIC_FEATURES,
    chip_tier,
    load_clean,
    processor_brand,
)
from src.model.train import META_PATH, MODEL_PATH, OOF_PATH

APP_NAME = "PhoneWorth"
TAGLINE = "Know what a phone should cost before you pay for it."

st.set_page_config(
    page_title=APP_NAME,
    page_icon=":material/smartphone:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================================================
# Theme + CSS
# ==========================================================================

PALETTES = {
    "light": dict(
        bg="#F6F7FB", surface="#FFFFFF", surface2="#F1F2F7", border="#E4E6EE",
        text="#111827", muted="#6B7280", accent="#4F46E5", accent_soft="#EEF2FF",
        good="#15803D", good_soft="#DCFCE7", warn="#B45309", warn_soft="#FEF3C7",
        bad="#B91C1C", bad_soft="#FEE2E2", gray="#9CA3AF", series2="#C2410C",
        shadow="0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.06)",
    ),
    "dark": dict(
        bg="#0E1015", surface="#171A21", surface2="#1F232C", border="#2A2F3A",
        text="#F3F4F6", muted="#9CA3AF", accent="#7C83F5", accent_soft="#23264A",
        good="#4ADE80", good_soft="#14301F", warn="#FBBF24", warn_soft="#3A2A08",
        bad="#F87171", bad_soft="#3B1515", gray="#6B7280", series2="#EA580C",
        shadow="0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.35)",
    ),
}


def theme_type() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # older Streamlit
        return "light"


THEME = theme_type()
P = PALETTES[THEME]

CSS = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root { --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
html, body, .stApp, .stApp * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.stApp { background: $bg; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1240px; }

/* Bordered containers -> cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: $surface; border: 1px solid $border !important; border-radius: 16px;
  box-shadow: $shadow; padding: 0.35rem 0.5rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: $surface2; padding: 4px; border-radius: 12px; border: 1px solid $border; }
.stTabs [data-baseweb="tab"] { height: 38px; border-radius: 9px; padding: 0 16px; font-weight: 500; color: $muted; background: transparent; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: $surface; color: $text; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* Inputs */
div[data-baseweb="select"] > div { border-radius: 10px !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background: $accent; border-color: $accent; }

/* Header */
.pw-header { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin-bottom: 18px; flex-wrap: wrap; }
.pw-brand { font-size: 30px; font-weight: 800; letter-spacing: -0.03em; color: $text; line-height: 1.1; }
.pw-brand span { color: $accent; }
.pw-tagline { color: $muted; font-size: 15px; margin-top: 4px; }
.pw-badge { display: inline-flex; align-items: center; gap: 8px; background: $accent_soft; color: $accent; border: 1px solid $border;
  padding: 7px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; white-space: nowrap; }
.pw-badge .dot { width: 8px; height: 8px; border-radius: 50%; background: $accent; }

/* Generic card */
.pw-card { background: $surface; border: 1px solid $border; border-radius: 16px; padding: 20px 22px; box-shadow: $shadow; margin-bottom: 14px; }
.pw-group-title { font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: $muted; margin: 2px 0 6px 2px; }
.pw-kicker { font-size: 13px; color: $muted; font-weight: 500; }
.pw-price { font-size: 46px; font-weight: 800; letter-spacing: -0.03em; color: $text; line-height: 1.1; margin: 6px 0 2px; }
.pw-sub { color: $muted; font-size: 14px; }
.pw-h { font-size: 16px; font-weight: 700; color: $text; margin: 0 0 10px; }

/* Range bar */
.pw-range { margin-top: 18px; }
.pw-range-track { position: relative; height: 10px; border-radius: 999px; background: $surface2; }
.pw-range-fill { position: absolute; top: 0; height: 10px; border-radius: 999px; background: linear-gradient(90deg, $accent_soft, $accent); opacity: .9; }
.pw-range-marker { position: absolute; top: -5px; width: 20px; height: 20px; margin-left: -10px; border-radius: 50%; background: $surface; border: 4px solid $accent; box-shadow: 0 1px 4px rgba(0,0,0,.2); }
.pw-range-labels { display: flex; justify-content: space-between; color: $muted; font-size: 12.5px; margin-top: 8px; }
.pw-range-labels b { color: $text; font-weight: 600; }

/* Reasons */
.pw-reason { display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-top: 1px solid $border; gap: 12px; }
.pw-reason:first-of-type { border-top: none; }
.pw-reason .lbl { font-size: 14px; color: $text; font-weight: 500; }
.pw-reason .why { font-size: 12.5px; color: $muted; margin-top: 1px; }
.pw-chip { font-size: 13px; font-weight: 700; padding: 4px 10px; border-radius: 999px; white-space: nowrap; }
.pw-chip.up { background: $good_soft; color: $good; }
.pw-chip.down { background: $bad_soft; color: $bad; }

/* Verdict */
.pw-verdict { border-radius: 16px; padding: 22px 24px; border: 1px solid $border; box-shadow: $shadow; margin-bottom: 14px; }
.pw-verdict.good { background: $good_soft; } .pw-verdict.good .v { color: $good; }
.pw-verdict.fair { background: $warn_soft; } .pw-verdict.fair .v { color: $warn; }
.pw-verdict.bad { background: $bad_soft; } .pw-verdict.bad .v { color: $bad; }
.pw-verdict .v { font-size: 34px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }
.pw-verdict .d { color: $text; font-size: 15px; margin-top: 6px; }
.pw-verdict .d b { font-weight: 700; }

/* Phone cards */
.pw-phone { background: $surface; border: 1px solid $border; border-radius: 16px; padding: 16px 16px 14px; box-shadow: $shadow; height: 100%; }
.pw-phone .name { font-weight: 700; font-size: 14.5px; color: $text; line-height: 1.25; min-height: 36px; }
.pw-phone .price { font-size: 21px; font-weight: 800; color: $text; margin: 8px 0 2px; letter-spacing: -0.02em; white-space: nowrap; }
.pw-phone .est { font-size: 12.5px; color: $muted; margin-bottom: 10px; }
.pw-phone .spec { font-size: 12.5px; color: $muted; padding: 4px 0; border-top: 1px solid $border; display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
.pw-phone .spec b { color: $text; font-weight: 600; text-align: right; margin-left: auto; }

/* Metrics */
.pw-metric { background: $surface; border: 1px solid $border; border-radius: 14px; padding: 14px 16px; box-shadow: $shadow; }
.pw-metric .k { font-size: 12px; color: $muted; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }
.pw-metric .v { font-size: 26px; font-weight: 800; color: $text; letter-spacing: -0.02em; margin-top: 2px; }
.pw-metric .s { font-size: 12.5px; color: $muted; }

/* Table */
table.pw-table { width: 100%; border-collapse: collapse; font-size: 14px; }
table.pw-table th { text-align: left; color: $muted; font-weight: 600; font-size: 12.5px; letter-spacing: .03em; padding: 8px 10px; border-bottom: 1px solid $border; }
table.pw-table td { padding: 9px 10px; border-bottom: 1px solid $border; color: $text; }
table.pw-table tr.best td { font-weight: 700; }
table.pw-table td.num, table.pw-table th.num { text-align: right; font-variant-numeric: tabular-nums; }
.pw-note { color: $muted; font-size: 13px; }
</style>
""").substitute(P)

st.markdown(CSS, unsafe_allow_html=True)

# ==========================================================================
# Data + model loading (cached)
# ==========================================================================


@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python run_pipeline.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_explainer():
    import shap

    return shap.TreeExplainer(load_bundle()["model"]["model"])


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    return json.loads(META_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_phones() -> pd.DataFrame:
    df = load_clean()
    df["label"] = df["model"]
    return df


@st.cache_data(show_spinner=False)
def load_oof() -> pd.DataFrame:
    return pd.read_csv(OOF_PATH)


@st.cache_data(show_spinner=False)
def phone_predictions() -> pd.DataFrame:
    """Fair price + range for every phone in the dataset."""
    df = load_phones()
    pred, lo, hi = predict(df[FEATURES])
    out = df.copy()
    out["pred"], out["lo"], out["hi"] = pred, lo, hi
    return out


# ==========================================================================
# Helpers
# ==========================================================================


def inr(x: float, prefix: str = "₹") -> str:
    """Format a number with Indian digit grouping: 1,14,990."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    neg = x < 0
    s = str(int(round(abs(x))))
    if len(s) > 3:
        last3, rest = s[-3:], s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        s = ",".join(parts) + "," + last3
    return ("−" if neg else "") + prefix + s


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def feature_frame(spec: dict) -> pd.DataFrame:
    row = {f: spec.get(f, np.nan) for f in FEATURES}
    df = pd.DataFrame([row])
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype(object)
    return df[FEATURES]


def predict(X: pd.DataFrame):
    b = load_bundle()
    pred = np.exp(b["model"].predict(X))
    lo = np.exp(b["q10"].predict(X))
    hi = np.exp(b["q90"].predict(X))
    lo = np.minimum(lo, pred * 0.97)
    hi = np.maximum(hi, pred * 1.03)
    return pred, lo, hi


def describe_value(feature: str, v, spec: dict) -> str:
    """Human text for a feature value, used in the 'why this price' list."""
    isnan = v is None or (isinstance(v, float) and math.isnan(v))
    if feature == "chip_tier":
        name = spec.get("chipset") or "Unknown chipset"
        return f"{name} · tier {v:.1f}/10" if not isnan else f"{name} (tier unknown)"
    if isnan:
        return f"{FEATURE_LABELS.get(feature, feature)} unknown"
    if feature == "ram_gb":
        return f"{v:g} GB RAM"
    if feature == "storage_gb":
        return f"{int(v) if v >= 1 else v:g} GB storage" if v < 1024 else "1 TB storage"
    if feature == "battery_mah":
        return f"{v:,.0f} mAh battery"
    if feature == "fast_charge_w":
        return f"{v:g} W fast charging" if v > 0 else "No fast charging"
    if feature == "screen_in":
        return f'{v:.2f}" screen'
    if feature == "pixels_mp":
        if spec.get("res_w") and spec.get("res_h") and not math.isnan(float(spec["res_w"])):
            return f"{int(spec['res_w'])} × {int(spec['res_h'])} display"
        return f"{v:.1f} MP display"
    if feature == "refresh_hz":
        return f"{v:.0f} Hz refresh rate"
    if feature == "main_cam_mp":
        return f"{v:g} MP main camera"
    if feature == "n_rear_cams":
        return f"{v:.0f} rear camera{'s' if v != 1 else ''}"
    if feature == "front_cam_mp":
        return f"{v:g} MP front camera"
    if feature == "has_5g":
        return "5G" if v else "No 5G"
    if feature == "has_nfc":
        return "NFC" if v else "No NFC"
    if feature == "has_ir":
        return "IR blaster" if v else "No IR blaster"
    if feature == "expandable":
        return "Expandable storage" if v else "No memory card slot"
    if feature == "os_version":
        return f"OS version {v:g}"
    if feature == "clock_ghz":
        return f"{v:g} GHz CPU clock"
    if feature == "num_cores":
        return f"{v:.0f}-core CPU"
    if feature == "brand":
        return f"{str(v).title()} brand"
    if feature == "processor_brand":
        return f"{str(v).title()} processor family"
    if feature == "os":
        return {"ios": "iOS", "android": "Android"}.get(v, "Other OS")
    return f"{FEATURE_LABELS.get(feature, feature)}: {v}"


def explain(X: pd.DataFrame, spec: dict, top: int = 5) -> list[dict]:
    """Top reasons as rupee deltas. The model works in log(price); a feature's
    SHAP value s moves the price by a factor exp(s). We convert that to rupees
    relative to the final prediction: delta = price * (1 - exp(-s))."""
    b = load_bundle()
    Xt = b["model"]["pre"].transform(X)
    sv = load_explainer().shap_values(Xt)[0]
    price = float(np.exp(b["model"].predict(X))[0])
    rows = []
    for f, s in zip(FEATURES, sv):
        delta = price * (1 - math.exp(-s))
        rows.append({"feature": f, "delta": delta, "text": describe_value(f, X.iloc[0][f], spec)})
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return rows[:top]


def spec_from_row(row: pd.Series) -> dict:
    spec = {f: row[f] for f in FEATURES}
    spec["chipset"] = row.get("chipset")
    spec["res_w"], spec["res_h"] = row.get("res_w"), row.get("res_h")
    spec["model"] = row["model"]
    return spec


def closest(options: list, value, default=None):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default if default is not None else options[0]
    return min(options, key=lambda o: abs(o - value))


# ==========================================================================
# UI pieces
# ==========================================================================


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def header(meta: dict):
    h = meta["headline"]
    html(f"""
    <div class="pw-header">
      <div>
        <div class="pw-brand">Phone<span>Worth</span></div>
        <div class="pw-tagline">{TAGLINE}</div>
      </div>
      <div class="pw-badge"><span class="dot"></span>±{h['mape_pct']:.0f}% avg error · R² {h['r2']:.2f} · {meta['n_phones']} phones</div>
    </div>""")


RES_OPTIONS = {
    "HD+ (720 × 1600)": (720, 1600),
    "FHD+ (1080 × 2400)": (1080, 2400),
    "FHD+ tall (1080 × 2412)": (1080, 2412),
    "1.5K (1220 × 2712)": (1220, 2712),
    "QHD+ (1440 × 3200)": (1440, 3200),
}
RAM_OPTIONS = [2, 3, 4, 6, 8, 12, 16]
STORAGE_OPTIONS = [32, 64, 128, 256, 512, 1024]
REFRESH_OPTIONS = [60, 90, 120, 144]
MAIN_CAM_OPTIONS = [8, 12, 13, 16, 48, 50, 64, 108, 200]
FRONT_CAM_OPTIONS = [5, 8, 12, 13, 16, 32, 50]
CORE_OPTIONS = [8, 6, 4]


def spec_form(prefix: str, meta: dict, defaults: dict | None = None) -> dict:
    """Spec input form in neat groups. Returns a spec dict for the model."""
    d = defaults or {}
    k = lambda name: f"{prefix}_{name}"  # noqa: E731

    brands = meta["choices"]["brands"]
    chips = meta["choices"]["chipsets"]
    chip_names = [c["chipset"] for c in chips]
    chip_tiers = {c["chipset"]: c["tier"] for c in chips}

    # --- Brand & chip -----------------------------------------------------
    with st.container(border=True):
        html('<div class="pw-group-title">Brand & chip</div>')
        c1, c2 = st.columns(2)
        brand = c1.selectbox("Brand", brands, index=brands.index(d["brand"]) if d.get("brand") in brands else brands.index("samsung"),
                             format_func=str.title, key=k("brand"))
        default_chip = d.get("chipset") if d.get("chipset") in chip_names else "Snapdragon 695"
        chipset = c2.selectbox("Chipset", chip_names, index=chip_names.index(default_chip) if default_chip in chip_names else 0,
                               format_func=lambda c: f"{c}  ·  tier {chip_tiers[c]:g}", key=k("chip"))
        c3, c4 = st.columns(2)
        cores = c3.selectbox("CPU cores", CORE_OPTIONS, index=CORE_OPTIONS.index(closest(CORE_OPTIONS, d.get("num_cores"), 8)), key=k("cores"))
        clock = c4.slider("CPU clock (GHz)", 1.2, 3.5, float(closest([round(x * 0.1, 1) for x in range(12, 36)], d.get("clock_ghz"), 2.2)), 0.1, key=k("clock"))

    # --- Memory -----------------------------------------------------------
    with st.container(border=True):
        html('<div class="pw-group-title">Memory</div>')
        c1, c2, c3 = st.columns([1, 1, 1])
        ram = c1.selectbox("RAM", RAM_OPTIONS, index=RAM_OPTIONS.index(closest(RAM_OPTIONS, d.get("ram_gb"), 6)), format_func=lambda v: f"{v} GB", key=k("ram"))
        storage = c2.selectbox("Storage", STORAGE_OPTIONS, index=STORAGE_OPTIONS.index(closest(STORAGE_OPTIONS, d.get("storage_gb"), 128)),
                               format_func=lambda v: "1 TB" if v == 1024 else f"{v} GB", key=k("storage"))
        expandable = c3.toggle("Card slot", value=bool(d.get("expandable", 1.0)) if not _nan(d.get("expandable")) else True, key=k("sd"))

    # --- Display ----------------------------------------------------------
    with st.container(border=True):
        html('<div class="pw-group-title">Display</div>')
        c1, c2 = st.columns(2)
        screen = c1.slider("Screen size (inches)", 4.0, 7.6, float(round(d.get("screen_in", 6.6), 2)) if not _nan(d.get("screen_in")) else 6.6, 0.05, key=k("screen"))
        refresh = c2.selectbox("Refresh rate", REFRESH_OPTIONS, index=REFRESH_OPTIONS.index(closest(REFRESH_OPTIONS, d.get("refresh_hz"), 120)),
                               format_func=lambda v: f"{v} Hz", key=k("hz"))
        res_names = list(RES_OPTIONS)
        default_res = res_names[1]
        if not _nan(d.get("res_w")) and not _nan(d.get("res_h")):
            default_res = min(res_names, key=lambda n: abs(RES_OPTIONS[n][0] * RES_OPTIONS[n][1] - d["res_w"] * d["res_h"]))
        res_name = st.selectbox("Resolution", res_names, index=res_names.index(default_res), key=k("res"))

    # --- Camera -----------------------------------------------------------
    with st.container(border=True):
        html('<div class="pw-group-title">Camera</div>')
        c1, c2, c3 = st.columns(3)
        main_cam = c1.selectbox("Main camera", MAIN_CAM_OPTIONS, index=MAIN_CAM_OPTIONS.index(closest(MAIN_CAM_OPTIONS, d.get("main_cam_mp"), 50)),
                                format_func=lambda v: f"{v} MP", key=k("cam"))
        n_rear = c2.selectbox("Rear cameras", [1, 2, 3, 4], index=[1, 2, 3, 4].index(int(closest([1, 2, 3, 4], d.get("n_rear_cams"), 3))), key=k("nrear"))
        front_cam = c3.selectbox("Front camera", FRONT_CAM_OPTIONS, index=FRONT_CAM_OPTIONS.index(closest(FRONT_CAM_OPTIONS, d.get("front_cam_mp"), 16)),
                                 format_func=lambda v: f"{v} MP", key=k("front"))

    # --- Battery ----------------------------------------------------------
    with st.container(border=True):
        html('<div class="pw-group-title">Battery</div>')
        c1, c2 = st.columns(2)
        battery = c1.slider("Capacity (mAh)", 2000, 7000, int(closest(list(range(2000, 7001, 50)), d.get("battery_mah"), 5000)), 50, key=k("mah"))
        fast = c2.slider("Fast charging (W, 0 = none)", 0, 240, int(closest(list(range(0, 241, 5)), d.get("fast_charge_w"), 33)), 5, key=k("watt"))

    # --- Connectivity -----------------------------------------------------
    with st.container(border=True):
        html('<div class="pw-group-title">Connectivity</div>')
        c1, c2, c3 = st.columns(3)
        has_5g = c1.toggle("5G", value=bool(d.get("has_5g", 1.0)) if not _nan(d.get("has_5g")) else True, key=k("5g"))
        has_nfc = c2.toggle("NFC", value=bool(d.get("has_nfc", 0.0)) if not _nan(d.get("has_nfc")) else False, key=k("nfc"))
        has_ir = c3.toggle("IR blaster", value=bool(d.get("has_ir", 0.0)) if not _nan(d.get("has_ir")) else False, key=k("ir"))

    rw, rh = RES_OPTIONS[res_name]
    os_name = "ios" if brand == "apple" else "android"
    return {
        "brand": brand, "chipset": chipset, "processor_brand": processor_brand(chipset), "chip_tier": chip_tier(chipset),
        "num_cores": float(cores), "clock_ghz": float(clock),
        "ram_gb": float(ram), "storage_gb": float(storage), "expandable": float(expandable),
        "screen_in": float(screen), "refresh_hz": float(refresh), "res_w": rw, "res_h": rh, "pixels_mp": rw * rh / 1e6,
        "main_cam_mp": float(main_cam), "n_rear_cams": float(n_rear), "front_cam_mp": float(front_cam),
        "battery_mah": float(battery), "fast_charge_w": float(fast),
        "has_5g": float(has_5g), "has_nfc": float(has_nfc), "has_ir": float(has_ir),
        "os": os_name, "os_version": d.get("os_version", np.nan) if d.get("brand") == brand else np.nan,
    }


def price_card(pred: float, lo: float, hi: float, title: str = "Estimated fair price", sub: str | None = None):
    w_lo, w_hi = lo * 0.88, hi * 1.12
    pct = lambda v: max(0, min(100, (v - w_lo) / (w_hi - w_lo) * 100))  # noqa: E731
    html(f"""
    <div class="pw-card">
      <div class="pw-kicker">{title}</div>
      <div class="pw-price">{inr(pred)}</div>
      <div class="pw-sub">{sub or 'Typical Indian retail price for these specs'}</div>
      <div class="pw-range">
        <div class="pw-range-track">
          <div class="pw-range-fill" style="left:{pct(lo):.1f}%;width:{pct(hi) - pct(lo):.1f}%"></div>
          <div class="pw-range-marker" style="left:{pct(pred):.1f}%"></div>
        </div>
        <div class="pw-range-labels"><span>Low <b>{inr(lo)}</b></span><span>Likely range (10–90%)</span><span>High <b>{inr(hi)}</b></span></div>
      </div>
    </div>""")


def reasons_card(reasons: list[dict]):
    rows = ""
    for r in reasons:
        up = r["delta"] >= 0
        rows += f"""<div class="pw-reason"><div><div class="lbl">{r['text']}</div><div class="why">{FEATURE_LABELS.get(r['feature'], r['feature'])}</div></div>
        <span class="pw-chip {'up' if up else 'down'}">{'+' if up else '−'}{inr(abs(r['delta']))}</span></div>"""
    html(f'<div class="pw-card"><div class="pw-h">Why this price</div>{rows}<div class="pw-note" style="margin-top:10px">Top 5 spec effects versus an average phone, from SHAP values.</div></div>')


def phone_card(row: pd.Series, pred: float | None = None, highlight: bool = False):
    est = f'<div class="est">Fair price {inr(pred)}</div>' if pred is not None else ""
    specs = [
        ("Chipset", row.get("chipset") or "–"),
        ("Memory", f"{row['ram_gb']:g} / {int(row['storage_gb'])} GB" if not _nan(row.get("ram_gb")) and not _nan(row.get("storage_gb")) else "–"),
        ("Display", f'{row["screen_in"]:.2f}" · {row["refresh_hz"]:.0f} Hz' if not _nan(row.get("screen_in")) else "–"),
        ("Camera", f"{row['main_cam_mp']:g} MP · {row['n_rear_cams']:.0f} rear" if not _nan(row.get("main_cam_mp")) else "–"),
        ("Battery", f"{row['battery_mah']:,.0f} mAh" if not _nan(row.get("battery_mah")) else "–"),
    ]
    spec_html = "".join(f'<div class="spec"><span>{a}</span><b>{b}</b></div>' for a, b in specs)
    border = f'style="border-color:{P["accent"]}"' if highlight else ""
    html(f'<div class="pw-phone" {border}><div class="name">{row["model"]}</div><div class="price">{inr(row["price"])}</div>{est}{spec_html}</div>')


def metric_tile(k: str, v: str, s: str = ""):
    html(f'<div class="pw-metric"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>')


def plot_layout(fig: go.Figure, height: int = 420, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13, color=P["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=P["surface"], font_color=P["text"], bordercolor=P["border"]),
        **kw,
    )
    fig.update_xaxes(gridcolor=P["border"], zeroline=False, linecolor=P["border"], tickfont=dict(color=P["muted"]), title_font=dict(color=P["muted"]))
    fig.update_yaxes(gridcolor=P["border"], zeroline=False, linecolor=P["border"], tickfont=dict(color=P["muted"]), title_font=dict(color=P["muted"]))
    return fig


# ==========================================================================
# Pages
# ==========================================================================


def page_predict(meta: dict, phones: pd.DataFrame):
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        preset_options = [None] + phones.index.tolist()
        preset = st.selectbox("Start from a phone in the dataset (optional)", preset_options,
                              format_func=lambda i: "Blank form" if i is None else phones.loc[i, "label"], key="p1_preset")
        defaults = spec_from_row(phones.loc[preset]) if preset is not None else None
        spec = spec_form("p1", meta, defaults)
    with right:
        X = feature_frame(spec)
        pred, lo, hi = predict(X)
        price_card(float(pred[0]), float(lo[0]), float(hi[0]))
        reasons_card(explain(X, spec))
        if preset is not None:
            actual = phones.loc[preset, "price"]
            diff = (actual - pred[0]) / pred[0] * 100
            html(f'<div class="pw-card"><div class="pw-kicker">Listed price in dataset</div><div class="pw-price" style="font-size:28px">{inr(actual)}</div>'
                 f'<div class="pw-sub">{"+" if diff >= 0 else "−"}{abs(diff):.0f}% versus the estimate</div></div>')


def verdict(shop: float, pred: float) -> tuple[str, str, float]:
    diff = (shop - pred) / pred * 100
    if diff <= -8:
        return "Good deal", "good", diff
    if diff <= 8:
        return "Fair price", "fair", diff
    return "Overpriced", "bad", diff


def gauge(diff: float, cls: str) -> go.Figure:
    color = {"good": P["good"], "fair": P["warn"], "bad": P["bad"]}[cls]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max(-40, min(40, diff)),
        number=dict(suffix="%", font=dict(size=34, color=P["text"]), valueformat="+.0f"),
        gauge=dict(
            axis=dict(range=[-40, 40], tickvals=[-40, -8, 0, 8, 40], ticksuffix="%", tickcolor=P["muted"], tickfont=dict(color=P["muted"])),
            bar=dict(color=color, thickness=0.28),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[
                dict(range=[-40, -8], color=P["good_soft"]),
                dict(range=[-8, 8], color=P["warn_soft"]),
                dict(range=[8, 40], color=P["bad_soft"]),
            ],
            threshold=dict(line=dict(color=P["text"], width=2), thickness=0.8, value=0),
        ),
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=P["text"]))
    return fig


def page_deal(meta: dict, phones: pd.DataFrame):
    mode = st.radio("How do you want to describe the phone?", ["Pick from the list", "Enter specs"], horizontal=True, key="p2_mode", label_visibility="collapsed")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        if mode == "Pick from the list":
            idx = st.selectbox("Phone", phones.index.tolist(), format_func=lambda i: phones.loc[i, "label"], key="p2_phone")
            row = phones.loc[idx]
            spec = spec_from_row(row)
            default_price = int(row["price"])
            phone_card(row)
            st.write("")
        else:
            spec = spec_form("p2", meta)
            default_price = 20000
        shop = st.number_input("Shop price (₹)", min_value=1000, max_value=400000, value=default_price, step=500, key="p2_price")
    with right:
        X = feature_frame(spec)
        pred, lo, hi = predict(X)
        pred, lo, hi = float(pred[0]), float(lo[0]), float(hi[0])
        label, cls, diff = verdict(shop, pred)
        sign = "+" if diff >= 0 else "−"
        html(f"""<div class="pw-verdict {cls}"><div class="v">{label}</div>
        <div class="d">Shop price <b>{inr(shop)}</b> is <b>{sign}{abs(diff):.0f}%</b> versus the fair price of <b>{inr(pred)}</b>.</div></div>""")
        st.plotly_chart(gauge(diff, cls), width="stretch", config={"displayModeBar": False})
        price_card(pred, lo, hi, title="PhoneWorth fair price", sub=f"You save {inr(pred - shop)}" if shop < pred else f"You pay {inr(shop - pred)} over the estimate")


def page_compare(meta: dict, phones: pd.DataFrame):
    defaults = [i for i in phones.index if phones.loc[i, "model"] in ("OnePlus 11 5G", "Samsung Galaxy S23 Ultra 5G")][:2]
    picks = st.multiselect("Pick 2 or 3 phones", phones.index.tolist(), default=defaults, max_selections=3,
                           format_func=lambda i: phones.loc[i, "label"], key="p3_picks")
    if len(picks) < 2:
        st.info("Choose at least two phones to compare.")
        return
    rows = phones.loc[picks]
    pred, lo, hi = predict(rows[FEATURES])

    cols = st.columns(len(picks))
    for c, (i, r), p in zip(cols, rows.iterrows(), pred):
        with c:
            phone_card(r, float(p))

    st.write("")
    fig = go.Figure()
    names = rows["model"].tolist()
    fig.add_bar(name="Listed price", x=names, y=rows["price"], marker=dict(color=P["accent"], cornerradius=4),
                text=[inr(v) for v in rows["price"]], textposition="outside", hovertemplate="%{x}<br>Listed: %{text}<extra></extra>")
    fig.add_bar(name="PhoneWorth estimate", x=names, y=pred, marker=dict(color=P["series2"], cornerradius=4),
                text=[inr(v) for v in pred], textposition="outside", hovertemplate="%{x}<br>Estimate: %{text}<extra></extra>")
    fig.update_traces(textfont=dict(color=P["muted"], size=12))
    plot_layout(fig, height=380, barmode="group", bargap=0.35, bargroupgap=0.08, yaxis_title="Price (₹)")
    fig.update_yaxes(tickprefix="₹", separatethousands=True, range=[0, max(max(rows["price"]), max(pred)) * 1.2])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    spec_rows = [
        ("Listed price", [inr(v) for v in rows["price"]]),
        ("Fair price (estimate)", [inr(v) for v in pred]),
        ("Likely range", [f"{inr(a)} – {inr(b)}" for a, b in zip(lo, hi)]),
        ("Chipset", [f"{r.get('chipset') or '–'} (tier {r['chip_tier']:g})" if not _nan(r.get("chip_tier")) else (r.get("chipset") or "–") for _, r in rows.iterrows()]),
        ("RAM / storage", [f"{r['ram_gb']:g} GB / {int(r['storage_gb'])} GB" if not _nan(r.get("ram_gb")) else "–" for _, r in rows.iterrows()]),
        ("Display", [f'{r["screen_in"]:.2f}" · {int(r["res_w"]) if not _nan(r.get("res_w")) else "?"}×{int(r["res_h"]) if not _nan(r.get("res_h")) else "?"} · {r["refresh_hz"]:.0f} Hz' for _, r in rows.iterrows()]),
        ("Rear camera", [f"{r['main_cam_mp']:g} MP · {r['n_rear_cams']:.0f} lenses" if not _nan(r.get("main_cam_mp")) else "–" for _, r in rows.iterrows()]),
        ("Front camera", [f"{r['front_cam_mp']:g} MP" if not _nan(r.get("front_cam_mp")) else "–" for _, r in rows.iterrows()]),
        ("Battery", [f"{r['battery_mah']:,.0f} mAh · {r['fast_charge_w']:g} W" if not _nan(r.get("battery_mah")) and not _nan(r.get("fast_charge_w")) else (f"{r['battery_mah']:,.0f} mAh" if not _nan(r.get("battery_mah")) else "–") for _, r in rows.iterrows()]),
        ("5G / NFC / IR", ["  ·  ".join(["5G" if r["has_5g"] else "no 5G", "NFC" if r["has_nfc"] else "no NFC", "IR" if r["has_ir"] else "no IR"]) for _, r in rows.iterrows()]),
        ("Memory card", ["Yes" if r["expandable"] else "No" for _, r in rows.iterrows()]),
    ]
    head = "".join(f"<th>{n}</th>" for n in names)
    body = "".join(f"<tr><td class='pw-kicker'>{k}</td>{''.join(f'<td>{v}</td>' for v in vals)}</tr>" for k, vals in spec_rows)
    html(f'<div class="pw-card"><table class="pw-table"><thead><tr><th>Spec</th>{head}</tr></thead><tbody>{body}</tbody></table></div>')


SIMILARITY_FEATURES = ["chip_tier", "ram_gb", "storage_gb", "battery_mah", "fast_charge_w", "screen_in", "pixels_mp",
                       "refresh_hz", "main_cam_mp", "n_rear_cams", "front_cam_mp", "has_5g", "has_nfc"]
SIMILARITY_WEIGHTS = {"chip_tier": 2.0, "ram_gb": 1.3, "storage_gb": 1.3}


def similar_phones(phones: pd.DataFrame, idx, n: int = 5) -> pd.DataFrame:
    M = phones[SIMILARITY_FEATURES].astype(float)
    M = M.fillna(M.median())
    Z = (M - M.mean()) / M.std().replace(0, 1)
    w = np.array([SIMILARITY_WEIGHTS.get(f, 1.0) for f in SIMILARITY_FEATURES])
    d = np.sqrt((((Z - Z.loc[idx]) ** 2) * w).sum(axis=1))
    d = d[phones["base_model"] != phones.loc[idx, "base_model"]]
    return phones.loc[d.sort_values().index[:n]]


def page_similar(meta: dict, phones: pd.DataFrame):
    idx = st.selectbox("Phone", phones.index.tolist(), format_func=lambda i: phones.loc[i, "label"], key="p4_phone")
    preds = phone_predictions()
    base = preds.loc[idx]
    c1, c2 = st.columns([1, 3])
    with c1:
        phone_card(base, float(base["pred"]), highlight=True)
    with c2:
        html('<div class="pw-card"><div class="pw-h">Closest phones by specs</div><div class="pw-note">Distance over chipset tier, memory, display, camera, battery and connectivity '
             '(chipset and memory weighted higher). Variants of the same phone are excluded.</div></div>')
        sims = similar_phones(preds, idx)
        cols = st.columns(len(sims))
        for c, (_, r) in zip(cols, sims.iterrows()):
            with c:
                phone_card(r, float(r["pred"]))


def page_report(meta: dict, phones: pd.DataFrame):
    m = meta["metrics"]
    best = m["LightGBM"]
    base = m["Brand median"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_tile("Avg error (MAE)", inr(best["MAE_inr"]), f"baseline {inr(base['MAE_inr'])}")
    with c2:
        metric_tile("Avg % error (MAPE)", f"{best['MAPE_pct']:.1f}%", f"baseline {base['MAPE_pct']:.1f}%")
    with c3:
        metric_tile("R²", f"{best['R2']:.3f}", f"baseline {base['R2']:.3f}")
    with c4:
        metric_tile("Within 20%", f"{best['within_20pct']:.0f}%", "of phones priced within ±20%")

    st.write("")
    order = ["Brand median", "Ridge", "LightGBM"]
    body = ""
    for name in order:
        r = m[name]
        cls = "best" if name == "LightGBM" else ""
        body += (f"<tr class='{cls}'><td>{name}{' (final model)' if cls else ''}</td><td class='num'>{inr(r['MAE_inr'])}</td>"
                 f"<td class='num'>{r['MAPE_pct']:.1f}%</td><td class='num'>{r['R2']:.3f}</td><td class='num'>{r['R2_log']:.3f}</td>"
                 f"<td class='num'>{r['within_10pct']:.0f}%</td><td class='num'>{r['within_20pct']:.0f}%</td></tr>")
    html(f"""<div class="pw-card"><div class="pw-h">Cross-validated accuracy</div>
    <table class="pw-table"><thead><tr><th>Model</th><th class="num">MAE</th><th class="num">MAPE</th><th class="num">R²</th><th class="num">R² (log)</th><th class="num">Within 10%</th><th class="num">Within 20%</th></tr></thead>
    <tbody>{body}</tbody></table>
    <div class="pw-note" style="margin-top:10px">{meta['cv']['folds']}-fold GroupKFold grouped by phone model, so storage/RAM variants of one phone never appear in both train and test.
    Trained on {meta['n_phones']} phones ({meta['n_distinct_models']} distinct models), prices ₹3,000 – ₹2,50,000.</div></div>""")

    left, right = st.columns(2, gap="large")
    oof = load_oof()
    with left:
        html('<div class="pw-h">Predicted vs actual price (held-out)</div>')
        fig = go.Figure()
        lim = [oof["price"].min() * 0.8, oof["price"].max() * 1.2]
        fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color=P["gray"], width=1.5, dash="dash"), name="Perfect prediction", hoverinfo="skip")
        fig.add_scatter(x=oof["price"], y=oof["pred_lightgbm"], mode="markers", name="Phone",
                        marker=dict(color=P["accent"], size=7, opacity=0.55, line=dict(width=1, color=P["surface"])),
                        customdata=oof["model"], hovertemplate="%{customdata}<br>Actual ₹%{x:,.0f}<br>Predicted ₹%{y:,.0f}<extra></extra>")
        plot_layout(fig, height=420, xaxis_title="Actual price (₹)", yaxis_title="Predicted price (₹)", showlegend=True)
        ticks = [3000, 5000, 10000, 20000, 50000, 100000, 200000]
        labels = [inr(t) for t in ticks]
        fig.update_xaxes(type="log", tickvals=ticks, ticktext=labels)
        fig.update_yaxes(type="log", tickvals=ticks, ticktext=labels)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        html('<div class="pw-h">What drives the price</div>')
        imp = pd.DataFrame(meta["shap_importance"]).head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=imp["label"], orientation="h",
                               marker=dict(color=P["accent"], cornerradius=4),
                               hovertemplate="%{y}<br>mean |SHAP| = %{x:.3f}<extra></extra>"))
        plot_layout(fig, height=420, xaxis_title="Mean |SHAP| (effect on log price)", bargap=0.35)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    html(f"""<div class="pw-card"><div class="pw-h">About the data</div>
    <div class="pw-note">Source: Kaggle “Smartphone Specifications and Prices in India” (scraped from Smartprix, CC0). 1,020 raw rows.
    Removed: {meta['removed_rows'].get('too_cheap', '–')} phones under ₹3,000, {meta['removed_rows'].get('too_pricey', '–')} over ₹2,50,000.
    The target is log(price) so errors are proportional; predictions are converted back to rupees. Model trained {meta['trained_at'][:10]}.</div></div>""")


# ==========================================================================
# Main
# ==========================================================================


def main():
    meta = load_meta()
    phones = load_phones()
    header(meta)
    tabs = st.tabs(["Predict price", "Deal checker", "Compare", "Similar phones", "Model report"])
    with tabs[0]:
        page_predict(meta, phones)
    with tabs[1]:
        page_deal(meta, phones)
    with tabs[2]:
        page_compare(meta, phones)
    with tabs[3]:
        page_similar(meta, phones)
    with tabs[4]:
        page_report(meta, phones)


if __name__ == "__main__":
    main()

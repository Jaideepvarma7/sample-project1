"""Clean the raw Smartprix text columns into numeric features.

The raw CSV stores every spec as free text, and the columns are not reliable:
when a phone has no memory-card slot, the OS text slides into the `card`
column; iPhones have no battery entry so `display` holds the camera text, and
so on. To be robust we join all spec columns of a row into one text "blob"
and pull each feature out of it with a pattern that only matches that feature
(e.g. "5000 mAh" can only be a battery). This means a value is found wherever
it sits.

Outputs
-------
build_features(raw_df) -> clean DataFrame with one row per phone variant,
all feature columns below, `price` (INR) and `log_price` (the model target).

Running this file directly writes data/processed/phones_clean.csv.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "phones_clean.csv"

# Price limits (INR). Rows outside are treated as outliers / not smartphones.
PRICE_MIN = 3_000
PRICE_MAX = 250_000

SPEC_COLS = ["sim", "processor", "ram", "battery", "display", "camera", "card", "os"]

# Feature columns used by the models. Order matters (saved with the model).
NUMERIC_FEATURES = [
    "chip_tier", "num_cores", "clock_ghz",
    "ram_gb", "storage_gb",
    "battery_mah", "fast_charge_w",
    "screen_in", "pixels_mp", "refresh_hz",
    "main_cam_mp", "n_rear_cams", "front_cam_mp",
    "has_5g", "has_nfc", "has_ir", "expandable",
    "os_version",
]
CATEGORICAL_FEATURES = ["brand", "processor_brand", "os"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Human-friendly names for charts and the "why this price" list.
FEATURE_LABELS = {
    "chip_tier": "Chipset tier",
    "num_cores": "CPU cores",
    "clock_ghz": "CPU clock (GHz)",
    "ram_gb": "RAM",
    "storage_gb": "Storage",
    "battery_mah": "Battery",
    "fast_charge_w": "Fast charging",
    "screen_in": "Screen size",
    "pixels_mp": "Screen resolution",
    "refresh_hz": "Refresh rate",
    "main_cam_mp": "Main camera",
    "n_rear_cams": "Rear cameras",
    "front_cam_mp": "Front camera",
    "has_5g": "5G",
    "has_nfc": "NFC",
    "has_ir": "IR blaster",
    "expandable": "Expandable storage",
    "os_version": "OS version",
    "brand": "Brand",
    "processor_brand": "Processor brand",
    "os": "Operating system",
}


# --------------------------------------------------------------------------
# Chipset handling
# --------------------------------------------------------------------------

def normalise_chipset(name: str | None) -> str | None:
    """Tidy a chipset string: fix typos, drop vendor prefixes and '5G'."""
    if not name or not isinstance(name, str):
        return None
    s = re.sub(r"\s+", " ", name).strip()
    s = s.replace("Sanpdragon", "Snapdragon")
    s = re.sub(r"^Qualcomm\s+(\d{3})$", r"Snapdragon \1", s, flags=re.I)   # "Qualcomm 215"
    s = re.sub(r"^(Qualcomm|Samsung|MediaTek|Apple)\s+(?=\w)", "", s, flags=re.I)
    s = re.sub(r"\s+5G$", "", s, flags=re.I)          # "Dimensity 700 5G" -> "Dimensity 700" (but keep "765G")
    s = re.sub(r"-", " ", s)
    s = re.sub(r"(\d)\s*\+", r"\1+", s)          # "8 +" -> "8+"
    s = re.sub(r"Gen\s*(\d)", r"Gen \1", s, flags=re.I)
    s = re.sub(r"Gen(\d)", r"Gen \1", s, flags=re.I)
    # "A13 Bionic" / "Apple A13" -> "Bionic A13"
    m = re.match(r"^A(\d{1,2})(?:\s*Bionic)?$", s, flags=re.I)
    if m:
        s = f"Bionic A{m.group(1)}"
    s = re.sub(r"^Fusion APL1024$", "Fusion A10", s, flags=re.I)
    s = s.title().replace("Gen ", "Gen ").replace("Sc", "SC").replace("Msm", "MSM").replace("Qm", "QM")
    s = s.replace("Snapdragon", "Snapdragon").replace("Dimensity", "Dimensity")
    return s


CHIP_FAMILY_PATTERNS = [
    ("snapdragon", r"snapdragon|qualcomm|qm215|msm89"),
    ("dimensity", r"dimensity"),
    ("helio", r"helio|mt6\d{3}"),
    ("exynos", r"exynos"),
    ("apple", r"bionic|fusion|apple a\d"),
    ("tensor", r"tensor"),
    ("kirin", r"kirin"),
    ("unisoc", r"unisoc|tiger|spreadtrum|\bsc\d{4}|\bt\d{3}\b"),
]


def processor_brand(chipset: str | None) -> str:
    if not chipset:
        return "unknown"
    s = chipset.lower()
    for fam, pat in CHIP_FAMILY_PATTERNS:
        if re.search(pat, s):
            return fam
    return "other"


def _num(s: str) -> int | None:
    m = re.search(r"(\d{3,4})", s)
    return int(m.group(1)) if m else None


def chip_tier(chipset: str | None) -> float:
    """Rough performance tier 1 (feature phone) .. 10 (current flagship).

    Rule based so that unseen chips of the same family still get a sensible
    score. Returns NaN when the chip is unknown.
    """
    if not chipset:
        return np.nan
    s = chipset.lower().replace("-", " ")
    plus = "+" in s or "plus" in s

    fam = processor_brand(chipset)

    if fam == "snapdragon":
        m = re.search(r"(\d)\+?\s*gen\s*(\d)", s)
        if m:
            series, gen = int(m.group(1)), int(m.group(2))
            # 8 Gen 1 = 8, 8+ Gen 1 = 8.5, 8 Gen 2 = 9, 8+ Gen 2 = 9.5, 8 Gen 3 = 10
            base, step = {8: (8.0, 1.0), 7: (6.5, 0.5), 6: (5.5, 0.5), 4: (4.0, 0.5)}.get(series, (4.0, 0.5))
            return min(10.0, base + step * (gen - 1) + (0.5 if plus else 0))
        n = _num(s)
        if n is None:
            return 2.0
        table = {
            888: 8.0, 870: 7.5, 865: 7.5, 860: 7.0, 855: 7.0, 845: 6.5, 835: 6.0,
            821: 5.5, 820: 5.5, 810: 5.0, 801: 4.5, 800: 4.5,
            782: 6.5, 780: 6.5, 778: 6.0, 775: 6.0, 768: 5.5, 765: 5.5, 750: 5.5,
            732: 5.0, 730: 5.0, 720: 5.0, 712: 4.5, 710: 4.5,
            695: 5.0, 690: 4.5, 685: 4.5, 680: 4.5, 678: 4.0, 675: 4.0, 670: 4.0,
            665: 3.5, 662: 3.5, 660: 3.5, 636: 3.0, 632: 3.0, 630: 3.0, 625: 3.0, 615: 2.5,
            480: 4.0, 460: 3.0, 450: 3.0, 439: 3.0, 435: 2.5, 430: 2.5, 429: 2.5, 425: 2.5,
        }
        if n in table:
            return table[n] + (0.25 if plus else 0)
        if n >= 800:
            return 6.0
        if n >= 700:
            return 5.0
        if n >= 600:
            return 3.5
        if n >= 400:
            return 2.5
        return 2.0

    if fam == "dimensity":
        n = _num(s)
        if n is None:
            return 5.0
        table = {
            9300: 10.0, 9200: 9.5, 9000: 9.0, 8300: 8.5, 8200: 8.0, 8100: 8.0, 8050: 7.5,
            8020: 7.5, 8000: 7.5, 7200: 7.0, 7050: 6.5, 7020: 6.0, 6100: 5.0, 6080: 5.0,
            6020: 5.0, 1300: 7.0, 1200: 7.0, 1100: 6.5, 1080: 6.5, 1000: 6.0, 930: 6.0,
            920: 6.0, 900: 5.5, 810: 5.0, 800: 5.0, 720: 5.0, 700: 4.5,
        }
        extra = 0.25 if (plus or "max" in s or "ultra" in s) else 0
        if n in table:
            return min(10.0, table[n] + extra)
        return 4.0 if n < 700 else 5.5

    if fam == "helio":
        m = re.search(r"([gpax])\s*(\d{2,3})", s)
        if m:
            letter, n = m.group(1), int(m.group(2))
            if letter == "g":
                table = {99: 5.0, 96: 4.5, 95: 4.5, 90: 4.5, 88: 4.0, 85: 4.0, 80: 4.0,
                         70: 3.5, 37: 3.0, 36: 3.0, 35: 3.0, 25: 2.5}
                return table.get(n, 3.0)
            if letter == "p":
                table = {95: 4.0, 90: 4.0, 70: 3.5, 65: 3.5, 60: 3.5, 35: 3.0, 23: 2.5, 22: 2.5}
                return table.get(n, 3.0)
            if letter == "a":
                return 2.0
            if letter == "x":
                return 3.5
        if re.search(r"mt6\d{3}", s):
            return 1.5
        return 3.0

    if fam == "apple":
        m = re.search(r"a\s*(\d{1,2})", s)
        n = int(m.group(1)) if m else 10
        table = {17: 10.0, 16: 9.5, 15: 9.0, 14: 8.5, 13: 8.0, 12: 7.5, 11: 6.5, 10: 5.5}
        return table.get(n, 4.5 if n < 10 else 10.0)

    if fam == "tensor":
        if "g3" in s:
            return 9.0
        if "g2" in s:
            return 8.5
        return 8.0

    if fam == "exynos":
        n = _num(s)
        if n is None:
            return 4.0
        table = {2400: 9.5, 2300: 9.0, 2200: 9.0, 2100: 8.5, 1380: 6.0, 1330: 5.0, 1280: 5.0,
                 1200: 5.0, 990: 7.5, 9825: 7.0, 9820: 7.0, 9810: 6.5, 980: 6.0, 9611: 3.5,
                 9610: 3.5, 880: 4.5, 850: 3.0, 7904: 3.0, 7885: 3.0, 7884: 3.0, 7870: 3.0}
        if n in table:
            return table[n]
        return 2.5 if n < 1000 or 7000 <= n < 9000 else 5.0

    if fam == "kirin":
        n = _num(s)
        table = {9000: 9.0, 990: 7.5, 980: 7.0, 970: 6.0, 810: 5.0, 710: 4.0, 659: 3.0}
        return table.get(n, 3.0)

    if fam == "unisoc":
        m = re.search(r"\bt\s*(\d{3,4})", s)
        if m:
            n = int(m.group(1))
            if n >= 820:
                return 5.0
            if n >= 700:
                return 4.0
            if n >= 600:
                return 3.0
            if n >= 300:
                return 2.5
            return 1.0
        if re.search(r"sc98|sc77|sc65|a55", s):
            return 1.5
        return 2.0

    return np.nan


# --------------------------------------------------------------------------
# Row parsing
# --------------------------------------------------------------------------

_CAM_WORDS = {"dual": 2, "triple": 3, "quad": 4, "penta": 5}


def _f(pattern: str, text: str, group: int = 1, flags=re.I) -> float:
    m = re.search(pattern, text, flags)
    return float(m.group(group)) if m else np.nan


def parse_specs(row: pd.Series) -> dict:
    """Parse one raw row into a dict of numeric/categorical features."""
    parts = [str(row[c]) for c in SPEC_COLS if c in row.index and pd.notna(row[c])]
    blob = " | ".join(parts)
    segments = [p.strip() for p in blob.split("|")]

    out: dict = {}

    # --- model / brand -----------------------------------------------------
    model = re.sub(r"\s+", " ", str(row["model"])).strip()
    out["model"] = model
    out["brand"] = model.split()[0].lower()
    # Base model name without the "(8GB RAM + 128GB)" variant suffix, used to
    # group variants of the same phone together in cross-validation.
    out["base_model"] = re.sub(r"\s*\(.*?\)\s*", " ", model).strip().lower()

    # --- memory ------------------------------------------------------------
    m = re.search(r"(\d+(?:\.\d+)?)\s*(GB|MB)\s*RAM", blob, re.I)
    out["ram_gb"] = (float(m.group(1)) / (1024 if m.group(2).upper() == "MB" else 1)) if m else np.nan
    m = re.search(r"(\d+(?:\.\d+)?)\s*(GB|MB|TB)\s*inbuilt", blob, re.I)
    if m:
        v, unit = float(m.group(1)), m.group(2).upper()
        out["storage_gb"] = v / 1024 if unit == "MB" else v * 1024 if unit == "TB" else v
    else:
        out["storage_gb"] = np.nan

    # --- battery -----------------------------------------------------------
    out["battery_mah"] = _f(r"(\d{3,5})\s*mAh", blob)
    has_fast = bool(re.search(r"fast charging", blob, re.I))
    w = _f(r"(\d+(?:\.\d+)?)\s*W\s*fast charging", blob)
    # Fast charging mentioned without wattage -> unknown (NaN); not mentioned -> 0.
    # No battery text at all (e.g. iPhones in this file) -> charging unknown too.
    out["fast_charge_w"] = (w if has_fast else 0.0) if not np.isnan(out["battery_mah"]) else np.nan

    # --- display -----------------------------------------------------------
    out["screen_in"] = _f(r"(\d+(?:\.\d+)?)\s*inch", blob)
    m = re.search(r"(\d{3,4})\s*x\s*(\d{3,4})\s*px", blob, re.I)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        out["res_w"], out["res_h"] = min(a, b), max(a, b)
        out["pixels_mp"] = a * b / 1e6
    else:
        out["res_w"] = out["res_h"] = out["pixels_mp"] = np.nan
    hz = _f(r"(\d{2,3})\s*Hz", blob)
    out["refresh_hz"] = hz if not np.isnan(hz) else (60.0 if not np.isnan(out["screen_in"]) else np.nan)

    # --- cameras -----------------------------------------------------------
    cam_seg = next((s for s in segments if re.search(r"\bRear\b", s, re.I)), None)
    if cam_seg:
        rear_txt, _, front_txt = cam_seg.partition("&")
        rear_mps = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*MP", rear_txt, re.I)]
        out["main_cam_mp"] = max(rear_mps) if rear_mps else np.nan
        word = re.search(r"\b(dual|triple|quad|penta)\b\s*rear", rear_txt, re.I)
        out["n_rear_cams"] = float(_CAM_WORDS[word.group(1).lower()]) if word else float(max(len(rear_mps), 1))
        front_mps = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*MP", front_txt, re.I)]
        out["front_cam_mp"] = max(front_mps) if front_mps else 0.0
    else:
        out["main_cam_mp"] = np.nan
        out["n_rear_cams"] = np.nan
        out["front_cam_mp"] = np.nan

    # --- connectivity ------------------------------------------------------
    sim_seg = next((s for s in segments if re.search(r"\bSim\b|VoLTE|Wi-Fi", s, re.I)), "")
    out["has_5g"] = float(bool(re.search(r"\b5G\b", sim_seg)))
    out["has_nfc"] = float(bool(re.search(r"\bNFC\b", blob)))
    out["has_ir"] = float(bool(re.search(r"IR Blaster", blob, re.I)))
    out["expandable"] = float(bool(re.search(r"Memory Card (Supported|\(Hybrid\))", blob, re.I)))

    # --- OS ----------------------------------------------------------------
    # (filled in after the processor is known, see below)

    # --- processor ---------------------------------------------------------
    chip = None
    proc_seg = next(
        (s for s in segments
         if re.search(r"snapdragon|dimensity|helio|exynos|bionic|apple\s*a\d|\ba\d{2}\b|tensor|kirin|unisoc|tiger|spreadtrum|sc\d{4}|\bt\d{3}\b|fusion|qualcomm|mt6\d{3}", s, re.I)),
        None,
    )
    if proc_seg:
        first = proc_seg.split(",")[0]
        if not re.search(r"\bCore\b|Processor$|RAM|mAh|inch|MP", first, re.I) or re.search(r"snapdragon|dimensity|helio|exynos|bionic|tensor|kirin|unisoc|tiger", first, re.I):
            chip = normalise_chipset(first)
    out["chipset"] = chip
    out["processor_brand"] = processor_brand(chip)
    out["chip_tier"] = chip_tier(chip)

    cores = re.search(r"\b(Octa|Hexa|Quad|Dual|Single)\s*Core", blob, re.I)
    out["num_cores"] = {"octa": 8, "hexa": 6, "quad": 4, "dual": 2, "single": 1}[cores.group(1).lower()] if cores else np.nan
    ghz = _f(r"(\d+(?:\.\d+)?)\s*GHz", blob)
    if np.isnan(ghz):
        mhz = _f(r"(\d+(?:\.\d+)?)\s*MHz", blob)
        ghz = mhz / 1000 if not np.isnan(mhz) else np.nan
    out["clock_ghz"] = ghz

    # --- OS ----------------------------------------------------------------
    # The raw file may not have an OS column; when the OS text is missing we
    # fall back on the brand (Apple -> iOS) and the chip family (a modern
    # smartphone chip -> Android). Everything else is "other" (feature phones).
    if re.search(r"\biOS\b", blob) or out["brand"] == "apple":
        out["os"] = "ios"
    elif re.search(r"\bAndroid\b", blob, re.I):
        out["os"] = "android"
    elif out["processor_brand"] in {"snapdragon", "dimensity", "helio", "exynos", "tensor", "kirin"}:
        out["os"] = "android"
    elif out["processor_brand"] == "unisoc" and (out["ram_gb"] if not np.isnan(out["ram_gb"]) else 0) >= 1:
        out["os"] = "android"
    else:
        out["os"] = "other"
    out["os_version"] = _f(r"(?:Android|iOS)\s*v?(\d+(?:\.\d+)?)", blob)

    out["rating"] = float(row["rating"]) if "rating" in row.index and pd.notna(row["rating"]) else np.nan
    return out


def parse_price(value) -> float:
    s = re.sub(r"[^\d.]", "", str(value))
    return float(s) if s else np.nan


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def build_features(raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Parse, clean and filter the raw dataset. Returns the clean table."""
    df = raw.copy()
    df["price"] = df["price"].map(parse_price)

    n_start = len(df)
    bad_price = df["price"].isna().sum()
    df = df[df["price"].notna()]

    parsed = pd.DataFrame([parse_specs(r) for _, r in df.iterrows()], index=df.index)
    parsed["price"] = df["price"].values

    too_cheap = (parsed["price"] < PRICE_MIN).sum()
    too_pricey = (parsed["price"] > PRICE_MAX).sum()
    parsed = parsed[(parsed["price"] >= PRICE_MIN) & (parsed["price"] <= PRICE_MAX)]

    # Drop exact duplicate rows (same model + same specs + same price).
    dups = parsed.duplicated(subset=["model", "ram_gb", "storage_gb", "price"]).sum()
    parsed = parsed.drop_duplicates(subset=["model", "ram_gb", "storage_gb", "price"])

    parsed["log_price"] = np.log(parsed["price"])
    parsed = parsed.reset_index(drop=True)

    if verbose:
        print(f"Raw rows: {n_start}")
        print(f"Removed {bad_price} rows with an unreadable price")
        print(f"Removed {too_cheap} outliers priced below Rs {PRICE_MIN:,}")
        print(f"Removed {too_pricey} outliers priced above Rs {PRICE_MAX:,}")
        print(f"Removed {dups} duplicate rows")
        print(f"Clean rows: {len(parsed)}  ({parsed['base_model'].nunique()} distinct phone models)")
        unknown = parsed["chip_tier"].isna().sum()
        print(f"Rows with unknown chipset tier: {unknown}")
        missing = parsed[NUMERIC_FEATURES].isna().sum()
        missing = missing[missing > 0]
        if len(missing):
            print("Missing values per feature (filled by the model):")
            print(missing.to_string())

    cols = ["model", "base_model", "price", "log_price", "chipset"] + FEATURES + ["res_w", "res_h", "rating"]
    out = parsed[cols]
    # Bookkeeping for the pipeline / model report (not saved in the CSV).
    out.attrs["removed"] = {
        "raw_rows": int(n_start),
        "bad_price": int(bad_price),
        "too_cheap": int(too_cheap),
        "too_pricey": int(too_pricey),
        "duplicates": int(dups),
        "clean_rows": int(len(out)),
    }
    return out


def save_clean(df: pd.DataFrame, path: Path = CLEAN_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python run_pipeline.py` first to build it."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    from src.data.load import load_raw

    clean = build_features(load_raw())
    out = save_clean(clean)
    print(f"\nSaved {out}")
    print(clean.head().to_string())

"""Module for tagging semiconductor industry events in news headlines.

Tags headlines into 5 domain event categories using keyword rules:
1. shortage: Supply deficit, lead time elongation, wafer crunch
2. tariff: Trade restrictions, export bans, custom duties
3. demand_surge: AI datacenter boom, HBM order spikes, hyperscaler purchases
4. supply_cut: Wafer reduction, production cuts, capex contraction
5. disaster: Fab fires, earthquakes, chemical contamination, power outages
"""

from __future__ import annotations

import logging
import re
from typing import Final, List
import pandas as pd

logger = logging.getLogger(__name__)

EVENT_KEYWORDS: Final[dict[str, List[str]]] = {
    "shortage": [
        "shortage",
        "scarcity",
        "deficit",
        "tight supply",
        "backlog",
        "lead time",
        "lead times",
        "crunch",
        "bottleneck",
        "component deficit",
        "out of stock",
    ],
    "tariff": [
        "tariff",
        "tariffs",
        "trade war",
        "duties",
        "duty",
        "customs",
        "export ban",
        "export bans",
        "export control",
        "export restrictions",
        "sanctions",
        "entity list",
        "section 301",
    ],
    "demand_surge": [
        "demand surge",
        "hbm demand",
        "ai server",
        "hyperscaler",
        "hyperscale",
        "order spike",
        "soaring demand",
        "record demand",
        "ai capex",
        "datacenter demand",
        "hbm3e",
        "hbm4",
        "surging demand",
    ],
    "supply_cut": [
        "production cut",
        "production cuts",
        "supply cut",
        "supply cuts",
        "capex cut",
        "wafer cut",
        "wafer reduction",
        "capacity cut",
        "halt production",
        "curtails output",
        "curtail",
        "fab idle",
        "reduce output",
        "reduce production",
    ],
    "disaster": [
        "fab fire",
        "fire",
        "earthquake",
        "power outage",
        "flood",
        "contamination",
        "explosion",
        "plant shutdown",
        "wafer damage",
        "cleanroom fire",
    ],
}


def tag_event_types(text: str) -> dict[str, int]:
    """Identify if a text matches any of the 5 event categories.

    Args:
        text: News headline or article summary.

    Returns:
        Dict mapping event names to 1 (present) or 0 (absent).
    """
    clean_text = " " + re.sub(r"[^\w\s]", " ", text.lower()) + " "
    flags: dict[str, int] = {}

    for event_name, keywords in EVENT_KEYWORDS.items():
        matched = False
        for kw in keywords:
            # Check pattern with word boundary flexibility
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, clean_text):
                matched = True
                break
        flags[f"event_{event_name}"] = 1 if matched else 0

    return flags


def get_event_tags_list(text: str) -> List[str]:
    """Return a human-readable list of active event tags for a headline."""
    flags = tag_event_types(text)
    tags = []
    for k, v in flags.items():
        if v == 1:
            tags.append(k.replace("event_", ""))
    return tags


def tag_events_dataframe(df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
    """Tag all headlines in a DataFrame with event indicators.

    Args:
        df: Input DataFrame.
        text_col: Headline column name.

    Returns:
        DataFrame with added columns:
        ['event_shortage', 'event_tariff', 'event_demand_surge', 'event_supply_cut', 'event_disaster']
    """
    texts = df[text_col].fillna("").astype(str).tolist()
    event_records = [tag_event_types(t) for t in texts]
    event_df = pd.DataFrame(event_records)

    for col in event_df.columns:
        df[col] = event_df[col].values

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    samples = [
        "Automakers warn of renewed semiconductor shortage as chip lead times stretch",
        "US announces 25% chip tariff and strict export ban on advanced memory",
        "Record AI datacenter capex fuels massive HBM demand surge at Nvidia suppliers",
        "Samsung and Micron implement aggressive DRAM production cuts to reduce inventory",
        "Fab fire at cleanroom packaging facility disrupts memory module shipments",
        "Micron reveals new DDR5 memory architecture roadmap",
    ]
    for s in samples:
        tags = get_event_tags_list(s)
        print(f"Headline: {s}\n -> Tags: {tags or ['general_news']}\n")

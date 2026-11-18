"""Loading and cleaning the FRED DEXMXUS series (Mexican pesos per U.S. dollar)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_levels(path: str | Path, expected_sha256: str | None = None) -> pd.DataFrame:
    """Read the raw FRED CSV and return non-missing daily quotations.

    FRED marks holidays with empty values; these rows are dropped, so each
    return below refers to two successive available quotations.
    """
    path = Path(path)
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError(f"Checksum mismatch for {path}: the raw data file was modified.")
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    date_col = "observation_date" if "observation_date" in raw else "date"
    value_col = "dexmxus" if "dexmxus" in raw else "value"
    df = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col]),
        "usdmxn": pd.to_numeric(raw[value_col], errors="coerce"),
    })
    df.attrs["n_raw_rows"] = len(df)
    df.attrs["n_missing"] = int(df["usdmxn"].isna().sum())
    df = df.dropna().sort_values("date").drop_duplicates("date")
    df = df[df["usdmxn"] > 0].reset_index(drop=True)
    return df


def load_returns(path: str | Path, start: str | None = None, end: str | None = None,
                 expected_sha256: str | None = None) -> pd.DataFrame:
    """Daily log changes in percent, r_t = 100 log(S_t / S_{t-1}).

    Positive values are peso depreciations. `start`/`end` select the dates of
    the returns (the first return uses the previous available quotation).
    """
    lv = load_levels(path, expected_sha256)
    out = lv.copy()
    out["ret"] = 100.0 * np.log(out["usdmxn"] / out["usdmxn"].shift(1))
    out["gap_days"] = out["date"].diff().dt.days
    out = out.dropna().reset_index(drop=True)
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    out = out.reset_index(drop=True)
    out.attrs.update(lv.attrs)
    return out

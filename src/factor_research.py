"""Reusable cross-sectional factor evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResearchConfig:
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    quantiles: int = 10
    min_samples: int = 20


def compute_forward_returns(
    frame: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Calculate close-to-close forward returns independently for each stock."""
    result = frame.sort_values(["code", "date"]).copy()
    result["adjusted_close"] = result["close"] * result["adjustment_factor"]
    by_code = result.groupby("code", sort=False)["adjusted_close"]
    for horizon in horizons:
        result[f"forward_return_{horizon}d"] = (
            by_code.shift(-horizon) / result["adjusted_close"] - 1
        )
    return result


def assign_quantiles(
    frame: pd.DataFrame,
    factor_col: str,
    quantiles: int,
) -> pd.DataFrame:
    """Assign deterministic, near-equal cross-sectional groups by trading day."""
    result = frame.dropna(subset=[factor_col]).copy()
    result = result.sort_values(["date", factor_col, "code"], kind="mergesort")
    grouped = result.groupby("date", sort=False)
    position = grouped.cumcount()
    count = grouped["code"].transform("size")
    result["group"] = (position * quantiles // count + 1).astype(int)
    return result


def evaluate_factor(
    frame: pd.DataFrame,
    factor_col: str,
    config: ResearchConfig = ResearchConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return summary metrics and daily IC/group-return observations."""
    grouped_frame = assign_quantiles(frame, factor_col, config.quantiles)
    summaries: list[dict[str, float | int | str]] = []
    daily_parts: list[pd.DataFrame] = []

    for horizon in config.horizons:
        return_col = f"forward_return_{horizon}d"
        work = grouped_frame[
            ["date", "code", factor_col, "group", return_col]
        ].dropna(subset=[factor_col, return_col])

        by_date = work.groupby("date", sort=False)
        rank_ic = (
            by_date[[factor_col, return_col]]
            .corr(method="spearman")
            .xs(factor_col, level=1)[return_col]
            .rename("rank_ic")
        )
        samples = by_date.size().rename("samples")
        group_returns = (
            work.groupby(["date", "group"])[return_col]
            .mean()
            .unstack("group")
            .reindex(columns=range(1, config.quantiles + 1))
            .rename(columns=lambda value: f"Q{value}")
        )

        daily = pd.concat([rank_ic, samples, group_returns], axis=1).reset_index()
        daily.loc[daily["samples"] < config.min_samples, "rank_ic"] = np.nan
        daily["horizon"] = horizon
        daily["top_bottom"] = daily[f"Q{config.quantiles}"] - daily["Q1"]
        daily_parts.append(daily)

        ic = daily["rank_ic"].dropna()
        ic_std = ic.std()
        summaries.append(
            {
                "factor": factor_col,
                "horizon": horizon,
                "periods": int(ic.count()),
                "rank_ic_mean": ic.mean(),
                "rank_ic_ir": ic.mean() / ic_std if ic_std else np.nan,
                "rank_ic_positive_ratio": (ic > 0).mean(),
                "top_bottom_mean": daily["top_bottom"].mean(),
            }
        )

    return pd.DataFrame(summaries), pd.concat(daily_parts, ignore_index=True)


def top_group_turnover(
    grouped_frame: pd.DataFrame,
    quantiles: int,
) -> pd.Series:
    """Calculate one-way turnover of the highest quantile using set overlap."""
    members = (
        grouped_frame.loc[grouped_frame["group"] == quantiles]
        .groupby("date")["code"]
        .agg(set)
    )
    values = [
        1 - 2 * len(previous & current) / (len(previous) + len(current))
        for previous, current in zip(members.iloc[:-1], members.iloc[1:])
    ]
    return pd.Series(values, index=members.index[1:], name="turnover")

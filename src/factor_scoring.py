"""Fixed-threshold factor scoring for one or two stock universes."""

from __future__ import annotations

import numpy as np
import pandas as pd


RECENT_WEIGHTS = {2023: 0.15, 2024: 0.20, 2025: 0.35, 2026: 0.30}
SCORE_WEIGHTS = {
    "adj_ic": 20,
    "adj_icir": 10,
    "adj_ic_hit": 5,
    "adj_layer_spearman": 15,
    "adj_spread": 15,
    "adj_spread_hit": 5,
    "recent_ic": 20,
}

POOL_STANDARDS = {
    "chinext": {
        "adj_ic": (0.010, 0.060),
        "adj_icir": (0.100, 0.500),
        "adj_ic_hit": (0.550, 0.680),
        "adj_layer_spearman": (0.050, 0.150),
        "adj_spread": (0.0010, 0.0080),
        "adj_spread_hit": (0.530, 0.630),
        "recent_ic": (0.010, 0.060),
    },
    "csi800": {
        "adj_ic": (0.010, 0.030),
        "adj_icir": (0.050, 0.250),
        "adj_ic_hit": (0.520, 0.590),
        "adj_layer_spearman": (0.010, 0.060),
        "adj_spread": (0.0003, 0.0020),
        "adj_spread_hit": (0.505, 0.550),
        "recent_ic": (0.005, 0.030),
    },
}


def _linear_score(values: pd.Series, low: float, high: float, weight: float) -> pd.Series:
    return ((values - low) / (high - low)).clip(0, 1) * weight


def prepare_scoring_inputs(
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build overall and annual IC inputs from the research pipeline outputs."""
    base = summary.loc[summary["horizon"].eq(horizon)].copy()
    base = base.rename(columns={"factor": "factor_name"})

    detail = daily.loc[daily["horizon"].eq(horizon)].copy()
    detail["date"] = pd.to_datetime(detail["date"])
    q_cols = sorted(
        [column for column in detail if column.startswith("Q")],
        key=lambda column: int(column[1:]),
    )
    group_rank = pd.Series(range(1, len(q_cols) + 1), index=q_cols)
    detail["layer_spearman"] = detail[q_cols].rank(axis=1).corrwith(group_rank, axis=1)

    layers = (
        detail.groupby("factor")
        .agg(
            layer_spearman_mean=("layer_spearman", "mean"),
            top_bottom_spread_mean=("top_bottom", "mean"),
            spread_positive_ratio=("top_bottom", lambda values: values.gt(0).mean()),
        )
        .rename_axis("factor_name")
        .reset_index()
    )
    overall = base[
        ["factor_name", "rank_ic_mean", "rank_ic_ir", "rank_ic_positive_ratio"]
    ].merge(layers, on="factor_name", validate="one_to_one")

    annual = (
        detail.assign(year=detail["date"].dt.year)
        .pivot_table(index="factor", columns="year", values="rank_ic", aggfunc="mean")
        .rename_axis(index="factor_name", columns=None)
        .reset_index()
    )
    return overall, annual


def score_pool(
    overall: pd.DataFrame,
    annual: pd.DataFrame,
    pool: str,
) -> pd.DataFrame:
    """Score factors on a fixed 100-point scale with stability gates."""
    standards = POOL_STANDARDS[pool]
    result = overall.merge(annual, on="factor_name", validate="one_to_one").copy()
    result["direction"] = np.where(
        result["rank_ic_mean"].abs() < 0.01,
        0,
        np.sign(result["rank_ic_mean"]),
    ).astype(int)

    direction = result["direction"]
    result["adj_ic"] = result["rank_ic_mean"].abs().where(direction.ne(0), 0)
    result["adj_icir"] = result["rank_ic_ir"].abs().where(direction.ne(0), 0)
    result["adj_ic_hit"] = np.where(
        direction.eq(1), result["rank_ic_positive_ratio"], 1 - result["rank_ic_positive_ratio"]
    )
    result["adj_layer_spearman"] = direction * result["layer_spearman_mean"]
    result["adj_spread"] = direction * result["top_bottom_spread_mean"]
    result["adj_spread_hit"] = np.where(
        direction.eq(1), result["spread_positive_ratio"], 1 - result["spread_positive_ratio"]
    )
    result.loc[direction.eq(0), ["adj_ic_hit", "adj_layer_spearman", "adj_spread", "adj_spread_hit"]] = 0

    base_metrics = [metric for metric in SCORE_WEIGHTS if metric != "recent_ic"]
    base_scores = []
    for metric in base_metrics:
        low, high = standards[metric]
        component = _linear_score(result[metric], low, high, SCORE_WEIGHTS[metric])
        result[f"{metric}_score"] = component.where(direction.ne(0), 0)
        base_scores.append(f"{metric}_score")
    result["base_score_70"] = result[base_scores].sum(axis=1)

    recent_years = list(RECENT_WEIGHTS)
    recent_ic = result[recent_years].mul(pd.Series(RECENT_WEIGHTS)).sum(axis=1, min_count=4)
    result["recent_ic"] = direction * recent_ic
    low, high = standards["recent_ic"]
    result["recent_score_20"] = _linear_score(result["recent_ic"], low, high, 20).where(direction.ne(0), 0)

    result["historical_ic"] = direction * result[list(range(2016, 2023))].mean(axis=1)
    result["recent_complete_ic"] = direction * result[[2023, 2024, 2025]].mean(axis=1)
    result["retention"] = np.where(
        direction.ne(0) & result["historical_ic"].ge(0.01),
        result["recent_complete_ic"] / result["historical_ic"],
        0,
    )
    result["decay_score_10"] = result["retention"].clip(0, 1) * 10
    result["pool_score_100"] = result["base_score_70"] + result["recent_score_20"] + result["decay_score_10"]

    adjusted_recent = result[recent_years].mul(direction, axis=0)
    result["recent_stable"] = (
        direction.ne(0)
        & adjusted_recent[2025].gt(0)
        & adjusted_recent[2026].gt(0)
        & adjusted_recent.gt(0).sum(axis=1).ge(3)
        & adjusted_recent.ge(-0.01).all(axis=1)
    )
    result["direction_clear"] = result["rank_ic_mean"].abs().ge(0.01)
    result["retention_pass"] = result["historical_ic"].ge(0.01) & result["retention"].ge(0.70)
    result["score_pass"] = result["pool_score_100"].ge(60)
    result["pool_pass"] = result[
        ["recent_stable", "direction_clear", "retention_pass", "score_pass"]
    ].all(axis=1)
    result["pool_rank"] = result["pool_score_100"].rank(method="min", ascending=False).astype(int)
    result["pool_grade"] = np.select(
        [
            ~result["direction_clear"],
            result["pool_pass"] & result["pool_score_100"].ge(85),
            result["pool_pass"] & result["pool_score_100"].ge(70),
        ],
        ["D", "A", "B"],
        default="C",
    )
    result["fail_reason"] = np.select(
        [
            ~result["direction_clear"],
            ~result["recent_stable"],
            ~result["retention_pass"],
            ~result["score_pass"],
        ],
        ["方向不明确", "近期方向不稳", "Retention不足", "PoolScore不足"],
        default="通过",
    )
    return result.sort_values("pool_score_100", ascending=False).reset_index(drop=True)


def combine_pools(chinext: pd.DataFrame, csi800: pd.DataFrame) -> pd.DataFrame:
    """Apply cross-universe consistency gates and produce a final ranking."""
    columns = [
        "factor_name", "pool_score_100", "direction", "recent_stable", "retention"
    ]
    result = chinext[columns].merge(
        csi800[columns], on="factor_name", suffixes=("_chinext", "_csi800"), validate="one_to_one"
    )
    result["final_score"] = result[["pool_score_100_chinext", "pool_score_100_csi800"]].mean(axis=1)
    result["direction_consistent"] = (
        result["direction_chinext"].ne(0)
        & result["direction_csi800"].ne(0)
        & result["direction_chinext"].eq(result["direction_csi800"])
    )
    result["retention_pass"] = result[["retention_chinext", "retention_csi800"]].ge(0.70).all(axis=1)
    result["both_score_pass"] = result[["pool_score_100_chinext", "pool_score_100_csi800"]].ge(60).all(axis=1)
    result["final_pass"] = (
        result["direction_consistent"]
        & result[["recent_stable_chinext", "recent_stable_csi800"]].all(axis=1)
        & result["retention_pass"]
        & result["both_score_pass"]
    )
    result["grade"] = np.select(
        [
            ~result["direction_consistent"],
            result["final_pass"] & result["final_score"].ge(85),
            result["final_pass"] & result["final_score"].ge(70),
        ],
        ["D", "A", "B"],
        default="C",
    )
    result["final_rank"] = result["final_score"].rank(method="min", ascending=False).astype(int)
    return result.sort_values("final_score", ascending=False).reset_index(drop=True)

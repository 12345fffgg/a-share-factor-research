"""Run a reproducible factor-research demo on synthetic A-share-style data."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.factor_research import (
    ResearchConfig,
    assign_quantiles,
    compute_forward_returns,
    evaluate_factor,
    top_group_turnover,
)


OUTPUT_DIR = Path("outputs")


def round_numeric(frame: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    """Round only numeric columns while preserving dates and labels."""
    result = frame.copy()
    numeric_columns = result.select_dtypes(include="number").columns
    result[numeric_columns] = result[numeric_columns].round(digits)
    return result


def build_demo_panel(seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=420)
    codes = [f"DEMO{i:04d}" for i in range(160)]
    market = rng.normal(0.0003, 0.009, len(dates))
    rows: list[pd.DataFrame] = []

    for index, code in enumerate(codes):
        idiosyncratic = rng.normal(0, 0.018, len(dates))
        returns = market + idiosyncratic
        close = (12 + index / 18) * np.cumprod(1 + returns)
        turnover = rng.lognormal(-2.2, 0.55, len(dates))
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "code": code,
                    "close": close,
                    "adjustment_factor": 1.0,
                    "turnover": turnover,
                }
            )
        )

    panel = pd.concat(rows, ignore_index=True).sort_values(["code", "date"])
    by_code = panel.groupby("code", sort=False)
    panel["momentum_20d"] = by_code["close"].pct_change(20)
    panel["reversal_5d"] = -by_code["close"].pct_change(5)
    panel["volatility_20d"] = (
        by_code["close"].pct_change().groupby(panel["code"]).rolling(20).std()
        .reset_index(level=0, drop=True)
    )
    panel["turnover_signal"] = np.log1p(panel["turnover"])
    return panel.reset_index(drop=True)


def main() -> None:
    config = ResearchConfig()
    panel = compute_forward_returns(build_demo_panel(), config.horizons)
    factors = ["momentum_20d", "reversal_5d", "volatility_20d", "turnover_signal"]
    summaries: list[pd.DataFrame] = []
    daily_results: list[pd.DataFrame] = []
    turnover_rows: list[dict[str, float | str]] = []

    for factor in factors:
        summary, daily = evaluate_factor(panel, factor, config)
        summaries.append(summary)
        daily.insert(0, "factor", factor)
        daily_results.append(daily)
        groups = assign_quantiles(panel, factor, config.quantiles)
        turnover = top_group_turnover(groups, config.quantiles)
        turnover_rows.append(
            {"factor": factor, "turnover_mean": turnover.mean(), "turnover_median": turnover.median()}
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    round_numeric(pd.concat(summaries, ignore_index=True)).to_csv(
        OUTPUT_DIR / "factor_summary.csv", index=False, encoding="utf-8"
    )
    round_numeric(pd.concat(daily_results, ignore_index=True)).to_csv(
        OUTPUT_DIR / "daily_factor_results.csv", index=False, encoding="utf-8"
    )
    round_numeric(pd.DataFrame(turnover_rows)).to_csv(
        OUTPUT_DIR / "turnover_summary.csv", index=False, encoding="utf-8"
    )
    print("Demo complete. Results written to outputs/.")


if __name__ == "__main__":
    main()

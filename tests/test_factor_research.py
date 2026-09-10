import unittest

import pandas as pd

from src.factor_research import assign_quantiles, compute_forward_returns


class FactorResearchTests(unittest.TestCase):
    def test_forward_returns_do_not_cross_stock_boundaries(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02"] * 2),
                "code": ["A", "A", "B", "B"],
                "close": [10.0, 11.0, 20.0, 18.0],
                "adjustment_factor": [1.0] * 4,
            }
        )
        result = compute_forward_returns(frame, [1])
        returns = result.groupby("code")["forward_return_1d"].first()
        self.assertAlmostEqual(returns["A"], 0.1)
        self.assertAlmostEqual(returns["B"], -0.1)

    def test_quantile_assignment_is_complete_and_bounded(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01"] * 10),
                "code": [f"S{i}" for i in range(10)],
                "factor": range(10),
            }
        )
        result = assign_quantiles(frame, "factor", 5)
        self.assertEqual(result["group"].value_counts().sort_index().tolist(), [2] * 5)
        self.assertEqual((result["group"].min(), result["group"].max()), (1, 5))


if __name__ == "__main__":
    unittest.main()

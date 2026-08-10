"""End-to-end: the pipeline recovers the generator's ground truth, and the
committed sample CSVs load cleanly through the validator."""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
from generate_synthetic import TRUE_ALPHA_ANNUAL, TRUE_BETA, generate  # noqa: E402

from src.benchmark import alpha_annualized, beta, r_squared  # noqa: E402
from src.data_loader import load_returns  # noqa: E402


def test_pipeline_recovers_ground_truth_at_large_n():
    # At N=50,000 the standard errors are ~0.003 (beta) and ~0.9% (annual alpha),
    # so these tolerances are several sigma wide yet still catch real formula bugs.
    df = generate(n_days=50_000, seed=123)
    port = pd.Series(df["portfolio_return"].values)
    bench = pd.Series(df["benchmark_return"].values)
    assert beta(port, bench) == pytest.approx(TRUE_BETA, abs=0.02)
    assert alpha_annualized(port, bench) == pytest.approx(TRUE_ALPHA_ANNUAL, abs=0.03)
    assert r_squared(port, bench) == pytest.approx(0.75, abs=0.02)


def test_committed_synthetic_sample_loads_clean():
    df, messages = load_returns(REPO_ROOT / "data" / "sample_synthetic.csv")
    assert len(df) == 756
    assert messages == []


def test_committed_real_sample_loads_clean():
    df, messages = load_returns(REPO_ROOT / "data" / "sample_real.csv")
    assert len(df) > 700
    assert [m for m in messages if m.severity == "warning"] == []

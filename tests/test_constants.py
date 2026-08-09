import pytest

from src.constants import TRADING_DAYS, daily_rf


def test_trading_days():
    assert TRADING_DAYS == 252


def test_daily_rf_is_simple_division():
    assert daily_rf(0.0504) == pytest.approx(0.0002)
    assert daily_rf(0.0) == pytest.approx(0.0)

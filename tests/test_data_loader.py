import io

import pandas as pd
import pytest

from src.data_loader import ValidationError, load_returns


def make_csv(rows, header="date,portfolio_return,benchmark_return"):
    return io.StringIO(header + "\n" + "\n".join(rows))


def valid_rows(n=70):
    dates = pd.bdate_range("2025-01-01", periods=n)
    return [f"{d.date()},0.001,0.002" for d in dates]


def test_valid_file_loads_clean():
    df, messages = load_returns(make_csv(valid_rows()))
    assert list(df.columns) == ["portfolio", "benchmark"]
    assert df.index.is_monotonic_increasing
    assert len(df) == 70
    assert messages == []


def test_missing_column_is_error():
    with pytest.raises(ValidationError, match="benchmark_return"):
        load_returns(make_csv(["2025-01-02,0.001"], header="date,portfolio_return"))


def test_case_insensitive_columns_and_extra_column_note():
    header = "Date,Portfolio_Return,BENCHMARK_RETURN,notes"
    rows = [r + ",x" for r in valid_rows()]
    df, messages = load_returns(make_csv(rows, header=header))
    assert len(df) == 70
    assert any("extra" in m.text.lower() for m in messages)


def test_unparseable_date_is_error():
    rows = valid_rows()
    rows[5] = "not-a-date,0.001,0.002"
    with pytest.raises(ValidationError, match="not-a-date"):
        load_returns(make_csv(rows))


def test_non_numeric_return_is_error():
    rows = valid_rows()
    rows[3] = rows[3].replace(",0.001,", ",abc,")
    with pytest.raises(ValidationError, match="abc"):
        load_returns(make_csv(rows))


def test_duplicate_dates_error():
    rows = valid_rows()
    rows[10] = rows[9]
    with pytest.raises(ValidationError, match="Duplicate"):
        load_returns(make_csv(rows))


def test_unsorted_dates_sorted_with_info():
    rows = valid_rows()
    rows.reverse()
    df, messages = load_returns(make_csv(rows))
    assert df.index.is_monotonic_increasing
    assert any(m.severity == "info" and "sorted" in m.text.lower() for m in messages)


def test_missing_values_dropped_with_warning():
    rows = valid_rows()
    rows[7] = rows[7].replace(",0.002", ",")  # blank benchmark cell
    df, messages = load_returns(make_csv(rows))
    assert len(df) == 69
    assert any(m.severity == "warning" and "Dropped 1" in m.text for m in messages)


def test_percent_scale_warning():
    dates = pd.bdate_range("2025-01-01", periods=70)
    rows = [f"{d.date()},0.8,1.1" for d in dates]  # clearly percent-scale values
    _, messages = load_returns(make_csv(rows))
    assert any("percent" in m.text.lower() for m in messages)


def test_extreme_return_warning():
    rows = valid_rows()
    rows[4] = rows[4].replace(",0.001,", ",0.75,")
    _, messages = load_returns(make_csv(rows))
    assert any("large daily return" in m.text.lower() for m in messages)


def test_too_few_rows_error():
    with pytest.raises(ValidationError, match="at least 60"):
        load_returns(make_csv(valid_rows(50)))


def test_empty_file_is_validation_error():
    with pytest.raises(ValidationError, match="empty"):
        load_returns(io.StringIO(""))


def test_binary_file_is_validation_error():
    with pytest.raises(ValidationError):
        load_returns(io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50))


def test_mixed_timezone_dates_are_validation_error():
    rows = valid_rows()
    rows[0] = "2025-01-01T00:00:00+00:00,0.001,0.002"
    rows[1] = "2025-01-02T00:00:00+05:30,0.001,0.002"
    with pytest.raises(ValidationError):
        load_returns(make_csv(rows))

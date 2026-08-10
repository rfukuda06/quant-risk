"""CSV loading and validation for portfolio/benchmark return files.

Pure (no Streamlit): returns a clean DataFrame plus info/warning messages;
raises ValidationError with a user-facing message when analysis is impossible.
"""

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = {"date", "portfolio_return", "benchmark_return"}
MIN_ROWS = 60
PERCENT_SNIFF_THRESHOLD = 0.05
EXTREME_RETURN_THRESHOLD = 0.5


class ValidationError(Exception):
    """The uploaded CSV cannot be analyzed; the message is user-facing."""


@dataclass
class LoaderMessage:
    severity: str  # "info" or "warning"
    text: str


def load_returns(file_or_buffer) -> tuple[pd.DataFrame, list[LoaderMessage]]:
    """Parse and validate a returns CSV per the design spec's §5 rules.

    Output: DataFrame with ascending DatetimeIndex and float64 columns
    'portfolio' and 'benchmark', plus a list of LoaderMessages.
    """
    messages: list[LoaderMessage] = []
    try:
        raw = pd.read_csv(file_or_buffer)
    except pd.errors.EmptyDataError:
        raise ValidationError(
            f"The file is empty. Upload a CSV with a header row and at least {MIN_ROWS} data rows."
        )
    except Exception as e:
        raise ValidationError(f"Could not read the file as CSV: {e}") from e
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValidationError(
            f"Missing required column(s): {', '.join(sorted(missing))}. "
            "Expected columns: date, portfolio_return, benchmark_return."
        )
    extra = set(raw.columns) - REQUIRED_COLUMNS
    if extra:
        messages.append(
            LoaderMessage("info", f"Ignoring extra column(s): {', '.join(sorted(extra))}.")
        )

    try:
        dates = pd.to_datetime(raw["date"], errors="coerce")
    except ValueError as e:
        raise ValidationError(
            f"Date column has mixed timezones or an unsupported format: {e}"
        ) from e
    bad_dates = raw.loc[dates.isna() & raw["date"].notna(), "date"]
    if not bad_dates.empty:
        examples = ", ".join(str(v) for v in bad_dates.head(3))
        raise ValidationError(f"Unparseable date(s), e.g.: {examples}.")

    parsed = {}
    for col, name in [("portfolio_return", "portfolio"), ("benchmark_return", "benchmark")]:
        coerced = pd.to_numeric(raw[col], errors="coerce")
        garbage = raw.loc[coerced.isna() & raw[col].notna(), col]
        if not garbage.empty:
            examples = ", ".join(str(v) for v in garbage.head(3))
            raise ValidationError(f"Non-numeric value(s) in {col}, e.g.: {examples}.")
        parsed[name] = coerced

    df = pd.DataFrame(parsed)
    df.index = pd.DatetimeIndex(dates)
    incomplete = df.index.isna() | df.isna().any(axis=1)
    if incomplete.any():
        messages.append(
            LoaderMessage("warning", f"Dropped {int(incomplete.sum())} row(s) with missing values.")
        )
        df = df.loc[~incomplete]

    if df.index.duplicated().any():
        dupes = df.index[df.index.duplicated()].strftime("%Y-%m-%d")
        raise ValidationError(f"Duplicate date(s): {', '.join(dupes[:3])}.")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        messages.append(
            LoaderMessage("info", "Rows were not in chronological order; sorted by date.")
        )

    if len(df) < MIN_ROWS:
        raise ValidationError(f"Need at least {MIN_ROWS} rows of data; got {len(df)}.")

    all_values = pd.concat([df["portfolio"], df["benchmark"]])
    if all_values.abs().median() > PERCENT_SNIFF_THRESHOLD:
        messages.append(
            LoaderMessage(
                "warning",
                "Values look like percentages, not decimals (median |return| > 5%). "
                "Expected decimals, e.g. 0.0021 for 0.21%. Not auto-converting.",
            )
        )

    extreme = df[(df.abs() > EXTREME_RETURN_THRESHOLD).any(axis=1)].index.strftime("%Y-%m-%d")
    if len(extreme) > 0:
        messages.append(
            LoaderMessage(
                "warning",
                f"Suspiciously large daily return(s) (>50%) on: {', '.join(extreme[:3])}. "
                "Check the data.",
            )
        )

    return df.astype(float), messages

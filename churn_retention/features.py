"""Customer-history features for one prediction date; no labels or training.

The listening and renewal summaries retain notebook 01's EDA definitions.
Pass source events that were available at prediction time; event dates alone
cannot establish ingestion availability or validate supplied label periods.
"""

import io
import json
from collections import Counter
from contextlib import closing
from pathlib import Path, PurePosixPath

import pandas as pd

FEATURE_COLUMNS = (
    "latest_auto_renew_status", "cancellation_events_28d", "transaction_events_28d",
    "days_since_last_transaction", "recorded_active_days_7d",
    "recorded_active_days_28d", "recorded_active_days_90d", "recorded_minutes_28d",
    "days_since_last_positive_log", "recent_to_baseline_ratio", "history_coverage",
    "observed_history_span_days", "baseline_comparison_eligible",
    "has_unusable_records_90d", "valid_days_28d",
)


class _ArchiveStream(io.RawIOBase):
    """Adapt decompressed archive blocks to pandas without a full disk extraction."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = iter(blocks)
        self.pending = memoryview(b"")

    def readable(self):
        return True

    def readinto(self, target):
        while not self.pending:
            try:
                self.pending = memoryview(next(self.blocks))
            except StopIteration:
                return 0
        count = min(len(target), len(self.pending))
        target[:count] = self.pending[:count]
        self.pending = self.pending[count:]
        return count


def csv_chunks(path, *, columns=None, dtype=None, chunksize=100_000):
    """Stream an explicitly named CSV or its .7z member without extracting to disk."""
    path = Path(path)
    if path.suffix != ".7z":
        with pd.read_csv(path, usecols=columns, dtype=dtype, chunksize=chunksize) as reader:
            yield from reader
        return
    import libarchive  # Optional archive dependency; no reads/installations at import time.

    with libarchive.file_reader(str(path)) as archive:
        for entry in archive:
            if PurePosixPath(entry.pathname).name != path.name.removesuffix(".7z"):
                continue
            with (
                io.BufferedReader(_ArchiveStream(entry.get_blocks(1024 * 1024))) as stream,
                pd.read_csv(stream, usecols=columns, dtype=dtype, chunksize=chunksize) as reader,
            ):
                yield from reader
            return
    raise ValueError(f"The supplied archive does not contain {path.name.removesuffix('.7z')}.")


def transaction_chunks(path, chunksize=100_000, *, all_columns=False):
    """Read transaction fields; all_columns=True also exposes source fields for previews."""
    columns = None if all_columns else ["msno", "transaction_date", "is_auto_renew", "is_cancel"]
    yield from csv_chunks(path, columns=columns, chunksize=chunksize,
                          dtype={"msno": "string", "transaction_date": "string"})


def source_date_ranges(path, date_columns, cache_path):
    """Inspect full-source date columns, caching aggregates against source file stats.

    Only requested date columns are loaded. Labels with no date columns get a
    schema check and an explicit no-date row. Invalid dates are counted separately.
    This inspects source coverage; it does not filter customers or prediction dates.
    """
    path, cache_path = Path(path), Path(cache_path)
    fingerprint = {"version": 1, "source": str(path.resolve()),
                   "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                   "date_columns": list(date_columns)}
    saved = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    previous = saved.get(path.name, {})
    if previous.get("fingerprint") == fingerprint:
        print("Using full-source date audit:", path.name, flush=True)
        return previous["ranges"]
    print("Checking source dates:", path.name, flush=True)
    if not date_columns:
        with closing(csv_chunks(path, chunksize=1)) as reader:
            columns = next(reader).columns.tolist()
        if set(columns) != {"msno", "is_churn"}:
            raise ValueError("Expected a date-free label file with msno and is_churn.")
        ranges = [{"file": path.name.removesuffix(".7z"), "date_column": "None",
                   "earliest": None, "latest": None, "rows_checked": None,
                   "invalid_dates": None, "scope": "Label file: no date columns"}]
    else:
        counts = {column: Counter() for column in date_columns}
        rows = 0
        for chunk in csv_chunks(path, columns=list(date_columns), dtype="string",
                                chunksize=1_000_000):
            rows += len(chunk)
            for column in date_columns:
                counts[column].update(chunk[column].value_counts(dropna=False).to_dict())
            if rows % 10_000_000 == 0:
                print(f"{path.name}: checked {rows:,} rows", flush=True)
        ranges = []
        for column, values in counts.items():
            dates = pd.to_datetime(pd.Series(list(values), dtype="string"),
                                   format="%Y%m%d", errors="coerce")
            valid = dates.dropna()
            ranges.append({
                "file": path.name.removesuffix(".7z"), "date_column": column,
                "earliest": None if valid.empty else valid.min().date().isoformat(),
                "latest": None if valid.empty else valid.max().date().isoformat(),
                "rows_checked": rows,
                "invalid_dates": sum(count for count, invalid in zip(values.values(), dates.isna())
                                     if invalid),
                "scope": "Full source file; no customer or cutoff filter",
            })
    saved[path.name] = {"fingerprint": fingerprint, "ranges": ranges}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(saved, indent=2, allow_nan=False))
    return ranges


def listening_windows(history, customer_ids, cutoff, windows=(7, 28, 90)):
    """Build label-free listening summaries, with coverage and quality reported."""
    dates = pd.to_datetime(history["date"], format="%Y%m%d", errors="coerce")
    selected_ids = history["msno"].isin(customer_ids)
    keep = selected_ids & dates.lt(cutoff)
    rows = history.loc[keep].copy()
    rows["parsed_date"] = dates.loc[keep]
    duplicate_day = rows.duplicated(["msno", "parsed_date"], keep=False)
    valid_duration = (rows["total_secs"].notna() & rows["total_secs"].ge(0)
                      & ~rows["total_secs"].isin([float("inf"), float("-inf")]))
    unusable = duplicate_day | ~valid_duration
    valid = rows.loc[~unusable].copy()
    positive = valid.loc[valid["total_secs"].gt(0)]
    result = pd.DataFrame(index=pd.Index(sorted(customer_ids), name="msno"))

    def counts(frame, column=None):
        grouped = frame.groupby("msno")
        values = grouped.size() if column is None else grouped[column].nunique()
        return values.reindex(result.index, fill_value=0)

    def minutes(frame):
        return frame.groupby("msno")["total_secs"].sum().reindex(result.index) / 60

    result["first_valid_log_date"] = valid.groupby("msno")["parsed_date"].min().reindex(result.index)
    result["observed_history_span_days"] = (cutoff - result["first_valid_log_date"]).dt.days
    last_positive = positive.groupby("msno")["parsed_date"].max().reindex(result.index)
    result["days_since_last_positive_log"] = (cutoff - last_positive).dt.days
    for suffix, days in [("all", None), *[(f"{n}d", n) for n in windows]]:
        start = None if days is None else cutoff - pd.Timedelta(days=days)
        raw_window = rows if start is None else rows.loc[rows["parsed_date"].ge(start)]
        valid_window = valid if start is None else valid.loc[valid["parsed_date"].ge(start)]
        positive_window = positive if start is None else positive.loc[positive["parsed_date"].ge(start)]
        result[f"source_rows_{suffix}"] = counts(raw_window)
        result[f"valid_days_{suffix}"] = counts(valid_window, "parsed_date")
        result[f"recorded_active_days_{suffix}"] = counts(positive_window, "parsed_date")
        result[f"recorded_minutes_{suffix}"] = minutes(valid_window)

    recent_start = cutoff - pd.Timedelta(days=28)
    baseline_start = cutoff - pd.Timedelta(days=90)
    midpoint = cutoff - pd.Timedelta(days=14)
    recent = valid.loc[valid["parsed_date"].ge(recent_start)]
    baseline = valid.loc[valid["parsed_date"].ge(baseline_start)
                         & valid["parsed_date"].lt(recent_start)]
    unusable_90_ids = set(rows.loc[unusable & rows["parsed_date"].ge(baseline_start), "msno"])
    result["observed_before_90d"] = result["first_valid_log_date"].le(baseline_start)
    result["has_unusable_records_90d"] = result.index.isin(unusable_90_ids)
    eligible = result["observed_before_90d"] & ~result["has_unusable_records_90d"]
    result["baseline_comparison_eligible"] = eligible
    # Rates describe recorded source activity. Absent window records contribute
    # zero only to this explicitly defined, older-observed comparison group.
    result["recent_recorded_minutes_per_day"] = (minutes(recent).fillna(0) / 28).where(eligible)
    result["baseline_recorded_minutes_per_day"] = (minutes(baseline).fillna(0) / 62).where(eligible)
    baseline_rate = result["baseline_recorded_minutes_per_day"].replace(0, float("nan"))
    result["recent_to_baseline_ratio"] = result["recent_recorded_minutes_per_day"] / baseline_rate

    early_ids = set(recent.loc[recent["parsed_date"].lt(midpoint), "msno"])
    late_ids = set(recent.loc[recent["parsed_date"].ge(midpoint), "msno"])
    result["observed_both_halves"] = result.index.isin(early_ids & late_ids)
    early_days = counts(positive.loc[positive["parsed_date"].ge(recent_start)
                                       & positive["parsed_date"].lt(midpoint)], "parsed_date")
    late_days = counts(positive.loc[positive["parsed_date"].ge(midpoint)], "parsed_date")
    result["recorded_day_change_when_observed"] = (late_days - early_days).where(result["observed_both_halves"])
    result["history_coverage"] = "No pre-cutoff source records"
    result.loc[result["source_rows_all"].gt(0), "history_coverage"] = "Older records only; none in last 28 days"
    result.loc[result["source_rows_28d"].gt(0), "history_coverage"] = "Source records in last 28 days"
    quality = {
        "cached_rows": len(history), "selected_pre_cutoff_rows": len(rows),
        "invalid_date_rows": int((selected_ids & dates.isna()).sum()),
        "rows_on_or_after_cutoff_excluded": int((selected_ids & dates.ge(cutoff)).sum()),
        "rows_excluded_for_duplicate_day_keys": int(duplicate_day.sum()),
        "rows_with_invalid_duration": int((~valid_duration).sum()),
        "usable_listening_rows": len(valid),
        "customers_without_any_source_records": int(result["source_rows_all"].eq(0).sum()),
        "customers_without_recent_source_records": int(result["source_rows_28d"].eq(0).sum()),
        "customers_older_records_only": int(result["history_coverage"].eq("Older records only; none in last 28 days").sum()),
        "first_selected_date": None if rows.empty else rows["parsed_date"].min().date().isoformat(),
        "last_selected_date": None if rows.empty else rows["parsed_date"].max().date().isoformat(),
    }
    return result, quality


def renewal_by_customer(chunks, customer_ids, cutoff):
    """Summarise all prior events and recent windows, preserving uncertain flags."""
    selected = []
    for chunk in chunks:
        rows = chunk.loc[chunk["msno"].isin(customer_ids)].copy()
        rows["parsed_date"] = pd.to_datetime(rows["transaction_date"],
                                             format="%Y%m%d", errors="coerce")
        rows = rows.loc[rows["parsed_date"].lt(cutoff)]
        if not rows.empty:
            selected.append(rows)
    result = pd.DataFrame(index=pd.Index(sorted(customer_ids), name="msno"))
    result["latest_auto_renew_status"] = "No earlier transaction"
    for suffix in ["all", "7d", "28d", "90d"]:
        result[f"transaction_events_{suffix}"] = 0
        result[f"cancellation_events_{suffix}"] = 0
    result["days_since_last_transaction"] = float("nan")
    if not selected:
        return result
    events = pd.concat(selected, ignore_index=True)
    last_dates = events.groupby("msno")["parsed_date"].transform("max")
    latest = events.loc[events["parsed_date"].eq(last_dates)]

    def flag_status(values):
        if not values.isin([0, 1]).all() or values.nunique() != 1:
            return "Ambiguous latest-day flags"
        return "Auto-renew 1" if values.iloc[0] == 1 else "Auto-renew 0"

    statuses = latest.groupby("msno")["is_auto_renew"].agg(flag_status)
    result.loc[statuses.index, "latest_auto_renew_status"] = statuses
    last_by_customer = latest.groupby("msno")["parsed_date"].max().reindex(result.index)
    result["days_since_last_transaction"] = (cutoff - last_by_customer).dt.days
    for suffix, days in [("all", None), ("7d", 7), ("28d", 28), ("90d", 90)]:
        frame = events if days is None else events.loc[
            events["parsed_date"].ge(cutoff - pd.Timedelta(days=days))]
        result[f"transaction_events_{suffix}"] = frame.groupby("msno").size().reindex(result.index, fill_value=0)
        cancellations = frame.loc[frame["is_cancel"].eq(1)]
        result[f"cancellation_events_{suffix}"] = cancellations.groupby("msno").size().reindex(result.index, fill_value=0)
    return result


def build_features(listening_history, transactions, customer_ids, prediction_date):
    """Return a feature table and quality counts, preserving every requested ID.

    Inputs: listening columns msno/date/total_secs; an iterable of transaction
    frames with msno/transaction_date/is_auto_renew/is_cancel; unique string IDs;
    and a timezone-free prediction date at midnight. Windows exclude that day.
    Output keys are customer_id and prediction_date. Labels are never required.
    """
    ids = list(customer_ids)
    if not ids or any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("Supply non-empty string customer IDs.")
    if len(ids) != len(set(ids)):
        raise ValueError("Customer IDs must be unique.")
    cutoff = pd.Timestamp(prediction_date)
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise ValueError("Prediction date must be a timezone-free date at midnight.")
    required = {"msno", "date", "total_secs"}
    if not required.issubset(listening_history.columns):
        raise ValueError(f"Listening history requires {sorted(required)}.")
    history = listening_history[list(required)].copy()
    history["total_secs"] = pd.to_numeric(history["total_secs"], errors="coerce")
    listening, quality = listening_windows(history, ids, cutoff)
    transaction_quality = {"invalid_date_rows": 0, "invalid_auto_renew_rows": 0,
                           "invalid_cancel_rows": 0}

    def checked_transactions():
        columns = ["msno", "transaction_date", "is_auto_renew", "is_cancel"]
        for chunk in transactions:
            if not set(columns).issubset(chunk.columns):
                raise ValueError(f"Transactions require {columns}.")
            rows = chunk.loc[chunk["msno"].isin(ids), columns].copy()
            dates = pd.to_datetime(rows["transaction_date"], format="%Y%m%d", errors="coerce")
            transaction_quality["invalid_date_rows"] += int(dates.isna().sum())
            rows = rows.loc[dates.lt(cutoff)]
            for flag, key in [("is_auto_renew", "invalid_auto_renew_rows"),
                              ("is_cancel", "invalid_cancel_rows")]:
                rows[flag] = pd.to_numeric(rows[flag], errors="coerce")
                transaction_quality[key] += int((~rows[flag].isin([0, 1])).sum())
            yield rows

    renewal = renewal_by_customer(checked_transactions(), ids, cutoff)
    features = listening.join(renewal, validate="one_to_one")[list(FEATURE_COLUMNS)]
    features = features.rename_axis("customer_id").reset_index()
    features["customer_id"] = features["customer_id"].astype("string")
    features.insert(1, "prediction_date", cutoff.date().isoformat())
    return features, {"listening": quality, "transactions": transaction_quality}

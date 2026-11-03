"""Rebuild churn labels with the provider's rules (data/raw/WSDMChurnLabeller.scala).

Used to check which expiry period each supplied label release describes and,
later, to build labelled cohorts for earlier months. Labels are outcomes: never
pass these columns to a feature function.
"""

import pandas as pd

LABEL_COLUMNS = ("last_expire", "renewal_date", "renewal_gap_days", "is_churn",
                 "outcome_window_complete")
TRANSACTION_COLUMNS = ("msno", "payment_method_id", "payment_plan_days", "plan_list_price",
                       "transaction_date", "membership_expire_date", "is_cancel")


def _dates(values):
    return pd.to_datetime(pd.Series(values).astype("string"), format="%Y%m%d", errors="coerce")


def provider_order(transactions):
    """Sort events like the provider: by date; on one date, plan signature descending
    (text of price + days + method); for one plan, renewals before cancellations,
    renewals by expiry ascending and cancellations by expiry descending."""
    rows = transactions.copy()
    rows["_plan_signature"] = (rows["plan_list_price"].astype(str)
                               + rows["payment_plan_days"].astype(str)
                               + rows["payment_method_id"].astype(str))
    expiry = rows["membership_expire_date"].astype("int64")
    rows["_signed_expiry"] = expiry.where(rows["is_cancel"].eq(0), -expiry)
    rows = rows.sort_values(
        ["msno", "transaction_date", "_plan_signature", "is_cancel", "_signed_expiry"],
        ascending=[True, True, False, True, True], kind="mergesort")
    return rows.drop(columns=["_plan_signature", "_signed_expiry"])


def provider_labels(transactions, history_start, history_end, expiry_start, expiry_end,
                    data_end, churn_gap_days=30):
    """Label customers whose last expiry (from history-window events) falls in the
    expiry window. All dates are inclusive and given as YYYYMMDD integers.

    A customer renews when their first non-cancel event after history_end comes
    fewer than churn_gap_days after the expiry (earlier cancellations can move the
    expiry earlier). No such event means churn. outcome_window_complete is False
    when a churn label relies on days after data_end, i.e. the label is immature.
    """
    rows = transactions[list(TRANSACTION_COLUMNS)].copy()
    for column in ("transaction_date", "membership_expire_date", "is_cancel"):
        rows[column] = rows[column].astype("int64")
    history = rows[rows["transaction_date"].between(history_start, history_end)]
    last_expire = provider_order(history).groupby("msno")["membership_expire_date"].last()
    candidates = last_expire[last_expire.between(expiry_start, expiry_end)]
    result = pd.DataFrame({"last_expire": candidates})

    future = rows[rows["transaction_date"].between(history_end + 1, data_end)
                  & rows["msno"].isin(result.index)]
    future = provider_order(future)
    renewal = future["is_cancel"].eq(0).astype(int)
    renewals_before = renewal.groupby(future["msno"]).cumsum() - renewal
    first_part = future[renewals_before.eq(0)]
    cancelled_expiry = first_part[first_part["is_cancel"].eq(1)].groupby("msno")[
        "membership_expire_date"].min()
    renewal_date = first_part[first_part["is_cancel"].eq(0)].groupby("msno")[
        "transaction_date"].first()

    expiry = pd.concat([result["last_expire"], cancelled_expiry.reindex(result.index)],
                       axis=1).min(axis=1).astype("int64")
    result["renewal_date"] = renewal_date.reindex(result.index).astype("Int64")
    gap = (_dates(result["renewal_date"]).values - _dates(expiry).values)
    result["renewal_gap_days"] = pd.Series(gap, index=result.index).dt.days.astype("Int64")
    renewed = result["renewal_gap_days"].lt(churn_gap_days).fillna(False)
    result["is_churn"] = (~renewed).astype("int8")
    window_end = _dates(expiry) + pd.Timedelta(days=churn_gap_days - 1)
    observed_to = pd.Timestamp(str(data_end))
    result["outcome_window_complete"] = (renewed | (window_end.values <= observed_to)).astype(bool)
    result.index.name = "msno"
    return result[list(LABEL_COLUMNS)]

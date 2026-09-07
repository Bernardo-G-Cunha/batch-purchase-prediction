import pandas as pd

def create_sessions(
    events,
    visitor_col="visitorid",
    timestamp_col="timestamp",
    timeout_minutes=30
):
    events = events.sort_values(
        [visitor_col, timestamp_col]
    ).copy()

    time_diff = (
        events
        .groupby(visitor_col)[timestamp_col]
        .diff()
    )

    session_timeout = pd.Timedelta(minutes=timeout_minutes)

    new_session = (
        time_diff.isna()
        | (time_diff > session_timeout)
    )

    events["session_id"] = (
        new_session
        .groupby(events[visitor_col])
        .cumsum()
    )

    return events


def aggregate_session_statistics(
    events,
    visitor_col="visitorid",
    timestamp_col="timestamp",
    session_col="session_id"
):
    sessions = (
        events
        .groupby([visitor_col, session_col])
        .agg(
            session_start=(timestamp_col, "min"),
            session_end=(timestamp_col, "max")
        )
        .reset_index()
    )

    sessions["session_duration_min"] = (
        sessions["session_end"] - sessions["session_start"]
    ).dt.total_seconds() / 60

    sessions["date"] = (
        sessions["session_start"].dt.floor("D")
    )

    daily_session_stats = (
        sessions
        .groupby([visitor_col, "date"])
        .agg(
            total_duration=("session_duration_min", "sum"),
            sessions=("session_duration_min", "count")
        )
        .reset_index()
    )

    return daily_session_stats


def add_rolling_session_features(
    daily_session_stats,
    snapshot_data,
    visitor_col="visitorid",
    date_col="date",
    window="30D"
):
    daily_session_stats = (
        daily_session_stats
        .merge(
            snapshot_data[[visitor_col, date_col]],
            on=[visitor_col, date_col],
            how="right"
        )
    )

    daily_session_stats["sessions"] = (
        daily_session_stats["sessions"]
        .fillna(0)
        .astype("int16")
    )

    daily_session_stats["total_duration"] = (
        daily_session_stats["total_duration"]
        .fillna(0)
    )

    rolling_sessions = (
        daily_session_stats
        .set_index(date_col)
        .groupby(visitor_col)["sessions"]
        .rolling(window, closed="left")
        .sum()
        .reset_index()
        .rename(columns={
            "sessions": "user_sessions_30d"
        })
    )

    rolling_duration = (
        daily_session_stats
        .set_index(date_col)
        .groupby(visitor_col)["total_duration"]
        .rolling(window, closed="left")
        .sum()
        .reset_index()
        .rename(columns={
            "total_duration": "total_duration_30d"
        })
    )

    session_features = rolling_sessions.merge(
        rolling_duration,
        on=[visitor_col, date_col],
        how="inner"
    )

    session_features["user_sessions_30d"] = (
        session_features["user_sessions_30d"]
        .fillna(0)
        .astype("int16")
    )

    session_features["total_duration_30d"] = (
        session_features["total_duration_30d"]
        .fillna(0)
    )

    session_features["avg_session_duration_30d"] = (
        session_features["total_duration_30d"]
        / session_features["user_sessions_30d"]
    ).fillna(0)

    return session_features[
        [
            visitor_col,
            date_col,
            "user_sessions_30d",
            "avg_session_duration_30d"
        ]
    ]


def create_purchase_target(user_data, transactions, days):
    targets = user_data[["visitorid", "date"]].copy()

    transaction_dates = transactions.copy()

    offsets = range(days)

    expanded = pd.concat(
        [
            transaction_dates.assign(
                date=transaction_dates["date"] - pd.Timedelta(days=offset)
            )
            for offset in offsets
        ],
        ignore_index=True
    )

    expanded = expanded.drop_duplicates(
        ["visitorid", "date"]
    )

    expanded[f"purchase_next_{days}d"] = 1

    targets = targets.merge(
        expanded,
        on=["visitorid", "date"],
        how="left"
    )

    targets[f"purchase_next_{days}d"] = (
        targets[f"purchase_next_{days}d"]
        .fillna(0)
        .astype("int8")
    )

    return user_data.merge(
        targets,
        on=["visitorid", "date"],
        how="left"
    )
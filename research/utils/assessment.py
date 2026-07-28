import pandas as pd

def missing_summary(df):
    summary = pd.DataFrame(index=df.columns)

    summary["null"] = df.isna().sum()
    summary["empty_string"] = 0

    text_cols = df.select_dtypes(include="object").columns

    for col in text_cols:
        summary.loc[col, "empty_string"] = df[col].str.strip().eq("").sum()

    summary["total_missing"] = (
        summary["null"] + summary["empty_string"]
    )

    summary["percent"] = (
        summary["total_missing"] / len(df) * 100
    )

    return (
        summary[summary["total_missing"] > 0]
        .sort_values("total_missing", ascending=False)
    )


def missing_by_category(
    df: pd.DataFrame,
    missing_column: str,
    categorical_columns: list[str] | None = None,
    min_missing: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """
    Analyze how missing values are distributed across categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    missing_column : str
        Column whose missing values will be analyzed.
    categorical_columns : list[str], optional
        Categorical columns to inspect. If None, object, string and category
        columns are used automatically.
    min_missing : float, default=0.0
        Only display categories with at least this percentage of missing values.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping each categorical column to its summary table.
    """

    if categorical_columns is None:
        categorical_columns = (
            df.select_dtypes(include=["object", "string", "category"])
              .columns
              .tolist()
        )

    results = {}

    for column in categorical_columns:

        summary = (
            df.groupby(column, dropna=False)
              .agg(
                  total_rows=(column, "size"),
                  missing_rows=(missing_column, lambda s: s.isna().sum())
              )
        )

        summary["missing_percent"] = (
            summary["missing_rows"] / summary["total_rows"] * 100
        ).round(2)

        summary = summary.loc[
            summary["missing_percent"] >= min_missing
        ]

        results[column] = summary

        print(f"\n{'=' * 20} {column} {'=' * 20}")
        display(summary)


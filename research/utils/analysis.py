import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def setup_plot(figsize=(10, 5)):
    plt.figure(figsize=figsize)


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str = "Customers"
):

    setup_plot()

    sns.histplot(
        data=df,
        x=column,
        bins=bins
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)

    plt.show()


def plot_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize=(10, 5),
    marker=None,
    diagonal=False
):
    setup_plot(figsize)

    sns.lineplot(
        data=df,
        x=x,
        y=y,
        marker=marker
    )

    if diagonal:
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            label="Equal Distribution"
        )
        plt.legend()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize=(10, 6),
    alpha=0.3
):

    setup_plot(figsize)

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        alpha=alpha
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)

    plt.show()


def plot_bar(
    df,
    x,
    y,
    title,
    xlabel,
    ylabel,
    hue=None,
    order=None,
    rotation=0,
    figsize=(10, 5)
):
    setup_plot(figsize)

    sns.barplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        order=order
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(rotation=rotation)

    if hue is not None:
        plt.legend(title=hue.capitalize())

    plt.tight_layout()
    plt.show()


def pareto_analysis(
    df: pd.DataFrame,
    value_col: str,
    entity_name: str,
    threshold: float
) -> pd.DataFrame:
    """
    Computes the cumulative distribution required for a Pareto analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose index represents the analyzed entities.
    value_col : str
        Column containing the metric to accumulate.
    entity_name : str
        Name of the analyzed entity (e.g. "visitors", "items", "products").
        Used to name the cumulative percentage column.
    threshold : float
        Threshold in decimal percentage of value column

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by descending value containing cumulative metrics.
    """

    pareto = (
        df[[value_col]]
        .sort_values(value_col, ascending=False)
        .copy()
    )

    pareto[f"cum_{value_col}"] = pareto[value_col].cumsum()

    pareto[f"cum_{value_col}_pct"] = (
        pareto[f"cum_{value_col}"] /
        pareto[value_col].sum()
    )

    pareto[f"cum_{entity_name}_pct"] = (
        range(1, len(pareto) + 1)
    )

    pareto[f"cum_{entity_name}_pct"] /= len(pareto)

    result = pareto.loc[
        pareto[f"cum_{value_col}_pct"] >= threshold
    ].iloc[0]

    print(
        f"{result[f'cum_{entity_name}_pct']:.2%} of {entity_name} "
        f"account for {result[f'cum_{value_col}_pct']:.2%} of all {value_col}."
    )

    return pareto


def build_funnel(df, entity_name):
    funnel = pd.DataFrame(
        {
            "stage": [
                "View",
                "Add to Cart",
                "Transaction"
            ],
            entity_name: [
                df["view"].gt(0).sum(),
                df["addtocart"].gt(0).sum(),
                df["transaction"].gt(0).sum(),
            ]
        }
    )

    return funnel


def compute_conversion_rates(funnel, count_col):
    return pd.DataFrame(
        {
            "transition": [
                "View → Add to Cart",
                "Add to Cart → Transaction",
                "View → Transaction"
            ],
            "conversion_rate": [
                (funnel.iloc[1][count_col] / funnel.iloc[0][count_col])*100,
                (funnel.iloc[2][count_col] / funnel.iloc[1][count_col])*100,
                (funnel.iloc[2][count_col] / funnel.iloc[0][count_col])*100,
            ]
        }
    )


def conversion_summary(
    df: pd.DataFrame,
    value_col: str,
    converted_col: str,
    bins: list,
    labels: list,
    entity_name: str,
    converted_name: str = "transactions",
) -> pd.DataFrame:
    """
    Summarize conversion metrics across value intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    value_col : str
        Numeric column to be binned.
    converted_col : str
        Boolean column indicating whether the entity converted.
    bins : list
        Bin edges.
    labels : list
        Labels corresponding to each interval.
    entity_name : str
        Name of the entity count column (e.g. "sessions", "visitors").
    converted_name : str
        Name of the converted entity column.

    Returns
    -------
    pd.DataFrame
    """

    summary = df.copy()

    summary["bin"] = pd.cut(
        summary[value_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    summary = (
        summary
        .groupby("bin", observed=True)
        .agg(
            **{
                entity_name: ("bin", "size"),
                converted_name: (converted_col, "sum"),
            }
        )
    )

    summary["conversion_rate (%)"] = (
        summary[converted_name]
        / summary[entity_name]
        * 100
    )

    summary["share (%)"] = (
        summary[entity_name]
        / summary[entity_name].sum()
        * 100
    )

    return summary.round(
        {
            "conversion_rate (%)": 2,
            "share (%)": 2,
        }
    )


def build_category_depth(category_tree: pd.DataFrame) -> pd.DataFrame:
    parent_map = category_tree.set_index("categoryid")["parentid"].to_dict()

    def get_category_depth(category_id):
        depth = 0
        current = category_id
        visited = set()

        while parent_map.get(current, -1) != -1:
            if current in visited:
                return None
            visited.add(current)
            current = parent_map[current]
            depth += 1

        return depth

    category_depth = pd.DataFrame(
        {
            "categoryid": category_tree["categoryid"],
            "depth": category_tree["categoryid"].apply(get_category_depth),
        }
    )

    return category_depth

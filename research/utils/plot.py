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

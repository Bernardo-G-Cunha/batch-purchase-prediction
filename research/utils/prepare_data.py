from pathlib import Path

import pandas as pd

RAW_PATH = Path("./research/data/raw/original")
LANDING_PATH = Path("./research/data/raw/landing")

EVENTS_PATH = RAW_PATH / "events.csv"

ITEM_PROPERTIES_1 = RAW_PATH / "item_properties_part1.csv"
ITEM_PROPERTIES_2 = RAW_PATH / "item_properties_part2.csv"
CATEGORY_TREE_PATH = RAW_PATH / "category_tree.csv"

EVENTS_OUTPUT = LANDING_PATH / "events"
ITEM_PROPERTIES_OUTPUT = LANDING_PATH / "item_properties.parquet"
CATEGORY_TREE_OUTPUT = LANDING_PATH / "category_tree.parquet"

EVENTS_OUTPUT.mkdir(parents=True, exist_ok=True)


def convert_events():
    """
    Converts events.csv into monthly Parquet files.
    """

    events = pd.read_csv(EVENTS_PATH)

    events["timestamp"] = pd.to_datetime(
        events["timestamp"],
        unit="ms"
    )

    events["year_month"] = events["timestamp"].dt.strftime("%Y-%m")

    for year_month, df_month in events.groupby("year_month"):

        output_path = EVENTS_OUTPUT / f"{year_month}.parquet"

        (
            df_month
            .drop(columns="year_month")
            .sort_values("timestamp")
            .to_parquet(output_path, index=False)
        )

        print(f"Saved {output_path.name} ({len(df_month):,} rows)")


def convert_item_properties():
    """
    Merges both item_properties files into a single Parquet file.
    """

    item_properties = pd.concat(
        [
            pd.read_csv(ITEM_PROPERTIES_1),
            pd.read_csv(ITEM_PROPERTIES_2)
        ],
        ignore_index=True
    )

    item_properties["timestamp"] = pd.to_datetime(
        item_properties["timestamp"],
        unit="ms"
    )

    item_properties["property"] = item_properties["property"].astype("category")
    
    item_properties["value"] = item_properties["value"].astype("category")

    (
        item_properties
        .sort_values(["itemid", "timestamp"])
        .to_parquet(
            ITEM_PROPERTIES_OUTPUT,
            index=False
        )
    )

    print(
        f"Saved {ITEM_PROPERTIES_OUTPUT.name} "
        f"({len(item_properties):,} rows)"
    )


def convert_category_tree():
    """
    Converts category_tree.csv into Parquet.
    """

    category_tree = pd.read_csv(CATEGORY_TREE_PATH)

    category_tree.to_parquet(
        CATEGORY_TREE_OUTPUT,
        index=False
    )

    print(
        f"Saved {CATEGORY_TREE_OUTPUT.name} "
        f"({len(category_tree):,} rows)"
    )


def main():

    LANDING_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    convert_events()
    convert_item_properties()
    convert_category_tree()

    print("\nDone.")


if __name__ == "__main__":
    main()
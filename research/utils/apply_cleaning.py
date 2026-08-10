from pathlib import Path

import pandas as pd

INPUT_DIR = Path("./research/data/raw/landing/events")
OUTPUT_DIR = Path("./research/data/clean/events")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for input_file in sorted(INPUT_DIR.glob("*.parquet")):
    print(f"Processing {input_file.name}...")

    df = pd.read_parquet(input_file)

    original_rows = len(df)

    df = df.drop_duplicates(ignore_index=True)

    removed_rows = original_rows - len(df)

    output_file = OUTPUT_DIR / input_file.name
    df.to_parquet(output_file, index=False)

    print(
        f"  Rows: {original_rows:,} -> {len(df):,} "
        f"(removed {removed_rows:,} duplicates)"
    )

print("\nFinished.")
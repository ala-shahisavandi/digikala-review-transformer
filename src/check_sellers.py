"""Quick check: seller_code dtype/uniques on the current raw-parquet and clean-parquet."""
from pathlib import Path
import polars as pl

for label, path in [("RAW", Path("data/interim/digikala_comments.parquet")),
                    ("CLEAN", Path("data/interim/digikala_clean.parquet"))]:
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    print(f"\n[{label}] seller_code dtype = {schema['seller_code']}, seller_title dtype = {schema['seller_title']}")
    r = lf.select([
        pl.len().alias("rows"),
        pl.col("seller_code").null_count().alias("seller_code_null"),
        (pl.col("seller_code").cast(pl.Utf8).str.to_lowercase() == "nan").sum().alias("seller_code_literal_nan"),
        pl.col("seller_code").n_unique().alias("seller_code_unique_incl_null"),
        pl.col("seller_title").n_unique().alias("seller_title_unique_incl_null"),
        (pl.col("seller_title").cast(pl.Utf8).str.to_lowercase() == "nan").sum().alias("seller_title_literal_nan"),
    ]).collect(engine="streaming").row(0, named=True)
    for k, v in r.items():
        print(f"  {k}: {v:,}")

lf = pl.scan_parquet("data/interim/digikala_clean.parquet")
top = (lf.group_by("seller_code").len().sort("len", descending=True).head(10)
       .collect(engine="streaming"))
total = lf.select(pl.len()).collect().item()
print("\nTop 10 seller_code by review count (CLEAN):")
for row in top.iter_rows(named=True):
    print(f"  {row['seller_code']}: {row['len']:,} ({row['len'] / total * 100:.2f}%)")

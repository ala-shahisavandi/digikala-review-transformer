"""
Utility functions for Part A: Descriptive Statistics & Data Understanding.
Designed for memory-efficient processing with Polars on large datasets (~7M rows).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import polars as pl
import matplotlib.pyplot as plt

# Project Constants (Strict Compliance with Project Definition)
SEED = 2026
EXPECTED_COLUMNS = [
    "id",
    "title",
    "body",
    "created_at",
    "rate",
    "recommendation_status",
    "is_buyer",
    "product_id",
    "advantages",
    "disadvantages",
    "likes",
    "dislikes",
    "seller_title",
    "seller_code",
    "true_to_size_rate",
]

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
ARTIFACTS_EDA_DIR = PROJECT_ROOT / "artifacts" / "eda"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def ensure_directories() -> None:
    """Ensure all required output directories exist."""
    for d in [DATA_RAW_DIR, DATA_INTERIM_DIR, ARTIFACTS_EDA_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def locate_raw_csv() -> Path:
    """Locate digikala-comments.csv in data/raw/ or data/."""
    candidates = [
        DATA_RAW_DIR / "digikala-comments.csv",
        PROJECT_ROOT / "data" / "digikala-comments.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find 'digikala-comments.csv' in {[str(c) for c in candidates]}"
    )


def save_json_artifact(data: Dict[str, Any], filename: str) -> Path:
    """Save a dictionary as a formatted JSON file in artifacts/eda/."""
    ensure_directories()
    if not filename.endswith(".json"):
        filename += ".json"
    out_path = ARTIFACTS_EDA_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[Artifact Saved] JSON -> {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


def df_to_markdown_table(df: pl.DataFrame) -> str:
    """Convert a Polars DataFrame into a clean GitHub-flavored Markdown table."""
    cols = df.columns
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for row in df.iter_rows():
        formatted_cells = []
        for val in row:
            if isinstance(val, float):
                formatted_cells.append(f"{val:,.4f}".rstrip("0").rstrip("."))
            elif isinstance(val, int):
                formatted_cells.append(f"{val:,}")
            else:
                formatted_cells.append(str(val) if val is not None else "NULL")
        rows.append("| " + " | ".join(formatted_cells) + " |")
    return "\n".join([header, separator] + rows)


def save_table_artifact(
    df: pl.DataFrame,
    base_name: str,
    title: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a Polars summary DataFrame as both Markdown (.md) and JSON (.json)
    in artifacts/eda/ (since *.csv is git-ignored).
    """
    ensure_directories()
    md_path = ARTIFACTS_EDA_DIR / f"{base_name}.md"
    json_path = ARTIFACTS_EDA_DIR / f"{base_name}.json"

    # 1. Save Markdown
    md_content = ""
    if title:
        md_content += f"# {title}\n\n"
    md_content += df_to_markdown_table(df) + "\n"
    if extra_metadata:
        md_content += "\n## Metadata\n"
        for k, v in extra_metadata.items():
            md_content += f"- **{k}**: `{v}`\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 2. Save JSON
    payload = {
        "title": title or base_name,
        "metadata": extra_metadata or {},
        "records": df.to_dicts(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"[Artifact Saved] Table -> {md_path.relative_to(PROJECT_ROOT)} & {json_path.relative_to(PROJECT_ROOT)}"
    )


def deterministic_sample_lazy(
    lf: pl.LazyFrame,
    id_col: str = "id",
    fraction: float = 0.05,
    seed: int = SEED,
) -> pl.LazyFrame:
    """
    Deterministic hash-based sampling on LazyFrame using `id` column and fixed seed.
    Avoids loading the full dataset into memory while guaranteeing 100% reproducibility.
    Rule: hash(id, seed) % 10_000 < int(fraction * 10_000)
    """
    threshold = int(fraction * 10_000)
    return lf.filter(
        (pl.col(id_col).cast(pl.Utf8).hash(seed=seed) % 10_000) < threshold
    )


def setup_plot_style() -> None:
    """Configure matplotlib for high-resolution, clean publication-grade plots with English labels."""
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"

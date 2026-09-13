"""
Data Loader Module for Customer Support on Twitter Dataset.

This module provides functions to:
1. Locate and load the raw dataset (twcs.csv) efficiently.
2. Inspect schema, row counts, missing values, and data types.
3. Compute exact, empirical dataset statistics without fabrication.
4. Provide sample rows for schema understanding and debugging.
"""

import os
import pandas as pd
from typing import Dict, Any, Optional, Tuple


def find_raw_dataset(search_dirs: Optional[list] = None) -> Optional[str]:
    """
    Search for twcs.csv in known cache and data directories.
    
    Args:
        search_dirs: Optional list of directories to look in.
        
    Returns:
        The absolute path to twcs.csv if found, else None.
    """
    if search_dirs is None:
        cache_dir = os.path.expanduser(r"~\.cache\kagglehub\datasets\thoughtvector\customer-support-on-twitter")
        search_dirs = [
            os.path.abspath("data/raw"),
            os.path.abspath("../data/raw"),
            cache_dir,
        ]
        
    for base_dir in search_dirs:
        if os.path.exists(base_dir):
            for root, _, files in os.walk(base_dir):
                for f in files:
                    if f.lower() in ["twcs.csv", "twcs.csv.zip"]:
                        return os.path.join(root, f)
    return None


def get_dataset_schema(file_path: str, nrows: int = 1000) -> pd.DataFrame:
    """
    Read a small sample of the dataset to inspect column names and types.
    
    Args:
        file_path: Path to twcs.csv.
        nrows: Number of rows to sample for schema inspection.
        
    Returns:
        A pandas DataFrame preview.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    return pd.read_csv(file_path, nrows=nrows)


def inspect_full_dataset(file_path: str, chunksize: int = 250000) -> Dict[str, Any]:
    """
    Stream through the complete raw dataset in chunks to compute exact statistics
    without exhausting system RAM.
    
    Args:
        file_path: Path to the raw CSV file.
        chunksize: Number of rows per chunk.
        
    Returns:
        A dictionary containing exact measured counts, null statistics, 
        inbound distribution, date range, and sample data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        
    total_rows = 0
    null_counts: Dict[str, int] = {}
    inbound_counts = {"True": 0, "False": 0, "Other": 0}
    unique_authors = set()
    min_date = None
    max_date = None
    columns = []
    dtypes_map = {}
    
    print(f"Inspecting dataset at: {file_path}")
    print("Streaming through chunks...")
    
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize, low_memory=False)):
        total_rows += len(chunk)
        
        if i == 0:
            columns = list(chunk.columns)
            dtypes_map = {col: str(dtype) for col, dtype in chunk.dtypes.items()}
            for col in columns:
                null_counts[col] = 0
                
        # Count nulls
        for col in columns:
            null_counts[col] += int(chunk[col].isnull().sum())
            
        # Inbound vs Outbound
        inbound_series = chunk["inbound"].astype(str)
        t_count = (inbound_series.str.lower() == "true").sum()
        f_count = (inbound_series.str.lower() == "false").sum()
        inbound_counts["True"] += int(t_count)
        inbound_counts["False"] += int(f_count)
        inbound_counts["Other"] += int(len(chunk) - (t_count + f_count))
        
        # Track unique author IDs (sampling up to 200k to bound memory)
        if len(unique_authors) < 200000:
            unique_authors.update(chunk["author_id"].dropna().unique().tolist())
            
        # Date range
        chunk_dates = chunk["created_at"].dropna()
        if not chunk_dates.empty:
            c_min = chunk_dates.min()
            c_max = chunk_dates.max()
            if min_date is None or c_min < min_date:
                min_date = c_min
            if max_date is None or c_max > max_date:
                max_date = c_max
                
        print(f"  Processed chunk {i+1} ({total_rows:,} rows total)...")
        
    # Read first 5 rows for sample demonstration
    sample_df = pd.read_csv(file_path, nrows=5)
    
    results = {
        "file_path": file_path,
        "file_size_bytes": os.path.getsize(file_path),
        "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
        "total_rows": total_rows,
        "columns": columns,
        "column_dtypes": dtypes_map,
        "null_counts": null_counts,
        "null_percentages": {col: round((null_counts[col] / total_rows) * 100, 2) for col in columns},
        "inbound_distribution": {
            "inbound_customer_messages": inbound_counts["True"],
            "outbound_brand_responses": inbound_counts["False"],
            "other_or_malformed": inbound_counts["Other"],
            "inbound_percentage": round((inbound_counts["True"] / total_rows) * 100, 2),
            "outbound_percentage": round((inbound_counts["False"] / total_rows) * 100, 2),
        },
        "sample_unique_authors_count": len(unique_authors),
        "date_range": {
            "earliest_created_at": str(min_date),
            "latest_created_at": str(max_date),
        },
        "sample_records": sample_df.to_dict(orient="records")
    }
    
    return results


if __name__ == "__main__":
    dataset_file = find_raw_dataset()
    if dataset_file:
        print(f"Found dataset: {dataset_file}")
        stats = inspect_full_dataset(dataset_file)
        print("\n--- MEASURED DATASET STATISTICS ---")
        print(f"Total Rows: {stats['total_rows']:,}")
        print(f"Columns: {stats['columns']}")
        print(f"Nulls: {stats['null_counts']}")
        print(f"Inbound / Outbound: {stats['inbound_distribution']}")
        print(f"Date Range: {stats['date_range']}")
    else:
        print("Dataset not found. Please download twcs.csv into data/raw/")

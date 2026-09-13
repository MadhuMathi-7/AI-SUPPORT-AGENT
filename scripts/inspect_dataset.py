"""
Dataset Inspection Script for Phase 1.
Computes and prints empirical statistics of the Customer Support on Twitter dataset.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import find_raw_dataset, inspect_full_dataset


def run_inspection():
    print("=== PHASE 1: DATASET INSPECTION ===")
    dataset_path = find_raw_dataset()
    
    if not dataset_path:
        print("ERROR: Dataset not found in known paths or cache.")
        sys.exit(1)
        
    print(f"Dataset located at: {dataset_path}")
    print("Computing verified dataset metrics...")
    
    stats = inspect_full_dataset(dataset_path)
    
    # Save stats to data/processed or reports for reproducibility
    os.makedirs("reports", exist_ok=True)
    report_file = os.path.join("reports", "dataset_inspection_phase1.json")
    with open(report_file, "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"\nInspection saved to: {report_file}")
    print("\n--- SUMMARY OF MEASURED RESULTS ---")
    print(f"Total Rows: {stats['total_rows']:,}")
    print(f"File Size: {stats['file_size_mb']} MB")
    print(f"Columns ({len(stats['columns'])}): {stats['columns']}")
    print("\nMissing Values:")
    for col, count in stats['null_counts'].items():
        pct = stats['null_percentages'][col]
        print(f"  - {col}: {count:,} ({pct}%)")
        
    print("\nInbound vs Outbound Distribution:")
    for k, v in stats['inbound_distribution'].items():
        print(f"  - {k}: {v:,}" if isinstance(v, int) else f"  - {k}: {v}%")
        
    print(f"\nDate Range: {stats['date_range']['earliest_created_at']} to {stats['date_range']['latest_created_at']}")
    print(f"Sample Authors Count: {stats['sample_unique_authors_count']:,}")


if __name__ == "__main__":
    run_inspection()

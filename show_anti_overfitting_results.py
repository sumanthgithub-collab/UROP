import json
import os
import sys
import io
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
ANTI_OVR_DIR = RESULTS_DIR / "anti_overfitting"

def main():
    table1_file = ANTI_OVR_DIR / "final_performance_table.csv"
    table2_file = ANTI_OVR_DIR / "baseline_vs_new_comparison.csv"
    
    if table1_file.exists() and table2_file.exists():
        df1 = pd.read_csv(table1_file)
        df2 = pd.read_csv(table2_file)
        
        print("==========================================================================", flush=True)
        print("ANTI-OVERFITTING FINAL RESULTS SUMMARY", flush=True)
        print("==========================================================================", flush=True)
        print("\nTABLE 1: ANTI-OVERFITTING FINAL PERFORMANCE METRICS", flush=True)
        print(df1.to_string(index=False), flush=True)
        
        print("\n" + "="*110, flush=True)
        print("TABLE 2: BASELINE VS ANTI-OVERFITTING COMPARISON (BEFORE vs AFTER)", flush=True)
        print("="*110, flush=True)
        print(df2.to_string(index=False), flush=True)
        
    # Run graph and terminal report script
    from training.generate_anti_overfitting_graphs import generate_anti_overfitting_visualizations_and_report
    generate_anti_overfitting_visualizations_and_report()

if __name__ == "__main__":
    main()

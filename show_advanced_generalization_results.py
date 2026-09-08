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
ADVANCED_DIR = RESULTS_DIR / "advanced_generalization"
METRICS_DIR = ADVANCED_DIR / "metrics"

def main():
    table1_file = METRICS_DIR / "final_performance_table.csv"
    table2_file = METRICS_DIR / "baseline_vs_current_vs_new_comparison.csv"
    
    if table1_file.exists() and table2_file.exists():
        df1 = pd.read_csv(table1_file)
        df2 = pd.read_csv(table2_file)
        
        print("==========================================================================", flush=True)
        print("STAGE 2 ADVANCED GENERALIZATION FINAL RESULTS SUMMARY", flush=True)
        print("==========================================================================", flush=True)
        print("\nTABLE 1: STAGE 2 ADVANCED GENERALIZATION FINAL PERFORMANCE METRICS", flush=True)
        print(df1.to_string(index=False), flush=True)
        
        print("\n" + "="*115, flush=True)
        print("TABLE 2: MULTI-STAGE PROGRESSION COMPARISON (BASELINE -> STAGE 1 -> STAGE 2)", flush=True)
        print("="*115, flush=True)
        print(df2.to_string(index=False), flush=True)
        
    from training.generate_advanced_generalization_graphs import generate_graphs_and_report
    generate_graphs_and_report()

if __name__ == "__main__":
    main()

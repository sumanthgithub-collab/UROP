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

# Multimodal-Sentiment-Analysis project root
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
RESULTS_DIR = BASE_DIR / "results"
STAGE3_DIR = RESULTS_DIR / "stage3_unseen_breakthrough"
METRICS_DIR = STAGE3_DIR / "metrics"

def main():
    table1_file = METRICS_DIR / "final_performance_table.csv"
    
    if table1_file.exists():
        df1 = pd.read_csv(table1_file)
        
        print("==========================================================================", flush=True)
        print("STAGE 3 UNSEEN DATA ACCURACY BREAKTHROUGH SUMMARY", flush=True)
        print("==========================================================================", flush=True)
        print("\nTABLE 1: STAGE 3 UNSEEN DATA FINAL PERFORMANCE METRICS (520 TOUCH-FREE SAMPLES)", flush=True)
        print(df1.to_string(index=False), flush=True)
        
    from training.generate_stage3_graphs import generate_stage3_visualizations
    generate_stage3_visualizations()

if __name__ == "__main__":
    main()

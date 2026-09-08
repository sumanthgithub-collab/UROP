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
METRICS_DIR = BASE_DIR / "results" / "sota_breakthrough" / "metrics"
SUMMARY_FILE = METRICS_DIR / "sota_performance_summary.csv"

def main():
    print("==========================================================================", flush=True)
    print("SOTA ULTRA-GENERALIZATION BREAKTHROUGH PERFORMANCE BENCHMARK", flush=True)
    print("==========================================================================", flush=True)
    
    if SUMMARY_FILE.exists():
        df = pd.read_csv(SUMMARY_FILE)
        print("\nTABLE: SOTA MULTIMODAL PERFORMANCE SUMMARY (520 TOUCH-FREE TEST SAMPLES)", flush=True)
        print(df.to_string(index=False), flush=True)
    else:
        print(f"Summary file not found at {SUMMARY_FILE}. Please run training/train_sota_ultra_generalization.py first.")

if __name__ == "__main__":
    main()

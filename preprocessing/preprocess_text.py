from pathlib import Path
import pandas as pd
import re
import html

BASE_DIR = Path(__file__).resolve().parent.parent
BALANCED_DIR = BASE_DIR / "dataset" / "balanced"
TEXT_DIR = BALANCED_DIR / "texts"
SPLITS_DIR = BASE_DIR / "dataset" / "splits"
OUTPUT_DIR = BASE_DIR / "dataset" / "preprocessed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_splits():
    for split_name in ["train", "validation", "test"]:
        input_file = SPLITS_DIR / f"{split_name}.csv"
        if not input_file.exists():
            continue
        print(f"Processing {split_name}...")
        df = pd.read_csv(input_file)
        cleaned_texts = []
        for _, row in df.iterrows():
            text_file = TEXT_DIR / row["text"]
            if text_file.exists():
                with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                    original_text = f.read()
            else:
                original_text = str(row.get("text", ""))
            cleaned_texts.append(clean_text(original_text))
        df["cleaned_text"] = cleaned_texts
        output_file = OUTPUT_DIR / f"{split_name}_cleaned.csv"
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"{split_name}: {len(df)} samples processed")

if __name__ == "__main__":
    preprocess_splits()

from pathlib import Path
from collections import Counter
import csv
import random
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
LABEL_FILE = BASE_DIR / "dataset" / "raw" / "labelResultAll.txt"
DATA_DIR = BASE_DIR / "dataset" / "raw" / "data"

OUTPUT_DIR = BASE_DIR / "dataset" / "balanced"
IMAGE_DIR = OUTPUT_DIR / "images"
TEXT_DIR = OUTPUT_DIR / "texts"

SEED = 42
SAMPLES_PER_CLASS = 1299

def majority(labels):
    counts = Counter(labels)
    for label in ["positive", "negative", "neutral"]:
        if counts[label] >= 2:
            return label
    return "unresolved"

def get_multimodal_label(annotations):
    text_labels = [x[0] for x in annotations]
    image_labels = [x[1] for x in annotations]
    text_final = majority(text_labels)
    image_final = majority(image_labels)
    if text_final == "unresolved" or image_final == "unresolved":
        return "unresolved"
    if text_final == image_final:
        return text_final
    if text_final == "neutral":
        return image_final
    if image_final == "neutral":
        return text_final
    return "conflict"

def main():
    if not LABEL_FILE.exists():
        print(f"Raw label file not found at {LABEL_FILE}. Skipping generation if balanced dataset already exists.")
        return
    samples = []
    with open(LABEL_FILE, "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            sample_id = parts[0]
            annotation_values = parts[1:]
            if len(annotation_values) < 3:
                continue
            annotations = []
            for value in annotation_values[:3]:
                if "," not in value:
                    continue
                text_sentiment, image_sentiment = value.split(",", 1)
                annotations.append((text_sentiment.strip().lower(), image_sentiment.strip().lower()))
            if len(annotations) != 3:
                continue
            final_label = get_multimodal_label(annotations)
            if final_label not in ["positive", "negative"]:
                continue
            image_file = DATA_DIR / f"{sample_id}.jpg"
            text_file = DATA_DIR / f"{sample_id}.txt"
            if not image_file.exists() or not text_file.exists():
                continue
            samples.append({"id": sample_id, "label": final_label, "image": image_file, "text": text_file})

    positive_samples = [x for x in samples if x["label"] == "positive"]
    negative_samples = [x for x in samples if x["label"] == "negative"]

    random.seed(SEED)
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)

    selected_samples = positive_samples[:SAMPLES_PER_CLASS] + negative_samples[:SAMPLES_PER_CLASS]
    random.shuffle(selected_samples)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_file = OUTPUT_DIR / "metadata.csv"
    with open(metadata_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "image", "text", "label"])
        for sample in selected_samples:
            sample_id = sample["id"]
            label = sample["label"]
            source_image = sample["image"]
            source_text = sample["text"]
            shutil.copy2(source_image, IMAGE_DIR / source_image.name)
            shutil.copy2(source_text, TEXT_DIR / source_text.name)
            writer.writerow([sample_id, source_image.name, source_text.name, label])

    print(f"Balanced dataset created at {OUTPUT_DIR} with {len(selected_samples)} total samples.")

if __name__ == "__main__":
    main()

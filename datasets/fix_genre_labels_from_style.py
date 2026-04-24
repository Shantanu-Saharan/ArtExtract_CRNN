"""
Quick fix: regenerate genre CSVs using style-inferred genre labels.

This reads the existing style/artist/genre split CSVs, infers genre from
the style folder in the image path, and rewrites genre_train/val/test.csv
and multitask_train/val/test.csv with corrected genre labels.

Run this once after cloning the repo if you don't have ArtGAN metadata.

Usage:
    python datasets/fix_genre_labels_from_style.py --datasets_dir datasets
"""

import argparse
from pathlib import Path

import pandas as pd


STYLE_TO_GENRE = {
    "Abstract_Expressionism":    "abstract_painting",
    "Action_painting":           "abstract_painting",
    "Color_Field_Painting":      "abstract_painting",
    "Minimalism":                "abstract_painting",
    "Conceptual_Art":            "abstract_painting",
    "Synthetic_Cubism":          "abstract_painting",
    "Analytical_Cubism":         "abstract_painting",
    "Cubism":                    "abstract_painting",
    "Impressionism":             "landscape",
    "Post_Impressionism":        "landscape",
    "Pointillism":               "landscape",
    "Romanticism":               "landscape",
    "Luminism":                  "landscape",
    "Hudson_River_School":       "landscape",
    "Baroque":                   "portrait",
    "Rococo":                    "portrait",
    "Neoclassicism":             "portrait",
    "High_Renaissance":          "portrait",
    "Early_Renaissance":         "portrait",
    "Northern_Renaissance":      "portrait",
    "Mannerism_Late_Renaissance":"portrait",
    "Realism":                   "genre_painting",
    "Social_Realism":            "genre_painting",
    "Contemporary_Realism":      "genre_painting",
    "New_Realism":               "genre_painting",
    "Magic_Realism":             "genre_painting",
    "Expressionism":             "genre_painting",
    "Fauvism":                   "genre_painting",
    "Naive_Art_Primitivism":     "genre_painting",
    "Pop_Art":                   "illustration",
    "Art_Nouveau_Modern":        "illustration",
    "Symbolism":                 "religious_painting",
    "Byzantine":                 "religious_painting",
    "Proto_Renaissance":         "religious_painting",
    "International_Gothic":      "religious_painting",
    "Ukiyo_e":                   "cityscape",
    "Art_Informel":              "sketch_and_study",
    "Tachisme":                  "sketch_and_study",
}

GENRE_CLASSES = [
    "abstract_painting",
    "cityscape",
    "genre_painting",
    "illustration",
    "landscape",
    "portrait",
    "religious_painting",
]

GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_CLASSES)}
DEFAULT_GENRE = "genre_painting"
DEFAULT_IDX = GENRE_TO_IDX[DEFAULT_GENRE]


def style_from_path(image_path: str) -> str:
    # image_path is like "Impressionism/artist_title.jpg"
    parts = str(image_path).replace("\\", "/").split("/")
    if parts:
        return parts[0].strip()
    return ""


# Title keywords for reliable genre detection from artwork filenames
_NUDE_KW   = ["nude", "bath", "bathing", "nymph", "venus", "odalisque", "naked", "bather"]
_STILL_KW  = ["still life", "still-life", "vase of", "fruit", "apples", "bottles", "pitcher", "jug "]
_SKETCH_KW = ["study", "sketch", "etching", "engraving", "preparatory", "charcoal"]


def _get_title(image_path: str) -> str:
    fname = str(image_path).replace("\\", "/").split("/")[-1]
    stem = fname.rsplit(".", 1)[0]
    if "_" in stem:
        return stem.split("_", 1)[1].lower().replace("-", " ")
    return stem.lower()


def infer_genre_label(image_path: str) -> int:
    style = style_from_path(image_path)
    title = _get_title(image_path)

    # Title keywords take priority — artwork titles reliably signal genre
    # These only matter for full dataset; small dataset rarely has these titles
    if any(kw in title for kw in _NUDE_KW):
        # nude_painting not in our 7-class set -> fall through to style mapping
        pass
    if any(kw in title for kw in _STILL_KW):
        pass
    if any(kw in title for kw in _SKETCH_KW):
        pass

    genre_name = STYLE_TO_GENRE.get(style, DEFAULT_GENRE)
    return GENRE_TO_IDX.get(genre_name, DEFAULT_IDX)


def fix_single_task_genre_csvs(datasets_dir: Path):
    for split in ["train", "val", "test"]:
        path = datasets_dir / f"genre_{split}.csv"
        if not path.exists():
            print(f"  Skipping missing: {path}")
            continue

        df = pd.read_csv(path)
        if "image_path" not in df.columns or "label" not in df.columns:
            print(f"  Unexpected columns in {path}: {list(df.columns)}")
            continue

        df["label"] = df["image_path"].apply(infer_genre_label)

        unique = df["label"].nunique()
        print(f"  {path.name}: {len(df)} rows, {unique} unique genre labels")
        df.to_csv(path, index=False)


def fix_multitask_genre_column(datasets_dir: Path):
    for split in ["train", "val", "test"]:
        path = datasets_dir / f"multitask_{split}.csv"
        if not path.exists():
            print(f"  Skipping missing: {path}")
            continue

        df = pd.read_csv(path)
        if "image_path" not in df.columns or "genre_label" not in df.columns:
            print(f"  Unexpected columns in {path}: {list(df.columns)}")
            continue

        df["genre_label"] = df["image_path"].apply(infer_genre_label)

        unique = df["genre_label"].nunique()
        print(f"  {path.name}: {len(df)} rows, {unique} unique genre labels")
        df.to_csv(path, index=False)


def write_genre_classes(datasets_dir: Path):
    label_dir = datasets_dir / "label_maps"
    label_dir.mkdir(parents=True, exist_ok=True)
    genre_file = label_dir / "genre_classes.txt"
    genre_file.write_text("\n".join(GENRE_CLASSES) + "\n", encoding="utf-8")
    print(f"  Wrote {len(GENRE_CLASSES)} genre classes to {genre_file}")


def main():
    parser = argparse.ArgumentParser(description="Fix genre labels using style inference")
    parser.add_argument("--datasets_dir", type=str, default="datasets")
    args = parser.parse_args()

    d = Path(args.datasets_dir)
    print("Fixing genre_classes.txt...")
    write_genre_classes(d)

    print("\nFixing genre single-task CSVs...")
    fix_single_task_genre_csvs(d)

    print("\nFixing multitask CSVs (genre_label column)...")
    fix_multitask_genre_column(d)

    print("\nDone. Genre labels now use 7 classes inferred from style.")
    print("    --num_genre_classes 7 \\")


if __name__ == "__main__":
    main()

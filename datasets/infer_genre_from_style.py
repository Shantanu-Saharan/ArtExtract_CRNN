"""
Infer genre labels from WikiArt style folder names.

This is a fallback approach when ArtGAN genre metadata is not available.
Each WikiArt style is mapped to its dominant genre based on art-historical knowledge.

The mapping is imperfect (e.g. Realism spans landscape, portrait, genre_painting)
but produces ~10 real genre classes instead of 1 "unknown" class.

Usage:
    python datasets/infer_genre_from_style.py \
        --master_csv datasets/wikiart_master.csv \
        --output_csv datasets/wikiart_master_with_genre.csv
"""

import argparse
from pathlib import Path

import pandas as pd


# Maps WikiArt style → dominant genre
# Based on: https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md
# and standard art history references.
STYLE_TO_GENRE = {
    # Abstract styles
    "Abstract_Expressionism":    "abstract_painting",
    "Action_painting":           "abstract_painting",
    "Color_Field_Painting":      "abstract_painting",
    "Minimalism":                "abstract_painting",
    "Conceptual_Art":            "abstract_painting",
    "Synthetic_Cubism":          "abstract_painting",
    "Analytical_Cubism":         "abstract_painting",
    "Cubism":                    "abstract_painting",

    # Landscape-dominant styles
    "Impressionism":             "landscape",
    "Post_Impressionism":        "landscape",
    "Pointillism":               "landscape",
    "Romanticism":               "landscape",
    "Luminism":                  "landscape",
    "Hudson_River_School":       "landscape",

    # Portrait-heavy styles
    "Baroque":                   "portrait",
    "Rococo":                    "portrait",
    "Neoclassicism":             "portrait",
    "High_Renaissance":          "portrait",
    "Early_Renaissance":         "portrait",
    "Northern_Renaissance":      "portrait",
    "Mannerism_Late_Renaissance":"portrait",

    # Genre painting
    "Realism":                   "genre_painting",
    "Social_Realism":            "genre_painting",
    "Contemporary_Realism":      "genre_painting",
    "New_Realism":               "genre_painting",
    "Magic_Realism":             "genre_painting",
    "Expressionism":             "genre_painting",
    "Fauvism":                   "genre_painting",
    "Naive_Art_Primitivism":     "genre_painting",

    # Illustration
    "Pop_Art":                   "illustration",
    "Art_Nouveau_Modern":        "illustration",

    # Religious
    "Symbolism":                 "religious_painting",
    "Byzantine":                 "religious_painting",
    "Proto_Renaissance":         "religious_painting",
    "International_Gothic":      "religious_painting",

    # Cityscape
    "Ukiyo_e":                   "cityscape",
}

DEFAULT_GENRE = "genre_painting"


def infer_genre(style: str) -> str:
    style = str(style).strip()
    if style in STYLE_TO_GENRE:
        return STYLE_TO_GENRE[style]
    # try with underscores normalized
    normalized = style.replace(" ", "_").replace("-", "_")
    if normalized in STYLE_TO_GENRE:
        return STYLE_TO_GENRE[normalized]
    return DEFAULT_GENRE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, default="datasets/wikiart_master.csv")
    parser.add_argument("--output_csv", type=str, default="datasets/wikiart_master_with_genre.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.master_csv)
    print(f"Loaded {len(df)} rows from {args.master_csv}")

    original_unknown = (df["genre"] == "unknown").sum()
    print(f"  Rows with genre='unknown': {original_unknown}")

    # fill unknown genres using style inference
    mask = df["genre"].isna() | (df["genre"].astype(str).str.strip().str.lower() == "unknown")
    df.loc[mask, "genre"] = df.loc[mask, "style"].apply(infer_genre)

    still_unknown = (df["genre"] == "unknown").sum()
    print(f"  Rows still 'unknown' after inference: {still_unknown}")

    genre_counts = df["genre"].value_counts()
    print(f"\nGenre distribution ({len(genre_counts)} classes):")
    for name, count in genre_counts.items():
        print(f"  {name}: {count}")

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved to: {out.resolve()}")
    print("\nNext step:")
    print("  python datasets/prepare_wikiart.py \\")
    print(f"    --master_csv {args.output_csv} \\")
    print("    --output_dir datasets")


if __name__ == "__main__":
    main()

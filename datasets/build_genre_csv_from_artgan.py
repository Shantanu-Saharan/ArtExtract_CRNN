"""
Convert ArtGAN genre label files into a genre metadata CSV.

ArtGAN genre CSVs use format: <relative_path> <genre_index>
This script builds: image_path,genre  (using genre class names)

Usage:
    python datasets/build_genre_csv_from_artgan.py \
        --artgan_dir datasets/artgan_meta \
        --output datasets/artgan_genre_metadata.csv
"""

import argparse
import csv
from pathlib import Path


# The 10 WikiArt genre classes from ArtGAN (in index order)
GENRE_CLASSES = [
    "abstract_painting",
    "cityscape",
    "genre_painting",
    "illustration",
    "landscape",
    "nude_painting",
    "portrait",
    "religious_painting",
    "sketch_and_study",
    "still_life",
]


def load_artgan_csv(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_path = parts[0].strip()
            try:
                genre_idx = int(parts[-1])
            except ValueError:
                continue
            rows.append((img_path, genre_idx))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artgan_dir", type=str, default="datasets/artgan_meta")
    parser.add_argument("--output", type=str, default="datasets/artgan_genre_metadata.csv")
    args = parser.parse_args()

    artgan_dir = Path(args.artgan_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for split in ["train", "val", "test"]:
        p = artgan_dir / f"genre_{split}.csv"
        if p.exists():
            rows = load_artgan_csv(p)
            all_rows.extend(rows)
            print(f"  {split}: {len(rows)} rows")
        else:
            print(f"  Missing: {p}")

    if not all_rows:
        print("No data found. Run download_genre_metadata.py first.")
        return

    print(f"\nTotal rows: {len(all_rows)}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "genre"])
        for img_path, genre_idx in all_rows:
            if 0 <= genre_idx < len(GENRE_CLASSES):
                genre_name = GENRE_CLASSES[genre_idx]
            else:
                genre_name = "unknown"
            writer.writerow([img_path, genre_name])

    print(f"Saved genre metadata to: {out_path.resolve()}")
    print("\nNext step:")
    print("  python datasets/build_wikiart_master.py \\")
    print("    --dataset_root <path/to/wikiart> \\")
    print("    --genre_metadata datasets/artgan_genre_metadata.csv \\")
    print("    --output_csv datasets/wikiart_master.csv")


if __name__ == "__main__":
    main()

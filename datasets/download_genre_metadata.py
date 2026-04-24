"""
Download genre annotation files from ArtGAN WikiArt repository.

The ArtGAN repository provides genre labels for WikiArt images at:
https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset

This script downloads:
- wikiart_genre_class.txt  (list of genre class names)
- genre_class.txt          (mapping of genre index to name)
- The train/val/test genre label CSVs

Usage:
    python datasets/download_genre_metadata.py --output_dir datasets/artgan_meta
"""

import argparse
import os
import urllib.request
from pathlib import Path


ARTGAN_BASE = "https://raw.githubusercontent.com/cs-chan/ArtGAN/master/WikiArt%20Dataset"

FILES_TO_DOWNLOAD = {
    "genre_class.txt": f"{ARTGAN_BASE}/Genre/genre_class.txt",
    "genre_train.csv": f"{ARTGAN_BASE}/Genre/genre_train.csv",
    "genre_val.csv": f"{ARTGAN_BASE}/Genre/genre_val.csv",
    "genre_test.csv": f"{ARTGAN_BASE}/Genre/genre_test.csv",
}


def download_file(url: str, dest: Path) -> bool:
    try:
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download ArtGAN genre metadata")
    parser.add_argument("--output_dir", type=str, default="datasets/artgan_meta")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ArtGAN genre metadata to: {out.resolve()}")
    ok = 0
    for name, url in FILES_TO_DOWNLOAD.items():
        dest = out / name
        if dest.exists():
            print(f"  Already exists: {dest}")
            ok += 1
        else:
            if download_file(url, dest):
                ok += 1

    print(f"\nDownloaded {ok}/{len(FILES_TO_DOWNLOAD)} files.")
    if ok < len(FILES_TO_DOWNLOAD):
        print("Some files failed. You may need to download them manually from:")
        print(f"  {ARTGAN_BASE}")
    else:
        print("\nNext step:")
        print("  python datasets/build_genre_csv_from_artgan.py \\")
        print("    --artgan_dir datasets/artgan_meta \\")
        print("    --output datasets/artgan_genre_metadata.csv")


if __name__ == "__main__":
    main()

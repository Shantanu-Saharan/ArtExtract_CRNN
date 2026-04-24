# scan a local WikiArt tree and write wikiart_master.csv
# expects <root>/<Style>/<images> or <root>/<Style>/<Artist>/<images>

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import pandas as pd

IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


# skip hidden dirs and yield image files
def _iter_images(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and d not in ("__pycache__", "venv", "node_modules")
        ]
        base = Path(dirpath)
        for name in filenames:
            if name.startswith("."):
                continue
            p = base / name
            if _is_image(p):
                yield p


def _slug_to_display(slug: str) -> str:
    # 'vincent-van-gogh' -> 'Vincent Van Gogh'
    s = slug.replace("_", "-")
    parts = [w.strip() for w in s.split("-") if w.strip()]
    if not parts:
        return "unknown"
    return " ".join(w.title() for w in parts)


def _artist_from_filename(stem: str) -> str:
    if "_" not in stem:
        return "unknown"
    slug = stem.split("_", 1)[0].strip()
    if not slug:
        return "unknown"
    return _slug_to_display(slug)


def _infer_style_and_artist(
    rel_parts: tuple[str, ...],
) -> tuple[str, str]:
    if len(rel_parts) == 2:
        style, fname = rel_parts[0], rel_parts[1]
        stem = Path(fname).stem
        artist = _artist_from_filename(stem)
        return style, artist
    if len(rel_parts) >= 3:
        style = rel_parts[0]
        artist_dir = rel_parts[1]
        if artist_dir:
            artist = _slug_to_display(artist_dir.replace("_", "-"))
        else:
            artist = "unknown"
        return style, artist
    return "unknown", "unknown"


def _load_genre_map(metadata_path: Optional[Path]) -> Dict[str, str]:
    if not metadata_path:
        return {}
    if not metadata_path.is_file():
        print(f"Warning: metadata file not found: {metadata_path}", file=sys.stderr)
        return {}

    df = pd.read_csv(metadata_path)
    cols = {c.lower(): c for c in df.columns}
    key_col = None
    for candidate in ("image_path", "path", "filename", "file", "rel_path"):
        if candidate in cols:
            key_col = cols[candidate]
            break
    if key_col is None:
        print(
            "Warning: metadata CSV needs a path column "
            "(image_path, path, filename, file, or rel_path). Ignoring metadata.",
            file=sys.stderr,
        )
        return {}

    genre_col = None
    for candidate in ("genre", "Genre"):
        if candidate in df.columns:
            genre_col = candidate
            break
    if genre_col is None:
        for c in df.columns:
            if c.lower() == "genre":
                genre_col = c
                break
    if genre_col is None:
        print("Warning: metadata CSV has no 'genre' column. Ignoring metadata.", file=sys.stderr)
        return {}

    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        key = str(row[key_col]).strip()
        g = row[genre_col]
        if pd.isna(g):
            continue
        g_str = str(g).strip()
        if not g_str:
            continue
        norm = key.replace("\\", "/")
        out[norm] = g_str
        out[os.path.basename(norm)] = g_str
    return out


def _lookup_genre(rel_posix: str, genre_map: Dict[str, str]) -> str:
    if not genre_map:
        return "unknown"
    if rel_posix in genre_map:
        return genre_map[rel_posix]
    base = Path(rel_posix).name
    if base in genre_map:
        return genre_map[base]
    return "unknown"


def build_master(
    dataset_root: Path,
    output_csv: Path,
    genre_metadata: Optional[Path] = None,
    limit: Optional[int] = None,
    log_every: int = 10_000,
) -> int:
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        print(f"Error: dataset root is not a directory: {dataset_root}", file=sys.stderr)
        return 1

    genre_map = _load_genre_map(genre_metadata)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning images under: {dataset_root}")
    if genre_map:
        print("Loaded optional genre metadata (CSV).")
    else:
        print("Genre column will be 'unknown' unless matched from optional metadata.")

    count = 0
    skipped_shallow = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "artist", "style", "genre"])

        for img_path in _iter_images(dataset_root):
            try:
                rel = img_path.relative_to(dataset_root)
            except ValueError:
                print(f"Warning: skipping path outside root: {img_path}", file=sys.stderr)
                continue

            parts = tuple(rel.as_posix().split("/"))
            if len(parts) < 2:
                skipped_shallow += 1
                continue

            style, artist = _infer_style_and_artist(parts)
            rel_posix = rel.as_posix()
            genre = _lookup_genre(rel_posix, genre_map)

            writer.writerow([rel_posix, artist, style, genre])
            count += 1
            if log_every > 0 and count % log_every == 0:
                print(f"  ... indexed {count} images")
            if limit is not None and count >= limit:
                break

    if skipped_shallow:
        print(f"Skipped {skipped_shallow} file(s) not under a style subfolder (need <root>/<Style>/... ).")

    print(f"Wrote {count} rows to {output_csv.resolve()}")
    print(f"Dataset root (resolved): {dataset_root}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build wikiart_master.csv from a local WikiArt-style image tree."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="datasets/wikiart_master.csv",
    )
    parser.add_argument(
        "--genre_metadata",
        type=str,
        default="",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10_000,
    )
    args = parser.parse_args()

    meta = Path(args.genre_metadata) if args.genre_metadata else None
    lim = args.limit if args.limit and args.limit > 0 else None

    code = build_master(
        dataset_root=Path(args.dataset_root),
        output_csv=Path(args.output_csv),
        genre_metadata=meta,
        limit=lim,
        log_every=args.log_every,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()

# build train/val/test CSVs from wikiart_master.csv
# writes the split CSVs and label_maps/*.txt

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def _parse_ratios(train_r: float, val_r: float, test_r: float) -> None:
    s = train_r + val_r + test_r
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must be 1.0, got {s}")


def _print_class_counts(df: pd.DataFrame, label: str) -> None:
    # quick count summary for one label column
    counts = df.groupby(label).size().sort_values(ascending=False)
    n = len(counts)
    head = counts.head(15)
    print(f"  {label}: {n} classes, {len(df)} images")
    for name, c in head.items():
        print(f"    - {name!r}: {int(c)}")
    if n > 15:
        print(f"    ... ({n - 15} more classes)")


def _filter_column(
    df: pd.DataFrame,
    col: str,
    top_k: Optional[int],
    min_samples: Optional[int],
) -> pd.DataFrame:
    if df.empty:
        return df

    counts = df[col].value_counts()
    classes = counts.index.tolist()

    if min_samples and min_samples > 1:
        classes = [c for c in classes if counts[c] >= min_samples]

    if top_k and top_k > 0:
        classes = sorted(classes, key=lambda c: counts[c], reverse=True)[:top_k]

    keep = set(classes)
    out = df[df[col].isin(keep)].copy()
    return out


def _build_label_map(classes: List[str]) -> Dict[str, int]:
    sorted_classes = sorted(classes)
    return {name: i for i, name in enumerate(sorted_classes)}


def _write_label_file(path: Path, classes: List[str]) -> None:
    sorted_classes = sorted(classes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for name in sorted_classes:
            f.write(name + "\n")


def _stratified_split(
    df: pd.DataFrame,
    label_col: str,
    train_r: float,
    val_r: float,
    test_r: float,
    seed: int,
    stratify: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    test_size = test_r
    val_size_of_tv = val_r / (train_r + val_r) if (train_r + val_r) > 0 else 0.0

    strat = df[label_col] if stratify else None
    try:
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=strat,
        )
        strat2 = train_val[label_col] if stratify else None
        train, val = train_test_split(
            train_val,
            test_size=val_size_of_tv,
            random_state=seed,
            stratify=strat2,
        )
    except ValueError as e:
        print(
            f"Warning: stratified split failed ({e}). Falling back to random split.",
            file=sys.stderr,
        )
        train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
        train, val = train_test_split(
            train_val,
            test_size=val_size_of_tv,
            random_state=seed,
        )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _single_task_frames(
    df: pd.DataFrame,
    label_col: str,
    label_map: Dict[str, int],
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "image_path": df["image_path"],
            "label": df[label_col].map(label_map),
        }
    )
    bad = out["label"].isna()
    if bad.any():
        print(f"Warning: dropping {bad.sum()} rows with unknown labels for {label_col}", file=sys.stderr)
        out = out[~bad].reset_index(drop=True)
    out["label"] = out["label"].astype(int)
    return out


def _multitask_frame(
    df: pd.DataFrame,
    maps: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    a_map, s_map, g_map = maps["artist"], maps["style"], maps["genre"]
    out = pd.DataFrame(
        {
            "image_path": df["image_path"],
            "artist_label": df["artist"].map(a_map),
            "style_label": df["style"].map(s_map),
            "genre_label": df["genre"].map(g_map),
        }
    )
    bad = out[["artist_label", "style_label", "genre_label"]].isna().any(axis=1)
    if bad.any():
        print(f"Warning: dropping {bad.sum()} multitask rows with unmapped labels", file=sys.stderr)
        out = out[~bad].reset_index(drop=True)
    out["artist_label"] = out["artist_label"].astype(int)
    out["style_label"] = out["style_label"].astype(int)
    out["genre_label"] = out["genre_label"].astype(int)
    return out


def prepare(
    master_csv: Path,
    output_dir: Path,
    train_r: float,
    val_r: float,
    test_r: float,
    seed: int,
    stratify: bool,
    stratify_column: str,
    top_styles: Optional[int],
    top_artists: Optional[int],
    top_genres: Optional[int],
    min_samples_style: Optional[int],
    min_samples_artist: Optional[int],
    min_samples_genre: Optional[int],
) -> int:
    _parse_ratios(train_r, val_r, test_r)

    if not master_csv.is_file():
        print(f"Error: master CSV not found: {master_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(master_csv)
    required = {"image_path", "artist", "style", "genre"}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: master CSV missing columns: {missing}", file=sys.stderr)
        return 1

    df = df.dropna(subset=["image_path", "style"]).copy()
    df["artist"] = df["artist"].fillna("unknown").astype(str)
    df["style"] = df["style"].astype(str)
    df["genre"] = df["genre"].fillna("unknown").astype(str)

    # Warn if genre column is mostly unknown — likely means genre metadata was not provided
    genre_unknown_ratio = (df["genre"].astype(str).str.strip().str.lower() == "unknown").mean()
    if genre_unknown_ratio > 0.5:
        print(
            f"\nWARNING: {genre_unknown_ratio:.1%} of rows have genre='unknown'.\n"
            "This means genre_classes.txt will have only 1 class and genre classification\n"
            "will be meaningless.\n"
            "To fix this, run one of:\n"
            "  (a) python datasets/download_genre_metadata.py\n"
            "      python datasets/build_genre_csv_from_artgan.py\n"
            "      python datasets/build_wikiart_master.py --genre_metadata datasets/artgan_genre_metadata.csv ...\n"
            "  (b) python datasets/infer_genre_from_style.py --master_csv datasets/wikiart_master.csv\n"
            "      python datasets/prepare_wikiart.py --master_csv datasets/wikiart_master_with_genre.csv\n"
        )

    print(f"Loaded {len(df)} rows from {master_csv.resolve()}")
    print(f"  rows={len(df)} | styles={df['style'].nunique()} | artists={df['artist'].nunique()} | genres={df['genre'].nunique()}")

    df = _filter_column(df, "style", top_styles, min_samples_style)
    print("After style filter:")
    print(f"  rows={len(df)} | styles={df['style'].nunique()} | artists={df['artist'].nunique()} | genres={df['genre'].nunique()}")

    df = _filter_column(df, "artist", top_artists, min_samples_artist)
    print("After artist filter:")
    print(f"  rows={len(df)} | styles={df['style'].nunique()} | artists={df['artist'].nunique()} | genres={df['genre'].nunique()}")

    df = _filter_column(df, "genre", top_genres, min_samples_genre)
    print("After genre filter:")
    print(f"  rows={len(df)} | styles={df['style'].nunique()} | artists={df['artist'].nunique()} | genres={df['genre'].nunique()}")
    
    if df["genre"].nunique() <= 1:
        raise ValueError(
            f"Genre collapsed to {df['genre'].nunique()} class(es) after filtering. "
            "Provide real genre metadata with --genre_metadata when building wikiart_master.csv "
            "or relax genre filters to preserve genre diversity."
        )

    if df.empty:
        print("Error: no rows left after filtering. Relax filters or check master CSV.", file=sys.stderr)
        return 1

    print("Final class histograms (top classes per column):")
    _print_class_counts(df, "style")
    _print_class_counts(df, "artist")
    _print_class_counts(df, "genre")

    artist_classes = sorted(df["artist"].unique().tolist())
    style_classes = sorted(df["style"].unique().tolist())
    genre_classes = sorted(df["genre"].unique().tolist())

    artist_map = _build_label_map(artist_classes)
    style_map = _build_label_map(style_classes)
    genre_map = _build_label_map(genre_classes)

    maps = {"artist": artist_map, "style": style_map, "genre": genre_map}

    label_dir = output_dir / "label_maps"
    _write_label_file(label_dir / "artist_classes.txt", artist_classes)
    _write_label_file(label_dir / "style_classes.txt", style_classes)
    _write_label_file(label_dir / "genre_classes.txt", genre_classes)

    print(
        "Label space: "
        f"artist={len(artist_map)} classes, "
        f"style={len(style_map)} classes, "
        f"genre={len(genre_map)} classes"
    )

    if stratify_column not in ("artist", "style", "genre"):
        print(
            f"Error: stratify_column must be artist, style, or genre (got {stratify_column!r})",
            file=sys.stderr,
        )
        return 1

    train_df, val_df, test_df = _stratified_split(
        df, stratify_column, train_r, val_r, test_r, seed, stratify
    )
    print(f"Global split stratified by: {stratify_column} (stratify={stratify})")

    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("artist", "style", "genre"):
        label_map = maps[name]
        tr_df = _single_task_frames(train_df, name, label_map)
        va_df = _single_task_frames(val_df, name, label_map)
        te_df = _single_task_frames(test_df, name, label_map)

        tr_path = output_dir / f"{name}_train.csv"
        va_path = output_dir / f"{name}_val.csv"
        te_path = output_dir / f"{name}_test.csv"
        tr_df.to_csv(tr_path, index=False)
        va_df.to_csv(va_path, index=False)
        te_df.to_csv(te_path, index=False)
        print(
            f"{name}: train={len(tr_df)} val={len(va_df)} test={len(te_df)}\n"
            f"  {tr_path.resolve()}\n"
            f"  {va_path.resolve()}\n"
            f"  {te_path.resolve()}"
        )

    mtr = _multitask_frame(train_df, maps)
    mva = _multitask_frame(val_df, maps)
    mte = _multitask_frame(test_df, maps)

    print(f"\nMultitask dataset label counts:")
    print(f"  Artist: {len(artist_map)} unique classes")
    print(f"  Style: {len(style_map)} unique classes") 
    print(f"  Genre: {len(genre_map)} unique classes")

    mt_train_p = output_dir / "multitask_train.csv"
    mt_val_p = output_dir / "multitask_val.csv"
    mt_test_p = output_dir / "multitask_test.csv"
    mtr.to_csv(mt_train_p, index=False)
    mva.to_csv(mt_val_p, index=False)
    mte.to_csv(mt_test_p, index=False)
    print(
        f"multitask: train={len(mtr)} val={len(mva)} test={len(mte)}\n"
        f"  {mt_train_p.resolve()}\n"
        f"  {mt_val_p.resolve()}\n"
        f"  {mt_test_p.resolve()}"
    )

    print(f"Label maps: {label_dir.resolve()}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WikiArt train/val/test splits from wikiart_master.csv")
    parser.add_argument("--master_csv", type=str, default="datasets/wikiart_master.csv")
    parser.add_argument("--output_dir", type=str, default="datasets")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_stratify",
        action="store_true",
    )
    parser.add_argument(
        "--stratify_column",
        type=str,
        default="style",
        choices=("artist", "style", "genre"),
    )
    parser.add_argument(
        "--top_styles",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--top_artists",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--top_genres",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--min_samples_style",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--min_samples_artist",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--min_samples_genre",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    top_s = args.top_styles if args.top_styles > 0 else None
    top_a = args.top_artists if args.top_artists > 0 else None
    top_g = args.top_genres if args.top_genres > 0 else None
    min_st = args.min_samples_style if args.min_samples_style > 0 else None
    min_ar = args.min_samples_artist if args.min_samples_artist > 0 else None
    min_ge = args.min_samples_genre if args.min_samples_genre > 0 else None

    code = prepare(
        master_csv=Path(args.master_csv),
        output_dir=Path(args.output_dir),
        train_r=args.train_ratio,
        val_r=args.val_ratio,
        test_r=args.test_ratio,
        seed=args.seed,
        stratify=not args.no_stratify,
        stratify_column=args.stratify_column,
        top_styles=top_s,
        top_artists=top_a,
        top_genres=top_g,
        min_samples_style=min_st,
        min_samples_artist=min_ar,
        min_samples_genre=min_ge,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()

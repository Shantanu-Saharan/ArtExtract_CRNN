import argparse
import csv
import os
import random
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(
        description="Oversample underrepresented style/genre classes in a multitask CSV."
    )
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--style_min_count", type=int, default=800,
                   help="Oversample each style class up to at least this many rows.")
    p.add_argument("--genre_min_count", type=int, default=0,
                   help="Optional genre floor. Set 0 to disable.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return rows, reader.fieldnames


def oversample_floor(rows, column, min_count, seed):
    if min_count <= 0:
        return rows, {}

    rng = random.Random(seed)
    by_label = {}
    counts = Counter()
    for row in rows:
        label = int(row[column])
        by_label.setdefault(label, []).append(row)
        counts[label] += 1

    extra_rows = []
    added = {}
    for label, count in sorted(counts.items()):
        if count >= min_count:
            continue
        need = min_count - count
        pool = by_label[label]
        for _ in range(need):
            extra_rows.append(dict(rng.choice(pool)))
        added[label] = need

    out = list(rows) + extra_rows
    rng.shuffle(out)
    return out, added


def write_rows(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows, fieldnames = read_rows(args.input_csv)
    original_rows = len(rows)

    rows, style_added = oversample_floor(rows, "style_label", args.style_min_count, args.seed)
    rows, genre_added = oversample_floor(rows, "genre_label", args.genre_min_count, args.seed + 1000)

    write_rows(args.output_csv, fieldnames, rows)

    print(f"Input rows:  {original_rows}")
    print(f"Output rows: {len(rows)}")
    print(f"Style oversampled classes: {len(style_added)}")
    if style_added:
        print("Style added:", style_added)
    print(f"Genre oversampled classes: {len(genre_added)}")
    if genre_added:
        print("Genre added:", genre_added)


if __name__ == "__main__":
    main()

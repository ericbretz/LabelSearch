#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


BASE_PATH = Path(__file__).resolve().parent
MATCHES_DIRNAME = "matches"


def next_available_path(target_dir: Path, original_name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)

    base = Path(original_name)
    candidate = target_dir / base.name
    if not candidate.exists():
        return candidate

    stem = base.stem
    suffix = base.suffix
    counter = 1
    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def collect_matches(excel_path: Path) -> None:
    matches_dir = Path.cwd() / MATCHES_DIRNAME

    if not excel_path.exists():
        raise SystemExit(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path, keep_default_na=False)

    if "matched_keywords" not in df.columns:
        raise SystemExit("Column 'matched_keywords' not found in the Excel file.")
    if "image_path" not in df.columns:
        raise SystemExit("Column 'image_path' not found in the Excel file.")

    copied = 0
    skipped = 0

    for _, row in df.iterrows():
        raw_val = str(row.get("matched_keywords", "")).strip()
        if not raw_val or raw_val.upper() == "NA":
            skipped += 1
            continue

        image_path_value = row.get("image_path")
        if not image_path_value:
            skipped += 1
            continue

        raw_path = str(image_path_value).strip()
        candidates: list[Path] = []

        path_obj = Path(raw_path).expanduser()
        if path_obj.is_absolute():
            candidates.append(path_obj)
        else:
            candidates.append(path_obj)

        cleaned_relative = raw_path.lstrip("/\\")
        candidates.append(BASE_PATH / cleaned_relative)

        name_only = Path(raw_path).name
        if name_only:
            candidates.append(BASE_PATH / name_only)
            candidates.append(BASE_PATH / "images" / name_only)

        src: Path | None = None
        for cand in candidates:
            if cand.is_file():
                src = cand
                break

        if src is None:
            print(f"Source file not found in expected locations, skipping: {raw_path}")
            skipped += 1
            continue

        dest = next_available_path(matches_dir, src.name)
        shutil.copy2(src, dest)
        copied += 1

    print(f"Copied {copied} file(s) to '{matches_dir.name}'. Skipped {skipped} row(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect matched images from OCR results")
    parser.add_argument("excel_file", type=Path, help="Path to the Excel file with OCR results")
    args = parser.parse_args()
    
    excel_path = Path(args.excel_file).expanduser().resolve()
    collect_matches(excel_path)

#!/usr/bin/env python3
# EC Bretz
# ebretz2@uic.edu
# Sorry for over-engineering and under-commenting

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

try:
    from help import print_logo, print_help, COLORS
except ImportError:
    def print_logo():
        pass
    def print_help():
        pass
    COLORS = [
        '\033[1;91m',  # Bold Red
        '\033[1;93m',  # Bold Yellow
        '\033[1;92m',  # Bold Green
        '\033[1;96m',  # Bold Cyan
        '\033[1;94m',  # Bold Blue
        '\033[1;95m',  # Bold Magenta
    ]

BASE_PATH = Path(__file__).resolve().parent

@dataclass
class Config:
    output_dir                 : Path     = BASE_PATH
    images_dir                 : Path     = BASE_PATH / "images"
    upscaled_dir               : Path     = BASE_PATH / "upscaled"
    keywords_file              : Path     = BASE_PATH / "keywords.txt"
    progress_interval          : int      = 25
    upscale_enabled            : bool     = False
    upscale_factor             : int      = 4
    upscale_max_long_side      : int      = 1024
    log_level                  : str      = "INFO"
    save_debug_images          : bool     = False
    save_upscaled_images       : bool     = False
    review_confidence_threshold: float    = 0.85
    convert_to_rgb             : bool     = True
    image_extensions           : set[str] = field(default_factory=lambda: {
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"
    })
    allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,()-/%"

_RE_TOKENIZE           = re.compile(r"[,.\n(); ]+")
_RE_NORMALIZE          = re.compile(r"[^a-z0-9]+")
_RE_SLUGIFY            = re.compile(r"[^a-z0-9]+")
_RE_KEYWORD_SPLIT_SEMI = re.compile(r"[;]")
_RE_KEYWORD_SPLIT      = re.compile(r"[\\/|;+]")

START_TIME             = datetime.now()

def _norm_token(s: str) -> str:
    return _RE_NORMALIZE.sub("", s.lower())

def _is_iupac_like(name: str) -> bool:
    s = name.upper()
    has_digit = any(c.isdigit() for c in s)
    has_struct = any(ch in s for ch in "-(),")
    return has_digit and has_struct

def _ensure_path(value: str | Path) -> Path:
    """Resolve path into an absolute path."""
    if isinstance(value, Path):
        candidate = value
    else:
        candidate = Path(value)
    return candidate.expanduser().resolve()


def next_available_path(target_dir: Path, original_name: str) -> Path:
    """Find the next available path for a file if duplicates"""
    target_dir.mkdir(parents=True, exist_ok=True)

    base      = Path(original_name)
    candidate = target_dir / base.name
    if not candidate.exists():
        return candidate

    stem    = base.stem
    suffix  = base.suffix
    counter = 1
    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def resolve_image_path(raw_path: str) -> Path | None:
    """Resolve image paths"""
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

    for cand in candidates:
        if cand.is_file():
            return cand
    
    return None


def collect_images_from_dataframe(
    df           : pd.DataFrame,
    output_dir   : Path,
    subdir_name  : str,
    filter_column: str,
    filter_func  : callable
) -> tuple[int, int]:
    """Function to collect images from results"""
    if "image_path" not in df.columns:
        logging.error(f"Column 'image_path' not found in the DataFrame.")
        return 0, 0
    
    if filter_column not in df.columns:
        logging.error(f"Column '{filter_column}' not found in the DataFrame.")
        return 0, 0

    target_dir  = output_dir / subdir_name
    mask        = df[filter_column].apply(filter_func)
    filtered_df = df[mask].copy()
    
    if filtered_df.empty:
        logging.info(f"No images to collect for '{subdir_name}'.")
        return 0, 0

    copied  = 0
    skipped = 0

    for row in filtered_df.itertuples(index=False):
        image_path_value = getattr(row, 'image_path', None)
        if not image_path_value:
            skipped += 1
            continue

        raw_path = str(image_path_value).strip()
        src = resolve_image_path(raw_path)

        if src is None:
            logging.warning(f"Source file not found, skipping: {raw_path}")
            skipped += 1
            continue

        try:
            dest = next_available_path(target_dir, src.name)
            shutil.copy2(src, dest)
            copied += 1
        except Exception as exc:
            logging.warning(f"Failed to copy {src} to {target_dir}: {exc}")
            skipped += 1

    logging.info(f"Copied {copied} file(s) to '{target_dir}'. Skipped {skipped} row(s).")
    return copied, skipped


def collect_images(excel_path: Path, output_dir: Path, collect_matches: bool, collect_manual: bool) -> None:
    if not collect_matches and not collect_manual:
        return
    
    if not excel_path.exists():
        logging.error(f"Excel file not found: {excel_path}")
        return

    try:
        df = pd.read_excel(excel_path, keep_default_na=False)
    except Exception as exc:
        logging.error(f"Failed to read Excel file: {exc}")
        return

    if collect_matches:
        def has_matches(val) -> bool:
            raw_val = str(val).strip()
            return bool(raw_val) and raw_val.upper() != "NA"
        
        logging.info("Collecting matched images...")
        collect_images_from_dataframe(
            df, output_dir, "matches", "matched_keywords", has_matches
        )

    if collect_manual:
        def needs_review(val) -> bool:
            if isinstance(val, bool):
                return val
            elif isinstance(val, str):
                return val.strip().upper() in ("TRUE", "1", "YES", "T")
            else:
                return bool(val)
        
        logging.info("Collecting images for manual review...")
        collect_images_from_dataframe(
            df, output_dir, "manual_review", "manual_review", needs_review
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """parse any overrides for the script."""
    defaults = Config()
    parser   = argparse.ArgumentParser(
        description     = "Batch OCR pipeline for labels",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-dir",
        type    = Path,
        default = defaults.images_dir,
        help    = "Directory for input images.",
    )
    parser.add_argument(
        "--upscaled-dir",
        type    = Path,
        default = defaults.upscaled_dir,
        help    = "Where to store generated upscaled or debug images.",
    )
    parser.add_argument(
        "--output-dir",
        type    = Path,
        default = defaults.output_dir,
        help    = "Directory for all output results.",
    )
    parser.add_argument(
        "--keywords-file",
        type    = Path,
        default = defaults.keywords_file,
        help    = "Text file containing the list of keywords to match.",
    )
    parser.add_argument(
        "--progress-interval",
        type    = int,
        default = defaults.progress_interval,
        help    = "Log a progress message after this many images.",
    )
    parser.add_argument(
        "--upscale-factor",
        type    = int,
        default = defaults.upscale_factor,
        help    = "Scale factor for upscaling before OCR.",
    )
    parser.add_argument(
        "--upscale-max-long-side",
        type    = int,
        default = defaults.upscale_max_long_side,
        help    = "Maximum allowed pixels for long edge of upscaled images.",
    )
    parser.add_argument(
        "--log-level",
        type    = str,
        default = defaults.log_level,
        choices = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help    = "Console log level for this run.",
    )
    parser.add_argument(
        "--review-confidence-threshold",
        type    = float,
        default = defaults.review_confidence_threshold,
        help    = "Flag results below this confidence for manual review.",
    )

    parser.add_argument(
        "--upscale-enabled",
        dest   = "upscale_enabled",
        action = "store_true",
        help   = "Enable image upscaling.",
    )
    parser.add_argument(
        "--no-upscale",
        dest   = "upscale_enabled",
        action = "store_false",
        help   = "Disable image upscaling.",
    )

    parser.add_argument(
        "--save-debug-images",
        dest   = "save_debug_images",
        action = "store_true",
        help   = "Save debug images from paddleocr.",
    )
    parser.add_argument(
        "--no-save-debug-images",
        dest   = "save_debug_images",
        action = "store_false",
        help   = "Skip saving PaddleOCR debug images.",
    )

    parser.add_argument(
        "--save-upscaled-images",
        dest   = "save_upscaled_images",
        action = "store_true",
        help   = "Write the upscaled copies of every processed image.",
    )
    parser.add_argument(
        "--no-save-upscaled-images",
        dest   = "save_upscaled_images",
        action = "store_false",
        help   = "Do not store the upscaled copies.",
    )

    parser.add_argument(
        "--convert-to-rgb",
        dest   = "convert_to_rgb",
        action = "store_true",
        help   = "Convert BGR images to RGB before using paddle.",
    )
    parser.add_argument(
        "--no-convert-to-rgb",
        dest   = "convert_to_rgb",
        action = "store_false",
        help   = "Pass raw BGR images to PaddleOCR.",
    )

    parser.add_argument(
        "--matches",
        dest   = "collect_matches",
        action = "store_true",
        help   = "Copy images with matched keywords to a 'matches' directory.",
    )

    parser.add_argument(
        "--manual",
        dest   = "collect_manual",
        action = "store_true",
        help   = "Copy images flagged for manual review to a 'manual_review' directory.",
    )

    parser.set_defaults(
        upscale_enabled      = defaults.upscale_enabled,
        save_debug_images    = defaults.save_debug_images,
        save_upscaled_images = defaults.save_upscaled_images,
        convert_to_rgb       = defaults.convert_to_rgb,
        collect_matches      = False,
        collect_manual       = False,
    )

    return parser.parse_args(argv)


def get_config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        output_dir                  = _ensure_path(args.output_dir),
        images_dir                  = _ensure_path(args.images_dir),
        upscaled_dir                = _ensure_path(args.upscaled_dir),
        keywords_file               = _ensure_path(args.keywords_file),
        progress_interval           = max(0, int(args.progress_interval)),
        upscale_factor              = max(1, int(args.upscale_factor)),
        upscale_max_long_side       = max(1, int(args.upscale_max_long_side)),
        log_level                   = str(args.log_level).upper(),
        review_confidence_threshold = float(args.review_confidence_threshold),
        save_debug_images           = bool(args.save_debug_images),
        save_upscaled_images        = bool(args.save_upscaled_images),
        upscale_enabled             = bool(args.upscale_enabled),
        convert_to_rgb              = bool(args.convert_to_rgb),
    )
def load_keywords(path: Path) -> list[str]:
    """Read keywords from a text file"""
    keywords: list[str] = []
    if not path.exists():
        logging.warning(f"Keywords file not found: {path}")
        return keywords

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("/"):
                continue
            
            tokens = [token.strip() for token in _RE_KEYWORD_SPLIT_SEMI.split(line) if token.strip()]
            if not tokens:
                tokens = [line]

            for token in tokens:
                keywords.append(token)
                        
    return keywords

def build_known_keywords(seed_terms: Sequence[str]) -> list[str]:
    """
    Turn the raw keyword list into a clean set of canonical keywords.
    by canonical i mean split compound phrases, de-duplicate them. less redundant searching
    """
    canonical: dict[str, str] = {}
    for term in seed_terms:
        normalized = term.strip()
        if not normalized:
            continue
        canonical.setdefault(normalized.lower(), normalized)
        for fragment in _RE_KEYWORD_SPLIT.split(normalized):
            frag_norm = fragment.strip()
            if frag_norm:
                canonical.setdefault(frag_norm.lower(), frag_norm)
    return list(canonical.values())

def slugify_keyword(keyword: str) -> str:
    """Turn a keyword into an excel friendly name."""
    slug = _RE_SLUGIFY.sub("_", keyword.lower()).strip("_")
    return slug or "keyword"

def _tokenize(text: str) -> list[str]:
    """Split text into tokens using the same logic for keywords and OCR output."""
    return [t.strip() for t in _RE_TOKENIZE.split(text) if t.strip()]

class KeywordMatcher:
    def __init__(self, known_keywords: Sequence[str]):
        self.known_keywords = [term.strip() for term in known_keywords if term.strip()]

    def clean(self, text: str, score_threshold: int = 80) -> tuple[list[str], list[str]]:
        """
        Basic cleaning and tokenization (tokenizing?) of the raw OCR text.
        """
        if not text:
            return [], []

        tokens = _tokenize(text)
        if not tokens:
            return [], []

        return list(dict.fromkeys(tokens)), []

def setup_logging(config: Config) -> None:
    """Set up logging so detailed logs go to a file and only minimal messages show up in the console"""
    os.environ["GLOG_minloglevel"]    = "3"
    os.environ["PADDLEOCR_LOG_LEVEL"] = "ERROR"
    
    log_dir = BASE_PATH / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"run_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.log_level, logging.INFO))
    console_fmt = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_fmt)

    class ConsoleFilter(logging.Filter):
        def filter(self, record):
            if record.name == "py.warnings":
                return False
            if record.name.startswith("ppocr") or record.name.startswith("paddle"):
                return False
            return True

    console_handler.addFilter(ConsoleFilter())
    root_logger.addHandler(console_handler)

    logging.captureWarnings(True)


def list_image_files(directory: Path, config: Config) -> list[Path]:
    """Recursively find all image files in the directory and return them as a sorted list."""
    if not directory.exists():
        logging.error(f"Image directory does not exist: {directory}")
        return []
    image_files: list[Path] = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in config.image_extensions
    ]
    if not image_files:
        logging.warning(f"No images found under {directory}")
    return sorted(image_files)

def initialize_paddle_engine() -> None:
    """load the paddle engine."""
    global PADDLE_OCR_ENGINE
    
    os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
    
    stdout_fd       = sys.stdout.fileno()
    stderr_fd       = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    
    '''super convaluted way of stopping paddleocr from spamming the terminal'''
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(null_fd, stdout_fd)
        os.dup2(null_fd, stderr_fd)
        os.close(null_fd)
        
        try:
            import paddleocr
            from paddleocr import PaddleOCR
            paddleocr.logger.setLevel(logging.CRITICAL)
        except ImportError as exc:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            raise SystemExit("PaddleOCR is required: pip install paddleocr paddlepaddle") from exc
        
        try:
            PADDLE_OCR_ENGINE = PaddleOCR(use_textline_orientation=True, lang="en")
        finally:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

        logging.getLogger().setLevel(logging.DEBUG)
        
    except SystemExit:
        raise
    except Exception as exc:
        try:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
        except Exception:
            pass
        logging.error(f"Failed to initialize PaddleOCR: {exc}")
        raise


PADDLE_OCR_ENGINE: Optional["PaddleOCR"] = None

def load_image(image_source: str | Path | np.ndarray) -> np.ndarray:
    """Load an image, always returning the image in BGR format."""
    if isinstance(image_source, (str, Path)):
        image = cv2.imread(str(image_source), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image at {image_source}")
        return image
    if isinstance(image_source, np.ndarray):
        if image_source.ndim == 2:
            return cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR)
        if image_source.ndim == 3:
            return image_source
    raise TypeError("image_source must be a path or numpy array")

def upscale_image(image_bgr: np.ndarray, config: Config, scale: int) -> np.ndarray:
    """
    Return an upscaled copy of the image using cubic interpolation.
    """
    if not config.upscale_enabled:
        return image_bgr.copy()
    if scale <= 1:
        return image_bgr.copy()

    height, width = image_bgr.shape[:2]
    longest_side = max(height, width)

    if longest_side >= config.upscale_max_long_side:
        return image_bgr.copy()

    max_allowed_scale = config.upscale_max_long_side / float(longest_side)
    effective_scale   = min(float(scale), max_allowed_scale)

    return cv2.resize(
        image_bgr,
        None,
        fx=effective_scale,
        fy=effective_scale,
        interpolation=cv2.INTER_CUBIC,
    )

def save_upscaled_image(image_bgr: np.ndarray, original_path: Path, config: Config) -> None:
    """Save the upscaled BGR image into the `upscaled/` folder."""
    if not config.save_upscaled_images:
        return

    try:
        try:
            relative = original_path.relative_to(config.images_dir)
        except ValueError:
            relative = original_path.name
        destination = config.upscaled_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image_bgr):
            logging.warning(f"Failed to write upscaled image for {original_path}")
    except Exception as exc:
        logging.warning(f"Unable to save upscaled image for {original_path}: {exc}")

def save_paddle_debug(
    image        : np.ndarray,
    results      : list,
    original_path: Path,
    config       : Config
) -> None:
    """Save debug images with OCR detection boxes drawn on top under an `upscaled/debug_...`."""
    if not results or not results[0]:
        return
        
    try:
        debug_img = image.copy()
        
        if isinstance(results[0], dict):
            page_res = results[0]
            boxes    = page_res.get('dt_polys', [])
            for box in boxes:
                box_int = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(debug_img, [box_int], isClosed=True, color=(0, 255, 0), thickness=2)
                
        elif isinstance(results[0], list):
            for line in results[0]:
                if not line: continue
                box = np.array(line[0], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(debug_img, [box], isClosed=True, color=(0, 255, 0), thickness=2)

        try:
            relative = original_path.relative_to(config.images_dir)
        except ValueError:
            relative = original_path.name
            
        debug_path = config.upscaled_dir / f"debug_{relative}"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), debug_img)
        
    except Exception as exc:
        logging.warning(f"Failed to save debug OCR image for {original_path}: {exc}")


def run_paddle_ocr(image: np.ndarray, config: Config, source_path: Path | None = None) -> tuple[str, float | None]:
    """Run paddle on the image and return all detected text as a string plus the average confidence score"""
    try:
        if config.convert_to_rgb:
            image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_input = image

        stdout_fd       = sys.stdout.fileno()
        stderr_fd       = sys.stderr.fileno()
        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)
        try:
            devnull_fd  = os.open(os.devnull, os.O_WRONLY)
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(devnull_fd, stdout_fd)
                os.dup2(devnull_fd, stderr_fd)
            finally:
                os.close(devnull_fd)

            results = PADDLE_OCR_ENGINE.ocr(image_input)
        finally:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
        
        if source_path and config.save_debug_images:
            save_paddle_debug(image, results, source_path, config)
            
        if isinstance(results, list) and results and isinstance(results[0], dict):
            page_res      = results[0]
            texts         = page_res.get('rec_texts', [])
            confidences   = page_res.get('rec_scores', [])
            combined_text = " ".join([t.strip() for t in texts if t.strip()])
            mean_conf     = float(np.mean(confidences)) if confidences else None
            return combined_text, mean_conf

        if not results or results[0] is None:
            return "", None

        texts      : list[str]   = []
        confidences: list[float] = []
        
        for line in results[0]:
            if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                text, score = line[1]
                token = str(text).strip()
                if token:
                    texts.append(token)
                    confidences.append(float(score))

        combined_text = " ".join(texts)
        mean_conf     = float(np.mean(confidences)) if confidences else None
        return combined_text, mean_conf
    except Exception as exc:
        logging.warning(f"PaddleOCR failed: {exc}")
        return "", None

def perform_ocr(image_path: Path, config: Config, scale: int = 4) -> dict[str, str | float | bool | None]:
    """Upscale an image, run paddle on it, then package the text, confidence, and upscaling flag into a dictionary."""
    image_bgr    = load_image(image_path)
    upscaled_bgr = upscale_image(image_bgr, config, scale=scale)
    save_upscaled_image(upscaled_bgr, image_path, config)

    was_upscaled = upscaled_bgr.shape[:2] != image_bgr.shape[:2]
    paddle_ready = upscaled_bgr

    paddle_text, paddle_conf = run_paddle_ocr(paddle_ready, config, source_path=image_path)
    return {
        "paddle_text" : paddle_text,
        "paddle_conf" : paddle_conf,
        "was_upscaled": was_upscaled,
    }

def contains_keywords(
    cleaned_keywords   : list[str],
    keywords           : Sequence[str],
    threshold          : int = 85,
) -> tuple[dict[str, bool | str], list[str]]:
    """
    Return a dictionary mapping each keyword to True or False.
    depending on whether it appears in the cleaned keyword list,
    plus a list of corrected strings from the fuzzy search.
    """
    hits       : dict[str, bool | str] = {}
    corrections: list[str]             = []

    tokens = [t for t in cleaned_keywords if t]
    if not tokens:
        for keyword in keywords:
            hits[keyword] = False
        return hits, corrections

    lower_tokens      = [t.lower() for t in tokens]
    norm_tokens       = {i: _norm_token(t) for i, t in enumerate(tokens)}
    matchable_indices = [
        i for i, norm_tok in norm_tokens.items() if len(norm_tok) >= 3
    ]

    keyword_data: dict[str, dict] = {}
    for keyword in keywords:
        kw = keyword.strip()
        if not kw:
            keyword_data[keyword] = {"valid": False}
            continue

        kw_lower  = kw.lower()
        kw_tokens = _tokenize(kw)
        if not kw_tokens:
            keyword_data[keyword] = {"valid": False}
            continue

        kw_norm_tokens      = {i: _norm_token(t) for i, t in enumerate(kw_tokens)}
        kw_tokens_matchable = [
            (i, kw_norm_tokens[i], kw_tokens[i]) for i in kw_norm_tokens
            if len(kw_norm_tokens[i]) >= 3
        ]
        if not kw_tokens_matchable:
            keyword_data[keyword] = {"valid": False}
            continue

        keyword_data[keyword] = {
            "valid"              : True,
            "kw"                 : kw,
            "kw_lower"           : kw_lower,
            "kw_clean"           : _RE_NORMALIZE.sub("", kw_lower),
            "kw_tokens_matchable": kw_tokens_matchable,
        }

    for keyword, kw_data in keyword_data.items():
        if not kw_data["valid"]:
            hits[keyword] = False
            continue

        kw                  = kw_data["kw"]
        kw_lower            = kw_data["kw_lower"]
        kw_clean            = kw_data["kw_clean"]
        kw_tokens_matchable = kw_data["kw_tokens_matchable"]

        best_positions: list[int]   = []
        best_scores   : list[float] = []
        all_matched                 = True

        for tok_idx, tok_norm, tok_orig in kw_tokens_matchable:
            tok_lower  = tok_orig.lower()
            best_score = -1.0
            best_idx   = -1
            for idx in matchable_indices: 
                raw_tok = lower_tokens[idx]
                score   = fuzz.partial_ratio(tok_lower, raw_tok)
                if score > best_score:
                    best_score = score
                    best_idx   = idx

            if best_score < threshold:
                all_matched = False
                break

            best_positions.append(best_idx)
            best_scores.append(best_score)

        if not all_matched:
            hits[keyword] = False
            continue

        coverage  = len(best_scores) / len(kw_tokens_matchable)
        avg_score = sum(best_scores) / len(best_scores)

        span = max(best_positions) - min(best_positions) + 1
        if coverage < 0.6 or span > len(kw_tokens_matchable) + 3:
            hits[keyword] = False
            continue

        snippet_tokens = tokens[min(best_positions): max(best_positions)+1]
        snippet        = " ".join(snippet_tokens)
        snip_clean     = _RE_NORMALIZE.sub("", snippet.lower())
        if not kw_clean or not snip_clean:
            hits[keyword] = False
            continue

        wr        = fuzz.WRatio(snip_clean, kw_clean)
        len_ratio = len(snip_clean) / len(kw_clean)
        if wr < 75 or len_ratio < 0.7 or len_ratio > 1.5:
            hits[keyword] = False
            continue

        hits[keyword] = True

        corrections.append(f"{kw} [{snippet}] ({avg_score:.2f})")

    return hits, corrections

def process_images(
    image_paths   : Sequence[Path],
    known_keywords: Sequence[str],
    keywords      : Sequence[str],
    config        : Config,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """iterate over all image files, run OCR on each one, and return a results dataframe plus some summary stats"""
    total_images        = len(image_paths)
    missing_images      = 0
    engine_counts       = {"paddleocr": 0}
    manual_review_count = 0
    upscaled_count      = 0
    keyword_slugs       = {keyword: slugify_keyword(keyword) for keyword in keywords}
    matcher             = KeywordMatcher(known_keywords)
    iupac_flags         = {keyword: _is_iupac_like(keyword) for keyword in keywords}
    
    rows: list[dict[str, str | float | bool | None]] = []
    RESET = '\033[0m'
    progress_bar = (
        tqdm(
            total      = total_images,
            desc       = "OCR",
            ncols      = 80,
            bar_format = "{l_bar}{bar} {n_fmt}/{total_fmt}  Elapsed: {elapsed}  ETA: {remaining}",
            leave      = False,
        )
        if total_images > 0
        else None
    )

    for idx, image_path in enumerate(image_paths):
        row_data: dict[str, str | float | bool | None] = {
            "image_name"       : image_path.name,
            "image_path"       : str(image_path),
            "raw_text"         : "",
            "confidence"       : None,
            "manual_review"    : False,
            "review_reason"    : "",
            "cleaned_keywords" : "",
            "fuzzy_corrections": "",
            "matched_keywords" : "",
            "was_upscaled"     : False,
        }
        for keyword, slug in keyword_slugs.items():
            row_data[f"contains_{slug}"] = False

        if not image_path.exists():
            missing_images += 1
            logging.warning(f"Image not found: {image_path}")
        else:
            try:
                ocr_result                    = perform_ocr(image_path, config, scale=config.upscale_factor)
                text_result                   = ocr_result["paddle_text"]
                conf_result                   = ocr_result["paddle_conf"]
                was_upscaled                  = bool(ocr_result.get("was_upscaled", False))
                cleaned_tokens, corrections   = matcher.clean(text_result)
                keyword_hits, det_corrections = contains_keywords(cleaned_tokens, keywords)
                matched_keywords              = ", ".join(kw for kw, hit in keyword_hits.items() if hit)
                requires_review               = False
                reasons                       = []

                if conf_result is None or conf_result < config.review_confidence_threshold:
                    requires_review = True
                    reasons.append(f"Low Confidence ({conf_result:.2f})" if conf_result else "No Confidence Score")

                if not cleaned_tokens and text_result:
                    requires_review = True
                    reasons.append("No Keywords Found")
                
                if not text_result:
                    requires_review = True
                    reasons.append("No Text Detected")

                if any(keyword_hits.get(kw, False) and iupac_flags.get(kw, False) for kw in keywords):
                    requires_review = True
                    reasons.append("IUPAC chemical name match")

                row_data.update(
                    {
                        "raw_text"         : text_result,
                        "confidence"       : conf_result,
                        "manual_review"    : requires_review,
                        "review_reason"    : "; ".join(reasons),
                        "cleaned_keywords" : ", ".join(cleaned_tokens),
                        "fuzzy_corrections": "; ".join(corrections + det_corrections),
                        "matched_keywords" : matched_keywords,
                        "was_upscaled"     : was_upscaled,
                    }
                )
                for keyword, slug in keyword_slugs.items():
                    row_data[f"contains_{slug}"] = keyword_hits.get(keyword, False)

                if was_upscaled:
                    upscaled_count += 1

                if text_result:
                    engine_counts["paddleocr"] += 1
                
                if requires_review:
                    manual_review_count += 1
                    
            except Exception as exc:
                logging.warning(f"OCR failed for {image_path}: {exc}")

        rows.append(row_data)

        if config.progress_interval > 0 and (idx + 1) % config.progress_interval == 0:
            if not progress_bar:
                logging.info(f"Processed {idx + 1}/{total_images} images")
        if progress_bar:
            color = COLORS[idx % len(COLORS)]
            progress_bar.bar_format = (
                "{l_bar}"
                f"{color}"
                "{bar}"
                f"{RESET} {{n_fmt}}/{{total_fmt}}  Elapsed: {{elapsed}}  ETA: {{remaining}}"
            )
            progress_bar.update(1)

    if progress_bar:
        progress_bar.close()

    df = pd.DataFrame(rows)
    if df.empty:
        base_columns = [
            "image_name",
            "image_path",
            "raw_text",
            "confidence",
            "manual_review",
            "review_reason",
            "cleaned_keywords",
            "fuzzy_corrections",
            "matched_keywords",
            "was_upscaled",
        ]
        df = pd.DataFrame(columns=base_columns + [f"contains_{slug}" for slug in keyword_slugs.values()])

    elapsed = datetime.now() - START_TIME
    stats = {
        "rows_processed"      : total_images,
        "missing_images"      : missing_images,
        "paddleocr_chosen"    : engine_counts["paddleocr"],
        "manual_review_count" : manual_review_count,
        "elapsed_time_seconds": round(elapsed.total_seconds(), 2),
        "upscaled_count"      : upscaled_count,
    }
    return df, stats


def main(argv: Sequence[str] | None = None) -> None:
    """Run the pipeline using defaults and any CLI overrides."""
    print_logo()
    
    if argv is None:
        argv = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if not argv or '--help' in argv or '-h' in argv:
        print_help()
        return
    
    args   = parse_args(argv)
    config = get_config_from_args(args)
    setup_logging(config)
    logging.info("Starting OCR batch run...")
    
    logging.info("Initializing OCR engine...")
    initialize_paddle_engine()

    if not config.images_dir.exists():
        logging.error(f"Image directory not found at {config.images_dir}")
        return

    keywords       = load_keywords(config.keywords_file)
    known_keywords = build_known_keywords(keywords)
    logging.info(f"Loaded {len(keywords)} raw keywords and {len(known_keywords)} canonical keywords")

    image_files = list_image_files(config.images_dir, config)
    logging.info(f"Discovered {len(image_files)} image(s) under {config.images_dir}")

    logging.info(f"Running OCR on {len(image_files)} image(s)...")
    processed_df, stats = process_images(
        image_paths      = image_files,
        known_keywords   = known_keywords,
        keywords         = keywords,
        config           = config,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_excel = config.output_dir / "results.xlsx"
    
    logging.info(f"Saving results to {output_excel}")
    processed_df = processed_df.replace("", pd.NA)
    processed_df.to_excel(output_excel, index=False, engine="openpyxl", na_rep="NA")

    if args.collect_matches or args.collect_manual:
        try:
            collect_images(output_excel, config.output_dir, args.collect_matches, args.collect_manual)
        except Exception as exc:
            logging.error(f"Failed to collect images: {exc}")

    summary_lines = [
        f"Images processed:   {stats['rows_processed']}",
        f"Missing images:     {stats['missing_images']}",
        f"Images upscaled:    {stats.get('upscaled_count', 'n/a')}",
        f"Flagged for Review: {stats['manual_review_count']}",
        f"Elapsed time (s):   {stats.get('elapsed_time_seconds', 'n/a')}",
    ]
    keyword_counts: list[tuple[str, int]] = []
    keyword_slug_map = {keyword: slugify_keyword(keyword) for keyword in keywords}
    for keyword, slug in keyword_slug_map.items():
        column_name = f"contains_{slug}"
        if column_name in processed_df:
            col_data    = processed_df[column_name].fillna(False)
            match_count = int(col_data.astype(bool).sum())
            keyword_counts.append((keyword, match_count))
    if keyword_counts:
        summary_lines.append("Counts")
        longest_keyword = max(len(keyword) for keyword, _ in keyword_counts)
        for keyword, match_count in keyword_counts:
            summary_lines.append(f"  {keyword:<{longest_keyword}}   {match_count}")
    if keywords and "matched_keywords" in processed_df:
        matched_col = processed_df["matched_keywords"].fillna("")
        multi_hit   = int(matched_col.astype(str).str.len().gt(0).sum())
        summary_lines.append(f"Images with hits:   {multi_hit}")

    for line in summary_lines:
        logging.info(line)

    logging.info("OCR batch run finished.")

if __name__ == "__main__":
    main()


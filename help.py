from pathlib import Path
import re

'''over engineered logo and help printer because why not'''

MAJOR = 1
MINOR = 0
PATCH = 1
VERSION = f"v{MAJOR}.{MINOR}.{PATCH}"

AUTHOR = "EC BRETZ"

FADE_IN = f'█▓▒░'
FADE_OUT= f'░▒▓█'

LOGO_LABEL = [
f' _           _           _ ',
f'| |         | |         | |',
f'| |     __ _| |__   ___ | |',
f'| |    / _` | \'_ \\ / _ \\| |',
f'| |___| (_| | |_) |  __/| |',
f'|______\\__,_|_.__/ \\___||_|'
]

LOGO_SEARCH = [
f'  _____                     _     ',
f' / ____|                   | |    ',
f'| (___   ___  __ _ _ __ ___| |__  ',
f' \\___ \\ / _ \\/ _` | \'__/ __| \'_ \\ ',
f' ____) |  __/ (_| | | | (__| | | |',
f'|_____/ \\___|\\__,_|_|  \\___|_| |_|'
]

COLORS = [
    '\033[1;91m',  # Bold Red
    '\033[1;93m',  # Bold Yellow
    '\033[1;92m',  # Bold Green
    '\033[1;96m',  # Bold Cyan
    '\033[1;94m',  # Bold Blue
    '\033[1;95m',  # Bold Magenta
]

def _strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def print_logo():
    RESET = '\033[0m'
    WIDTH = 80
    left_border_raw  = FADE_IN
    right_border_raw = FADE_OUT
    inner_width      = WIDTH - len(left_border_raw) - len(right_border_raw)
    
    for i, line in enumerate(LOGO_LABEL):
        label_color  = COLORS[i % len(COLORS)]
        search_color = COLORS[(5 - i) % len(COLORS)]

        left_border_col  = label_color  + left_border_raw  + RESET
        right_border_col = search_color + right_border_raw + RESET

        label = label_color + line + RESET
        search = search_color + LOGO_SEARCH[i] + RESET
        combined = label + search
        visible_length = len(_strip_ansi(combined))
        padding_left = max(0, (inner_width - visible_length) // 2)
        padding_right = max(0, inner_width - visible_length - padding_left)
        line_str = (
            left_border_col +
            ' ' * padding_left +
            combined +
            ' ' * padding_right +
            right_border_col
        )
        print(line_str)
    
    print(f'{AUTHOR:>{WIDTH}}')
    print(f'{VERSION:>{WIDTH}}')

def print_help():
    BASE_PATH = Path(__file__).resolve().parent
    
    defaults = {
        'images_dir'                 : BASE_PATH / "images",
        'upscaled_dir'               : BASE_PATH / "upscaled",
        'output_excel'               : BASE_PATH / "ocr_results.xlsx",
        'keywords_file'              : BASE_PATH / "keywords.txt",
        'progress_interval'          : 25,
        'upscale_factor'             : 4,
        'upscale_max_long_side'      : 1024,
        'log_level'                  : "INFO",
        'review_confidence_threshold': 0.85,
    }
    
    WHITE_BG = '\033[47m'
    RESET = '\033[0m'
    WIDTH = 80
    
    print(f"{WHITE_BG}{'USAGE:':<80}{RESET}")
    print("  python labelsearch.py [OPTIONS]\n")
    
    print(f"{WHITE_BG}{'DESCRIPTION:':<80}{RESET}")
    print("  Pipeline for extracting and matching ingredient labels from images.\n")
    
    print(f"{WHITE_BG}{'INPUT/OUTPUT PATHS:':<80}{RESET}\n")
    
    paths = [
        ("--images-dir",    "Directory containing input images to process", defaults['images_dir']),
        ("--output-excel",  "Destination Excel file for OCR results", defaults['output_excel']),
        ("--keywords-file", "Text file containing keywords to match", defaults['keywords_file']),
        ("--upscaled-dir",  "Directory for upscaled/debug images", defaults['upscaled_dir']),
    ]
    
    for arg, desc, default in paths:
        print(f"  {arg:<30}  {desc}")
        if isinstance(default, Path):
            try:
                if default.parent == BASE_PATH:
                    default_str = default.name
                else:
                    default_str = str(default.relative_to(BASE_PATH))
            except ValueError:
                default_str = str(default)
        else:
            default_str = str(default)
        print(f"  {'':<32}  Default: {default_str}\n")
    
    print(f"{WHITE_BG}{'PROCESSING OPTIONS:':<80}{RESET}\n")
    
    processing = [
        ("--progress-interval",             "Log progress message after N images", defaults['progress_interval']),
        ("--log-level",                     "Log level (CRITICAL|ERROR|WARNING|INFO|DEBUG)", defaults['log_level']),
        ("--review-confidence-threshold",   "Flag results below this confidence for review", defaults['review_confidence_threshold']),
    ]
    
    for arg, desc, default in processing:
        print(f"  {arg:<30}  {desc}")
        print(f"  {'':<32}  Default: {default}\n")
    
    print(f"{WHITE_BG}{'UPSCALING OPTIONS:':<80}{RESET}\n")
    
    upscaling = [
        ("--upscale-enabled",       "Enable image upscaling before OCR", "False"),
        ("--no-upscale",            "Disable image upscaling (default)", ""),
        ("--upscale-factor",        "Scale factor for upscaling", defaults['upscale_factor']),
        ("--upscale-max-long-side", "Max pixels for long edge of upscaled images", defaults['upscale_max_long_side']),
    ]
    
    for arg, desc, default in upscaling:
        print(f"  {arg:<30}  {desc}")
        if default:
            print(f"  {'':<32}  Default: {default}\n")
        else:
            print()
    
    print(f"{WHITE_BG}{'DEBUG/OUTPUT OPTIONS:':<80}{RESET}\n")
    
    debug_options = [
        ("--save-debug-images",         "Save debug images with OCR detection boxes", "False"),
        ("--no-save-debug-images",      "Skip saving debug images (default)", ""),
        ("--save-upscaled-images",      "Save upscaled copies of processed images", "False"),
        ("--no-save-upscaled-images",   "Do not save upscaled copies (default)", ""),
        ("--convert-to-rgb",            "Convert BGR images to RGB before OCR", "True"),
        ("--no-convert-to-rgb",         "Pass raw BGR images to PaddleOCR", ""),
    ]
    
    for arg, desc, default in debug_options:
        print(f"  {arg:<30}  {desc}")
        if default:
            print(f"  {'':<32}  Default: {default}\n")
        else:
            print()

if __name__ == "__main__":
    print_logo()
    print_help()
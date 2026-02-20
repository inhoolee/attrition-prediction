"""Plot style helpers for cross-platform Korean text rendering."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns


KOREAN_FONT_CANDIDATES = [
    "Apple SD Gothic Neo",
    "AppleGothic",
    "Malgun Gothic",
    "NanumGothic",
    "NanumBarunGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "Source Han Sans KR",
]


def _pick_korean_font() -> str | None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in KOREAN_FONT_CANDIDATES:
        if font_name in available_fonts:
            return font_name
    return None


def _enable_korean_font_support() -> None:
    """Try to register Korean fonts shipped by optional helper packages."""
    try:
        # This package configures matplotlib to use bundled Nanum fonts.
        import koreanize_matplotlib  # noqa: F401
    except Exception:
        return


def apply_plot_style(use_seaborn_theme: bool = False, figsize: tuple[int, int] = (12, 6)) -> str | None:
    """Apply matplotlib defaults and choose a Korean-capable font if available."""
    _enable_korean_font_support()
    selected_font = _pick_korean_font()

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["axes.unicode_minus"] = False

    if selected_font:
        plt.rcParams["font.family"] = selected_font
        if use_seaborn_theme:
            sns.set_theme(style="whitegrid", font=selected_font)
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
        if use_seaborn_theme:
            sns.set_theme(style="whitegrid")
        # Keep notebooks readable by silencing repeated missing-glyph warnings.
        warnings.filterwarnings(
            "ignore",
            message=r"Glyph .* missing from font\(s\) DejaVu Sans",
            category=UserWarning,
        )

    return selected_font

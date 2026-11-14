"""House figure style, shared by the scripts that draw figures outside the notebooks.

The values are those of the notebooks' figures (the palette of common/trending_shared.py): small type, near-black ink, warm greys, and the categorical palette whose first
colour (blue) is the accent. Font: Arial, not the notebooks' Helvetica Neue, because macOS ships Helvetica Neue
as a .ttc collection of which matplotlib loads only the regular face (bold silently falls back to regular),
and because Arial exists on almost every system. Use:
    from common.figstyle import apply_style, style_axes, fig_title, INK, SECONDARY, MUTED, NEUTRAL, GRID, CAT
    apply_style()
"""
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

SURFACE, INK, SECONDARY, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
NEUTRAL = "#b5b4ab"          # data that is not the point of the figure
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ACCENT = CAT[0]

STYLE_RC = {
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "text.color": INK, "axes.labelcolor": SECONDARY, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "xtick.color": SECONDARY, "ytick.color": SECONDARY, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.minor.size": 0, "ytick.minor.size": 0, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "savefig.facecolor": SURFACE, "savefig.dpi": 200, "savefig.bbox": "tight",
}


def apply_style():
    plt.rcParams.update(STYLE_RC)


def style_axes(ax, grid="y"):
    """No box, light gridlines on `grid` ("x", "y", "both" or None), axis line at the bottom only."""
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def fig_title(fig, title, subtitle=None):
    """Finding as a bold title at the top left, with a quiet subtitle under it (as in the notebook figures)."""
    # Both sit just above the figure area, at fixed point offsets, so their spacing does not depend on the
    # figure's height; savefig's tight bounding box keeps them.
    above = lambda pts: offset_copy(fig.transFigure, fig=fig, y=pts, units="points")  # noqa: E731
    fig.text(0.01, 1.0, title, fontsize=13, fontweight="bold", color=INK, ha="left", va="bottom",
             transform=above(16 if subtitle else 4))
    if subtitle:
        fig.text(0.01, 1.0, subtitle, fontsize=8.5, color=SECONDARY, ha="left", va="bottom", transform=above(3))

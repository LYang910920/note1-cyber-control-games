"""Compatibility layer for the shared foundation plotting helpers."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

try:
    from cybercontrol.plotting import (  # type: ignore
        PUBLICATION_COLORS,
        PUBLICATION_LINESTYLES,
        PUBLICATION_MARKERS,
        add_arrow,
        add_box,
        guide_style,
        panel_label,
        publication_style,
        save_guide_figure,
        save_publication_figure,
        style_axis,
    )
except ImportError:
    PUBLICATION_COLORS = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#000000",
    )
    PUBLICATION_LINESTYLES = ("-", "--", "-.", ":", (0, (5, 2, 1, 2)))
    PUBLICATION_MARKERS = ("o", "s", "^", "D", "v", "P", "X")

    @contextmanager
    def publication_style():
        """Matplotlib context close to the foundation publication style."""

        import matplotlib as mpl

        with mpl.rc_context(
            {
                "figure.dpi": 120,
                "savefig.dpi": 600,
                "font.size": 8,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "lines.linewidth": 1.4,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.bbox": "tight",
            }
        ):
            yield

    @contextmanager
    def guide_style():
        """Matplotlib context for tutorial diagrams."""

        import matplotlib as mpl

        with mpl.rc_context(
            {
                "figure.dpi": 120,
                "savefig.dpi": 220,
                "font.size": 10,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "lines.linewidth": 2.0,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.bbox": "tight",
            }
        ):
            yield

    def panel_label(ax, label: str, x: float = -0.12, y: float = 1.04) -> None:
        """Place a compact label in axes coordinates."""

        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize="medium",
            fontweight="bold",
        )

    def style_axis(
        ax,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        legend: bool = False,
        grid: bool = True,
        alpha: float = 0.25,
    ) -> None:
        """Apply common labels, optional panel title, grid, and legend."""

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            panel_label(ax, title)
        if grid:
            ax.grid(True, alpha=alpha)
        if legend:
            ax.legend(loc="best", frameon=False)

    def _save_with_manifest(fig, stem, *, formats, dpi, style, metadata=None):
        stem = Path(stem)
        if stem.suffix:
            stem = stem.with_suffix("")
        stem.parent.mkdir(parents=True, exist_ok=True)
        written = []
        for fmt in formats:
            out = stem.with_suffix(f".{fmt}")
            fig.savefig(out, dpi=dpi if fmt == "png" else None)
            written.append(out)
        manifest = stem.with_suffix(".figure.json")
        manifest.write_text(
            json.dumps(
                {
                    "style": style,
                    "formats": list(formats),
                    "dpi": dpi,
                    "files": [path.name for path in written],
                    "metadata": dict(metadata or {}),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(manifest)
        return written

    def save_publication_figure(fig, stem, *, formats=("pdf", "png"), dpi=600, metadata=None):
        """Save vector/raster publication outputs."""

        return _save_with_manifest(fig, stem, formats=formats, dpi=dpi, style="publication", metadata=metadata)

    def save_guide_figure(fig, stem, *, formats=("png",), dpi=220, metadata=None):
        """Save tutorial guide outputs."""

        return _save_with_manifest(fig, stem, formats=formats, dpi=dpi, style="guide", metadata=metadata)

    def add_box(ax, xy, text, width=1.8, height=0.55, fc="#f7f7f7", ec="#333333", fontsize=8.5):
        """Draw a labeled rectangle."""

        import matplotlib.patches as patches

        box = patches.FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.0,
            facecolor=fc,
            edgecolor=ec,
        )
        ax.add_patch(box)
        ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)
        return box

    def add_arrow(ax, start, end, color="#333333", linewidth=1.4):
        """Draw an arrow between two diagram points."""

        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": linewidth},
        )

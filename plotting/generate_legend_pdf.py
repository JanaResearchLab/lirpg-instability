"""Render a standalone legend, to pair with tb_events_to_report.py --no_legend.

Usage: python generate_legend_pdf.py --labels value policy --colors '#d62728' '#9467bd' --output legend.pdf
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 15})


def parse_args():
    parser = argparse.ArgumentParser(description="Render a standalone legend.")
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Legend entry labels, in order.",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        required=True,
        help="Matplotlib colors, one per label.",
    )
    parser.add_argument(
        "--linestyles",
        nargs="+",
        default=None,
        help="Linestyles, one per label (default: solid).",
    )
    parser.add_argument(
        "--hatches",
        nargs="+",
        default=None,
        help="Hatch patterns, one per label ('' for none); draws patches instead of lines.",
    )
    parser.add_argument(
        "--hatch_color",
        type=str,
        default="black",
        help="Hatch color for patch swatches.",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=2.5,
        help="Line thickness for legend swatches.",
    )
    parser.add_argument(
        "--ncol",
        type=int,
        default=None,
        help="Legend columns (default: one row).",
    )
    parser.add_argument(
        "--fontsize",
        type=float,
        default=15,
        help="Font size for legend text.",
    )
    parser.add_argument(
        "--frame",
        action="store_true",
        help="Draw a frame around the legend.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.labels) != len(args.colors):
        raise SystemExit(
            f"--labels ({len(args.labels)}) and --colors ({len(args.colors)}) "
            "must have the same length."
        )
    linestyles = args.linestyles or ["-"] * len(args.labels)
    if len(linestyles) != len(args.labels):
        raise SystemExit(
            f"--linestyles ({len(linestyles)}) must match --labels "
            f"({len(args.labels)}) when provided."
        )

    if args.hatches is not None:
        if len(args.hatches) != len(args.labels):
            raise SystemExit(
                f"--hatches ({len(args.hatches)}) must match --labels "
                f"({len(args.labels)}) when provided."
            )
        handles = [
            Patch(
                facecolor=c,
                edgecolor="none",
                hatch=h if h else None,
                linewidth=0.0,
            )
            for c, h in zip(args.colors, args.hatches)
        ]
    else:
        handles = [
            Line2D([0], [0], color=c, linestyle=ls, linewidth=args.linewidth)
            for c, ls in zip(args.colors, linestyles)
        ]

    ncol = args.ncol if args.ncol is not None else len(args.labels)

    fig = plt.figure(figsize=(0.01, 0.01))
    legend = fig.legend(
        handles=handles,
        labels=args.labels,
        loc="center",
        ncol=ncol,
        fontsize=args.fontsize,
        frameon=args.frame,
        handlelength=2.0,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    fig.savefig(args.output, bbox_inches=bbox)
    plt.close(fig)
    print(f"Saved legend to {args.output}")


if __name__ == "__main__":
    main()

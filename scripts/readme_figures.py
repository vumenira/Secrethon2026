"""Draw the README's charts into docs/img/ (no episode is played: the numbers are the logged ones).

    uv run python scripts/readme_figures.py

- ``progress_board.png``: the public board (Small) score of each upload that moved our best, from ``sbf status``.
- ``progress_steps.png``: the local Small and Full scores of each step of the pulse agent, from SOLUTION.md's table
  (root 2026: 64 Small episodes, 16 Full episodes).
- ``pulse.png``: a made-up grid that shows the pulse's idea (round numbers, not a played episode).
"""

from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUT = Path(__file__).resolve().parents[1] / "docs" / "img"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"  # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2

# (version, public board score): the uploads that raised our best score, in order
BOARD = [
    ("mpc_det_h20\n(baseline)", 0.6751),
    ("ft_buffer", 0.8289),
    ("v2", 0.8429),
    ("v3", 0.8498),
    ("v4", 0.8577),
    ("v6", 0.8714),
    ("v8", 0.8768),
    ("v9", 0.8827),
    ("v10", 0.8854),
    ("v12", 0.8910),
    ("v13", 0.8972),
]

# (step, Small, Full): SOLUTION.md, "What it scores"; each step adds to the one before
STEPS = [
    ("LP\nplanner", 0.669, 0.290),
    ("+ window-end\nvalue", 0.743, 0.619),
    ("+ pulse,\nreserve,\npremium", 0.821, 0.857),
    ("+ recovery\nforecast", 0.839, 0.867),
    ("+ wafer\ntop-up", 0.844, 0.871),
    ("+ small\nfuel", 0.850, 0.873),
    ("+ fuel\ntrims", 0.865, 0.882),
    ("+ opening\nhold", 0.874, 0.886),
    ("v9", 0.877, 0.894),
    ("v10", 0.880, 0.898),
    ("v11", 0.886, 0.905),
    ("v12", 0.886, 0.913),
    ("v13", 0.893, 0.915),
]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def board():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150, facecolor=SURFACE)
    style(ax)
    x = range(len(BOARD))
    y = [s for _, s in BOARD]
    ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    for i, s in enumerate(y):
        ax.annotate(
            f"{s:.4f}",
            (i, s),
            textcoords="offset points",
            xytext=(12, -3) if i == 0 else (0, 9),
            ha="left" if i == 0 else "center",
            fontsize=8,
            color=TEXT if i in (0, len(y) - 1) else TEXT_2,
            fontweight="bold" if i == len(y) - 1 else None,
        )
    ax.set_xticks(list(x), [v for v, _ in BOARD])
    ax.set_ylim(0.64, 0.92)
    ax.set_ylabel("score (0 = naive, 1 = clairvoyant)", color=TEXT_2, fontsize=9)
    ax.set_title("Public board (Small): our best upload over the hackathon", loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "progress_board.png", facecolor=SURFACE)


def steps():
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150, facecolor=SURFACE)
    style(ax)
    x = list(range(len(STEPS)))
    series = ((1, "Small (public board's network)", BLUE), (2, "Full (private board's network)", ORANGE))
    for col, name, color in series:
        y = [row[col] for row in STEPS]
        ax.plot(
            x,
            y,
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=name,
        )
        ax.annotate(
            f"{name.split()[0]} {y[-1]:.3f}",
            (x[-1], y[-1]),
            textcoords="offset points",
            xytext=(8, 4 if col == 2 else -10),
            fontsize=9,
            color=TEXT,
        )
    ax.annotate(
        "Full 0.290: the planner forgets\neverything after its 20-week window",
        (0, 0.290),
        textcoords="offset points",
        xytext=(12, -4),
        fontsize=8,
        color=TEXT_2,
        va="center",
    )
    ax.set_xticks(x, [s for s, _, _ in STEPS], fontsize=8)
    ax.set_xlim(-0.4, len(STEPS) + 0.6)
    ax.set_ylim(0.25, 0.96)
    ax.set_ylabel("local score, episodes not tuned on", color=TEXT_2, fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TEXT)
    ax.set_title(
        "What each step of the pulse agent added (local scores, root 2026)", loc="left", color=TEXT, fontsize=12
    )
    fig.tight_layout()
    fig.savefig(OUT / "progress_steps.png", facecolor=SURFACE)


def pulse_figure():
    """Ten weeks of a grid whose fuel covers 90 % of its need: delivered evenly, or held and delivered in whole weeks.

    Base load 90 and fabs 10 units of power a week; the fuel brings 90 a week on average. The simulator serves base
    load first, so delivered evenly the fabs get nothing in any week. Held one week and delivered in full the other
    nine, the same fuel runs the fabs in 9 weeks; the cost is one week of shed base load.
    """
    weeks = range(1, 11)
    even_base, even_fab = [90] * 10, [0] * 10
    pulse_base = [0] + [90] * 9  # week 1: the fuel waits at the terminal
    pulse_fab = [0] + [10] * 9
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150, facecolor=SURFACE, sharey=True)
    for ax, base, fab, title in (
        (axes[0], even_base, even_fab, "Fuel delivered evenly: 90 a week"),
        (axes[1], pulse_base, pulse_fab, "The pulse: held one week, a whole week's 100 after"),
    ):
        style(ax)
        ax.bar(weeks, base, width=0.8, color=BLUE, label="base load served (homes, industry)")
        ax.bar(weeks, fab, width=0.8, bottom=base, color=ORANGE, label="fabs powered: chips made")
        shed = [90 - b for b in base]
        ax.bar(
            weeks,
            shed,
            width=0.8,
            bottom=base,
            color="none",
            edgecolor=TEXT_2,
            hatch="///",
            linewidth=0,
            label="base load shed (paid at VOLL)",
        )
        ax.axhline(100, color=TEXT_2, linewidth=1, linestyle="--")
        ax.set_xticks(list(weeks))
        ax.set_xlabel("week", color=TEXT_2, fontsize=9)
        ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    axes[0].annotate("need: 100", (10.4, 100), color=TEXT_2, fontsize=8, va="bottom", ha="right")
    axes[0].text(0.5, 104, "fabs dark every week", color=TEXT, fontsize=9, fontweight="bold")
    axes[1].text(0.5, 104, "fabs run 9 weeks of 10", color=TEXT, fontsize=9, fontweight="bold")
    axes[0].set_ylabel("power (made-up units)", color=TEXT_2, fontsize=9)
    axes[0].set_ylim(0, 114)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "pulse.png", facecolor=SURFACE)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    board()
    steps()
    pulse_figure()
    print("wrote progress_board.png, progress_steps.png and pulse.png to", OUT)

"""
Plot the committed stress-suite results (results/stress_runs.json) for the README and docs.

  python results/plot_results.py        writes docs/img/{pass_rate_by_run,scenario_reliability}_{light,dark}.png

Needs matplotlib (not in requirements.txt: only this script uses it).
Every number comes from stress_runs.json; nothing is typed in by hand except the phase labels.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "stress_runs.json").read_text(encoding="utf-8"))
OUT = ROOT / "docs" / "img"

# Which code change each run belongs to (run names as recorded in stress_runs.json).
PHASES = [
    ("Baseline", ["new"]),
    ("Interrupt gate,\nconfirmation FSM,\nclarifier reverted", ["new2", "new3", "new4"]),
    ("Save-target fix\n(+ diagnostic runs)", ["new5", "isolated", "traced"]),
    ("Context guards", ["new6_attempt1", "new6_attempt2", "new6_attempt3"]),
]
RUN_LABELS = {"new": "run 1", "new2": "run 2", "new3": "run 3", "new4": "run 4", "new5": "run 5",
              "isolated": "isolated", "traced": "traced", "new6_attempt1": "run 6a",
              "new6_attempt2": "run 6b", "new6_attempt3": "run 6c"}

THEMES = {
    "light": dict(surface="#ffffff", band="#f3f2ef", text="#0b0b0b", text2="#52514e", grid="#e4e3df",
                  series="#2a78d6", noise="#cde2fb", passing="#c9c7c0"),
    "dark": dict(surface="#0d1117", band="#161b22", text="#ffffff", text2="#c3c2b7", grid="#2a2f37",
                 series="#3987e5", noise="#1c3350", passing="#4a4a46"),
}
# Status colours: fixed across themes, always shipped with a text label.
FAILS, FLAKY = "#d03b3b", "#ec835a"


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["text2"], length=0, labelsize=9)


def pass_rate_by_run(theme):
    t = THEMES[theme]
    runs = {r["run"]: r for r in DATA["runs"]}
    order = [name for _, names in PHASES for name in names]
    total = runs[order[0]]["total"]

    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    _style(ax, t)

    # Phase bands
    x = 0
    for i, (label, names) in enumerate(PHASES):
        if i % 2:
            ax.axvspan(x - 0.5, x + len(names) - 0.5, color=t["band"], zorder=0, lw=0)
        ax.text(x + (len(names) - 1) / 2, total + 0.6, label, ha="center", va="bottom",
                fontsize=8, color=t["text2"], linespacing=1.15)
        x += len(names)

    # Run-to-run noise: min..max of the continuous runs after the baseline
    after = [runs[n]["passed"] for n in order[1:] if runs[n]["mode"] == "continuous"]
    ax.axhspan(min(after), max(after), color=t["noise"], alpha=0.55, zorder=1, lw=0)
    ax.text(len(order) - 0.45, (min(after) + max(after)) / 2,
            f"range after fixes\n{min(after)}-{max(after)} of {total}", ha="left", va="center",
            fontsize=8, color=t["text2"])

    xs = list(range(len(order)))
    ys = [runs[n]["passed"] for n in order]
    cont = [i for i, n in enumerate(order) if runs[n]["mode"] == "continuous"]
    ax.plot([xs[i] for i in cont], [ys[i] for i in cont], color=t["series"], lw=2, zorder=3,
            solid_capstyle="round")
    for i, n in enumerate(order):
        hollow = runs[n]["mode"] != "continuous"
        ax.scatter(i, ys[i], s=55, zorder=4, color=t["surface"] if hollow else t["series"],
                   edgecolors=t["series"], linewidths=2)

    first, last = order[0], order[-1]
    ax.annotate(f"{runs[first]['passed']}/{total} ({runs[first]['pass_rate']})", (0, ys[0]),
                xytext=(10, -2), textcoords="offset points", fontsize=9, color=t["text"], va="center")
    iso = order.index("isolated")
    ax.annotate("memory wiped between\nscenarios (not comparable)", (iso, ys[iso]), xytext=(0, 10),
                textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=t["text2"])

    ax.set_xticks(xs)
    ax.set_xticklabels([RUN_LABELS[n] for n in order], fontsize=8.5)
    ax.set_xlim(-0.5, len(order) + 1.3)
    ax.set_ylim(0, total + 3.2)
    ax.set_yticks([0, 5, 10, 15, total])
    ax.yaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(f"scenarios passed (of {total})", color=t["text2"], fontsize=9)
    fig.suptitle("Core suite: scenarios passed per run", x=0.065, ha="left", fontsize=12,
                 fontweight="bold", color=t["text"])
    fig.text(0.065, 0.895, f"{DATA['model']} · hollow marker = isolated (diagnostic) run",
             fontsize=8.5, color=t["text2"])
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def scenario_reliability(theme):
    t = THEMES[theme]
    counts = DATA["per_scenario_pass_counts_continuous_runs"]
    rows = sorted(counts.items(), key=lambda kv: (kv[1]["passed"] / kv[1]["runs"], kv[0]), reverse=True)
    n_runs = rows[0][1]["runs"]

    def colour(c):
        if c["passed"] == 0:
            return FAILS
        return t["passing"] if c["passed"] == c["runs"] else FLAKY

    fig, ax = plt.subplots(figsize=(9, 6.4), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    _style(ax, t)

    ys = range(len(rows))
    vals = [c["passed"] for _, c in rows]
    ax.barh(list(ys), [n_runs] * len(rows), height=0.62, color=t["grid"], alpha=0.45, zorder=1)
    # a 0/9 bar has no length, so never-passing scenarios get a short stub to stay visible
    ax.barh(list(ys), [v if v else 0.12 for v in vals], height=0.62, color=[colour(c) for _, c in rows], zorder=2)
    for y, (name, c) in zip(ys, rows):
        if c["passed"] < c["runs"]:
            note = "  (rover hardware offline)" if name.startswith("rover_") else ""
            if c["passed"] == 0:
                note = "  never passes"
            ax.text(max(c["passed"], 0.12) + 0.12, y, f"{c['passed']}/{c['runs']}{note}", va="center",
                    fontsize=8.5, color=t["text"])

    ax.set_yticks(list(ys))
    ax.set_yticklabels([name for name, _ in rows], fontsize=8.5, family="monospace", color=t["text"])
    ax.set_xlim(0, n_runs)
    ax.set_xticks(range(0, n_runs + 1, 3))
    ax.set_xlabel(f"continuous runs passed (of {n_runs})", color=t["text2"], fontsize=9)
    ax.xaxis.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=FAILS, label="fails every run: logic bug in the pipeline"),
                       Patch(color=FLAKY, label="fails sometimes: 3B sampling noise / environment"),
                       Patch(color=t["passing"], label="passes every run")],
              loc="lower center", bbox_to_anchor=(0.4, -0.2), ncol=3, frameon=False, fontsize=8,
              labelcolor=t["text2"], handlelength=1.2, columnspacing=1.4)
    fig.suptitle("Where it still fails: pass count per scenario", x=0.02, ha="left", fontsize=12,
                 fontweight="bold", color=t["text"])
    fig.text(0.02, 0.925, "Two scenarios never pass; the rest fail only intermittently.",
             fontsize=8.5, color=t["text2"])
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        for name, make in (("pass_rate_by_run", pass_rate_by_run), ("scenario_reliability", scenario_reliability)):
            fig = make(theme)
            path = OUT / f"{name}_{theme}.png"
            fig.savefig(path, facecolor=fig.get_facecolor())
            plt.close(fig)
            print("wrote", path.relative_to(ROOT))

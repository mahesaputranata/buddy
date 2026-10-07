"""
simulation_manager.py

Buddy-side wrapper around the LQR rover simulation (rover_sim binary +
analyze_run.py). Same architectural shape as RoverTaskManager: a small
session-scoped state object with methods a tool dispatch table calls
into. No FSM — state here is just "what did we last run", used so
follow-up questions ("show me the plot") don't need the filename
repeated.

Two artifacts per run, both written by rover_sim itself:
    results.csv    -- full time series (consumed by analyze_run.py)
    summary.json   -- verdict + metrics (consumed directly by Buddy)

Buddy never touches results.csv directly — it reads summary.json for
verdicts, and shells out to analyze_run.py only when a plot is asked
for. This keeps Buddy's reasoning fast (JSON read) and keeps plotting
logic in one place (the existing matplotlib script), not duplicated.
"""

from __future__ import annotations
import subprocess
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional


class SimulationManager:
    def __init__(
        self,
        sim_binary: str,
        analyze_script: str,
        results_root: str = "simulation_results",
        run_timeout_s: float = 60.0,
        plot_timeout_s: float = 30.0,
    ):
        self.sim_binary = sim_binary
        self.analyze_script = analyze_script
        self.results_root = results_root
        self.run_timeout_s = run_timeout_s
        self.plot_timeout_s = plot_timeout_s

        self.last_run_dir: Optional[str] = None
        self.last_summary: Optional[dict] = None
        self._last_run_ts: float | None = None

    # ── Tool: run_rover_simulation ──────────────────────────────────────────
    def run_simulation(
        self,
        scenario: str,
        amplitude: Optional[float] = None,
        wavelength: Optional[float] = None,
    ) -> str:
        if scenario not in ("straight", "sinusoidal"):
            return f"Unknown scenario '{scenario}' — I only know 'straight' or 'sinusoidal'."

        out_dir = self._build_out_dir(scenario, amplitude, wavelength)

        cmd = [self.sim_binary, "--scenario", scenario, "--out", out_dir]
        if amplitude is not None:
            cmd += ["--amplitude", str(amplitude)]
        if wavelength is not None:
            cmd += ["--wavelength", str(wavelength)]

        if not Path(self.sim_binary).exists():
            return (
                f"I can't find the simulation binary at {self.sim_binary!r}. "
                "Has it been built yet, and is the path configured correctly?"
            )

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.run_timeout_s
            )
        except subprocess.TimeoutExpired:
            return f"The simulation didn't finish within {self.run_timeout_s:.0f}s — it may be stuck."
        except Exception as e:
            return f"I couldn't launch the simulation: {e}"

        summary_path = Path(out_dir) / "summary.json"
        if not summary_path.exists():
            stderr_tail = (result.stderr or "").strip()[-300:]
            return (
                f"The simulation ran but didn't produce a summary.json in {out_dir}. "
                f"{'stderr: ' + stderr_tail if stderr_tail else ''}"
            )

        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception as e:
            return f"Simulation finished but I couldn't parse its summary: {e}"

        self.last_run_dir = out_dir
        self._last_run_ts = time.time()
        self.last_summary = summary
        return self._summarize_verdict(summary)

    # ── Tool: report_last_simulation_result ─────────────────────────────────
    def report_last_result(self) -> str:
        if not self.last_summary:
            return "I haven't run a simulation this session yet — want me to run one?"
        return self._summarize_verdict(self.last_summary)

    # ── Tool: show_simulation_plot ───────────────────────────────────────────
    def show_plot(self, open_file_fn) -> str:
        """
        open_file_fn: callable(Path) -> None, injected so this class doesn't
        need to duplicate ConversationManager._open_file_for_user's OS
        branching logic. Pass conversation_manager._open_file_for_user.
        """
        if not self.last_run_dir:
            return "I don't have a simulation run to plot yet — run one first."

        if not Path(self.analyze_script).exists():
            return f"I can't find the analysis script at {self.analyze_script!r}."

        try:
            result = subprocess.run(
                [sys.executable, self.analyze_script, self.last_run_dir],
                capture_output=True, text=True, timeout=self.plot_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return "Plot generation timed out."
        except Exception as e:
            return f"I couldn't run the plotting script: {e}"

        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip()[-300:]
            return f"Plotting script failed. {stderr_tail}"

        png_path = Path(self.last_run_dir) / "diagnostic.png"
        if not png_path.exists():
            return "The plotting script ran but I couldn't find diagnostic.png."

        try:
            open_file_fn(png_path)
        except Exception as e:
            return f"Generated the plot at {png_path}, but couldn't open it automatically: {e}"

        return "Here's the diagnostic plot — I opened it for you."

    # ── internal ──────────────────────────────────────────────────────────
    def _build_out_dir(self, scenario: str, amplitude, wavelength) -> str:
        name = scenario
        if scenario == "sinusoidal" and amplitude is not None and wavelength is not None:
            name += f"_A{amplitude}_L{wavelength}"
        else:
            # avoid collisions across repeated straight-line runs in one session
            name += f"_{int(time.time())}"
        return str(Path(self.results_root) / name)

    def _summarize_verdict(self, summary: dict) -> str:
        m = summary.get("metrics", {})
        scen = summary.get("scenario", {})
        return (
            f"{summary.get('verdict', 'UNKNOWN')} — {scen.get('description', 'run')}. "
            f"Cross-track RMS {m.get('cross_track_rms_cm', 0):.1f}cm "
            f"(max {m.get('cross_track_max_cm', 0):.1f}cm), "
            f"heading RMS {m.get('heading_rms_deg', 0):.1f}deg, "
            f"saturation {m.get('saturation_pct', 0):.1f}%."
        )
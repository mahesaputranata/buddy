"""
rover_telemetry.py

Stage 1 + 2 of the pipeline (parse -> windowed aggregate).
No ROS2, no LLM. Pure Python so it can be unit tested against
dummy log lines before touching real hardware or a real subscriber.

Wire format this expects (raw ESP32 serial lines, as now published
on /esp32/telemetry, /esp32/warnings, /esp32/odom by bridge_node.py —
the "ESP32: " logger prefix is NOT part of the topic payload):

    PID dt=0.0200 velA=0.0000 velB=0.0000 spA=0.2000 spB=0.2000 pwmA=250 pwmB=250
    WARN WATCHDOG_STOP
    OK SET_VEL 0.20 0.20
    ODOM 12 12 20
    ERR UNKNOWN_COMMAND
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import Optional
import math


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: parse a single raw line into a typed dict
# ─────────────────────────────────────────────────────────────────────────────

_PID_RE = re.compile(
    r"PID\s+dt=(?P<dt>[-\d.]+)\s+"
    r"velA=(?P<velA>[-\d.]+)\s+velB=(?P<velB>[-\d.]+)\s+"
    r"spA=(?P<spA>[-\d.]+)\s+spB=(?P<spB>[-\d.]+)\s+"
    r"pwmA=(?P<pwmA>[-\d]+)\s+pwmB=(?P<pwmB>[-\d]+)"
)
_OK_SET_VEL_RE = re.compile(r"OK\s+SET_VEL\s+(?P<vL>[-\d.]+)\s+(?P<vR>[-\d.]+)")
_ODOM_RE = re.compile(r"ODOM\s+(?P<dA>[-\d]+)\s+(?P<dB>[-\d]+)\s+(?P<dt_ms>[-\d]+)")


def parse_line(line: str, ts: Optional[float] = None) -> Optional[dict]:
    """
    Parse one raw ESP32 telemetry line into a structured dict.
    Returns None for anything unrecognized (caller decides whether
    that itself is worth flagging).
    """
    line = line.strip()
    if not line:
        return None
    ts = ts if ts is not None else time.time()

    m = _PID_RE.search(line)
    if m:
        g = m.groupdict()
        return {
            "type": "PID",
            "ts": ts,
            "dt": float(g["dt"]),
            "velA": float(g["velA"]),
            "velB": float(g["velB"]),
            "spA": float(g["spA"]),
            "spB": float(g["spB"]),
            "pwmA": int(g["pwmA"]),
            "pwmB": int(g["pwmB"]),
            "raw": line,
        }

    if line.startswith("WARN"):
        return {"type": "WARN", "ts": ts, "code": line[len("WARN"):].strip(), "raw": line}

    if line.startswith("ERR"):
        return {"type": "ERR", "ts": ts, "code": line[len("ERR"):].strip(), "raw": line}

    m = _OK_SET_VEL_RE.search(line)
    if m:
        return {
            "type": "ACK_SET_VEL",
            "ts": ts,
            "v_left": float(m.group("vL")),
            "v_right": float(m.group("vR")),
            "raw": line,
        }

    m = _ODOM_RE.search(line)
    if m:
        return {
            "type": "ODOM",
            "ts": ts,
            "dA": int(m.group("dA")),
            "dB": int(m.group("dB")),
            "dt_ms": int(m.group("dt_ms")),
            "raw": line,
        }

    if line == "PONG":
        return {"type": "PONG", "ts": ts, "raw": line}

    # Unrecognized — still return something so we don't silently drop data
    return {"type": "UNKNOWN", "ts": ts, "raw": line}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: rolling window that aggregates parsed samples into a summary
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WindowSummary:
    """What gets handed to the anomaly checker / LLM. Deliberately small."""
    duration_s: float
    sample_count: int
    mean_velA: float
    mean_velB: float
    max_velA: float
    max_velB: float
    mean_spA: float
    mean_spB: float
    max_pwmA: int
    max_pwmB: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    vel_never_left_zero: bool = False
    watchdog_stopped: bool = False


class RoverTelemetryWindow:
    """
    Accumulates parsed telemetry samples for one command/observation
    window, then produces a compact WindowSummary on close().

    Usage:
        w = RoverTelemetryWindow()
        w.feed(parse_line(raw_line))
        ...
        summary = w.close()
    """

    def __init__(self):
        self._samples: list[dict] = []
        self._warnings: list[str] = []
        self._errors: list[str] = []
        self._start_ts: Optional[float] = None
        self._end_ts: Optional[float] = None

    def feed(self, parsed: Optional[dict]):
        if parsed is None:
            return
        ts = parsed["ts"]
        self._start_ts = ts if self._start_ts is None else min(self._start_ts, ts)
        self._end_ts = ts if self._end_ts is None else max(self._end_ts, ts)

        if parsed["type"] == "PID":
            self._samples.append(parsed)
        elif parsed["type"] == "WARN":
            self._warnings.append(parsed["code"])
        elif parsed["type"] == "ERR":
            self._errors.append(parsed["code"])
        # ACK_SET_VEL / ODOM / PONG / UNKNOWN currently not aggregated here —
        # extend when needed (e.g. ODOM -> distance traveled).

    def get_samples(self) -> list[dict]:
        """Raw PID samples collected so far (for CSV export / simulation input)."""
        return list(self._samples)

    def close(self) -> WindowSummary:
        n = len(self._samples)
        duration = (
            (self._end_ts - self._start_ts)
            if (self._start_ts is not None and self._end_ts is not None)
            else 0.0
        )

        if n == 0:
            return WindowSummary(
                duration_s=round(duration, 3),
                sample_count=0,
                mean_velA=0.0, mean_velB=0.0,
                max_velA=0.0, max_velB=0.0,
                mean_spA=0.0, mean_spB=0.0,
                max_pwmA=0, max_pwmB=0,
                warnings=list(self._warnings),
                errors=list(self._errors),
                vel_never_left_zero=False,
                watchdog_stopped=any("WATCHDOG" in w for w in self._warnings),
            )

        velAs = [s["velA"] for s in self._samples]
        velBs = [s["velB"] for s in self._samples]
        spAs = [s["spA"] for s in self._samples]
        spBs = [s["spB"] for s in self._samples]
        pwmAs = [s["pwmA"] for s in self._samples]
        pwmBs = [s["pwmB"] for s in self._samples]

        commanded_nonzero = any(abs(sp) > 1e-6 for sp in spAs + spBs)
        vel_never_left_zero = commanded_nonzero and all(abs(v) < 1e-6 for v in velAs + velBs)

        return WindowSummary(
            duration_s=round(duration, 3),
            sample_count=n,
            mean_velA=round(sum(velAs) / n, 4),
            mean_velB=round(sum(velBs) / n, 4),
            max_velA=round(max(velAs), 4),
            max_velB=round(max(velBs), 4),
            mean_spA=round(sum(spAs) / n, 4),
            mean_spB=round(sum(spBs) / n, 4),
            max_pwmA=max(pwmAs),
            max_pwmB=max(pwmBs),
            warnings=list(self._warnings),
            errors=list(self._errors),
            vel_never_left_zero=vel_never_left_zero,
            watchdog_stopped=any("WATCHDOG" in w for w in self._warnings),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 (deterministic half): threshold-based flag checker
# No LLM here — this decides WHETHER Buddy should say anything unprompted.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TelemetryVerdict:
    ok: bool
    flags: list[str]
    summary: WindowSummary
    samples: list[dict] = field(default_factory=list)  # raw PID samples, for export/sim


def check_summary(summary: WindowSummary, wheels_off: bool = False, samples: list[dict] = None) -> TelemetryVerdict:
    """
    Deterministic pass over a WindowSummary. Returns flags describing
    anything worth mentioning. `wheels_off` downgrades the "commanded
    but no measured velocity" case from a fault to an expected bench-test
    signature (drive signal responded, no load feedback since wheels
    are detached).
    """
    flags: list[str] = []

    if summary.sample_count == 0 and not summary.warnings and not summary.errors:
        flags.append("no_telemetry_received")

    if summary.errors:
        flags.append(f"esp32_error: {', '.join(summary.errors)}")

    if summary.vel_never_left_zero:
        if wheels_off:
            flags.append(
                "wheels_off_expected: drive signal (pwm) responded to command, "
                "no measured velocity because wheels are detached"
            )
        else:
            flags.append(
                "possible_stall_or_encoder_fault: commanded velocity nonzero "
                "but measured velocity stayed at zero for the whole window"
            )

    # PWM saturating near max while velocity still isn't tracking setpoint
    # is worth a flag on its own, independent of the "never left zero" case,
    # since it also covers partial/degraded response.
    PWM_MAX = 255
    if summary.sample_count > 0:
        pwm_near_max = summary.max_pwmA >= PWM_MAX - 10 or summary.max_pwmB >= PWM_MAX - 10
        tracking_poorly = (
            abs(summary.mean_spA) > 1e-6 and
            abs(summary.mean_velA - summary.mean_spA) > 0.5 * abs(summary.mean_spA)
        )
        if pwm_near_max and tracking_poorly and not wheels_off:
            flags.append("pwm_saturating_without_tracking: PID output near max but not reaching setpoint")

    if summary.watchdog_stopped:
        # Not inherently bad — this is the firmware's own safety cutoff after
        # ~15s without a fresh cmd_vel. Reported as informational, not a fault,
        # unless it happened mid-command (caller can decide based on context).
        flags.append("watchdog_stop_observed")

    ok = not any(f.startswith(("possible_stall", "esp32_error", "no_telemetry", "pwm_saturating")) for f in flags)
    return TelemetryVerdict(ok=ok, flags=flags, summary=summary, samples=samples or [])


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: export raw samples for simulation / offline analysis
# ─────────────────────────────────────────────────────────────────────────────

def export_samples_csv(samples: list[dict], filepath: str) -> int:
    """
    Write raw PID samples to a CSV: t, dt, velA, velB, spA, spB, pwmA, pwmB.
    Generic time-series format — usable as input for a step-response
    analysis (forward/backward) or later folded into a tracking/LQR
    analysis (follow_me) without changing the save format.

    Returns the number of rows written.
    """
    import csv
    fieldnames = ["ts", "dt", "velA", "velB", "spA", "spB", "pwmA", "pwmB"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow({k: s.get(k) for k in fieldnames})
    return len(samples)


def compute_step_response_metrics(samples: list[dict], settle_tolerance: float = 0.05) -> dict:
    """
    Classic step-response metrics for a single motor's velocity trace,
    computed from the SAME sample list export_samples_csv writes.
    This is what's actually being tested by a plain "move forward" /
    "move backward" command — a step input on the PID setpoint.

    Only meaningful when spA/spB hold a roughly constant nonzero value
    for the whole window (a single step), not for multi-command windows.
    """
    if not samples:
        return {"error": "no_samples"}

    t0 = samples[0]["ts"]
    setpoint = samples[-1]["spA"]  # assume A/B symmetric for straight-line moves
    if abs(setpoint) < 1e-6:
        return {"error": "no_nonzero_setpoint_in_window"}

    rise_time = None
    settle_time = None
    max_vel = 0.0

    for s in samples:
        t = s["ts"] - t0
        v = s["velA"]
        max_vel = max(max_vel, abs(v))

        if rise_time is None and abs(v) >= 0.9 * abs(setpoint):
            rise_time = t

        band = settle_tolerance * abs(setpoint)
        if abs(abs(v) - abs(setpoint)) <= band:
            if settle_time is None:
                settle_time = t
        else:
            settle_time = None  # left the band again — reset

    overshoot_pct = max(0.0, (max_vel - abs(setpoint)) / abs(setpoint) * 100.0)
    final_vel = samples[-1]["velA"]
    steady_state_error = abs(setpoint) - abs(final_vel)

    return {
        "setpoint": setpoint,
        "rise_time_s": round(rise_time, 3) if rise_time is not None else None,
        "settle_time_s": round(settle_time, 3) if settle_time is not None else None,
        "overshoot_pct": round(overshoot_pct, 1),
        "steady_state_error": round(steady_state_error, 4),
        "reached_setpoint": rise_time is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: plot a saved rover_log CSV (step-response view)
# Separate from the LQR sim's diagnostic.png — this is for real hardware
# telemetry, not simulation output.
# ─────────────────────────────────────────────────────────────────────────────

def plot_rover_log(csv_path: str, out_png_path: str) -> str:
    """
    Reads a rover_log_*.csv (as written by export_samples_csv) and renders
    a step-response style plot: velocity vs setpoint on top, PWM on bottom.
    Returns the path written, or raises on failure (caller handles the message).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV has no rows to plot.")

    t0 = df["ts"].iloc[0]
    t = df["ts"] - t0

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Rover response — {csv_path}", fontsize=12)

    axes[0].plot(t, df["velA"], label="velA (measured)", color="tab:blue")
    axes[0].plot(t, df["spA"], "--", label="spA (setpoint)", color="tab:blue", alpha=0.5)
    axes[0].plot(t, df["velB"], label="velB (measured)", color="tab:orange")
    axes[0].plot(t, df["spB"], "--", label="spB (setpoint)", color="tab:orange", alpha=0.5)
    axes[0].set_ylabel("velocity (m/s)")
    axes[0].set_title("Wheel velocity vs setpoint")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, df["pwmA"], label="pwmA", color="tab:blue")
    axes[1].plot(t, df["pwmB"], label="pwmB", color="tab:orange")
    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel("PWM output")
    axes[1].set_title("Drive signal")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=140)
    plt.close(fig)
    return out_png_path

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: estimate trajectory from the COMMAND SEQUENCE, not telemetry
#
# Earlier version integrated spA/spB directly, assuming they were signed
# per-wheel setpoints. Real hardware data disproved this: spA/spB from the
# ESP32's own PID debug print are unsigned magnitudes (same value for both
# wheels even mid-turn) -- direction is handled elsewhere in firmware and
# is NOT present in this telemetry. So there is no reliable way to recover
# turn direction from spA/spB after the fact.
#
# Fix: use the command Buddy actually issued (action/direction/speed_percent,
# known authoritatively at send time -- no inference needed) run through the
# SAME kinematics rover_server.py's command_to_twist already uses on the Pi,
# then integrate over each command's duration. This has zero ambiguity.
# ─────────────────────────────────────────────────────────────────────────────

MAX_SPEED_MPS = 0.5   # must match rover_server.py's MAX_SPEED_MPS constant
TURN_STYLE = "arc"    # must match rover_server.py's TURN_STYLE constant


def _command_to_twist(action: str, direction: str, speed_pct: int, wheel_separation: float) -> tuple[float, float]:
    """
    Mirrors rover_server.py's command_to_twist exactly, so the estimate
    always matches what the Pi actually computed and sent to /cmd_vel.
    Returns (linear_x, angular_z).
    """
    v = (speed_pct / 100.0) * MAX_SPEED_MPS

    if action == "stop" or direction == "stop":
        return 0.0, 0.0

    if action == "move":
        if direction == "forward":
            return v, 0.0
        if direction in ("backward", "back"):
            return -v, 0.0
        return 0.0, 0.0

    if action == "turn":
        ang = v / (wheel_separation / 2.0)
        lin = v / 2.0 if TURN_STYLE == "arc" else 0.0
        if direction == "left":
            return lin, +ang
        if direction == "right":
            return lin, -ang
        return 0.0, 0.0

    return 0.0, 0.0


def estimate_trajectory_from_commands(
    command_segments: list[dict],
    wheel_separation: float = 0.25,
) -> list[dict]:
    """
    command_segments: list of {"action", "direction", "speed_percent", "duration_s"}
    -- exactly what RoverTaskManager already knows when it issues each command,
    no telemetry parsing involved. Integrates unicycle kinematics per segment.

    Returns [{"t", "x", "y", "psi"}, ...] with one point per segment boundary
    plus fine-grained interpolation within each segment.
    """
    if not command_segments:
        return []

    x, y, psi = 0.0, 0.0, 0.0
    t = 0.0
    path = [{"t": 0.0, "x": 0.0, "y": 0.0, "psi": 0.0}]

    STEP = 0.1  # seconds per integration step within a segment

    for seg in command_segments:
        lin, ang = _command_to_twist(
            seg.get("action", ""), seg.get("direction", ""),
            seg.get("speed_percent", 0), wheel_separation,
        )
        duration = max(0.0, float(seg.get("duration_s", 0.0)))
        n_steps = max(1, int(duration / STEP))
        dt = duration / n_steps if n_steps else 0.0

        for _ in range(n_steps):
            x += lin * dt * math.cos(psi)
            y += lin * dt * math.sin(psi)
            psi += ang * dt
            t += dt
            path.append({"t": round(t, 3), "x": round(x, 5), "y": round(y, 5), "psi": round(psi, 5)})


    return path


def plot_estimated_trajectory(command_segments: list[dict], out_png_path: str, wheel_separation: float = 0.25) -> str:
    """
    Plots the estimated top-down path from the COMMAND SEQUENCE (not
    telemetry -- see module note above for why). Labelled clearly as an
    estimate derived from commanded intent, not a physical measurement.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = estimate_trajectory_from_commands(command_segments, wheel_separation)
    if len(path) < 2:
        raise ValueError("Not enough command history to estimate a trajectory.")

    xs = [p["x"] for p in path]
    ys = [p["y"] for p in path]
    ts = [p["t"] for p in path]
    psis = [p["psi"] for p in path]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Estimated path from COMMAND SEQUENCE (not a physical measurement)", fontsize=11)

    axes[0].plot(xs, ys, color="tab:green")
    axes[0].scatter([xs[0]], [ys[0]], color="black", zorder=5, label="start")
    axes[0].scatter([xs[-1]], [ys[-1]], color="red", zorder=5, label="end")
    axes[0].set_xlabel("X (m, estimated)")
    axes[0].set_ylabel("Y (m, estimated)")
    axes[0].set_title("Estimated top-down path")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8)

    axes[1].plot(ts, [__import__("math").degrees(p) for p in psis], color="tab:purple")
    axes[1].axhline(0, color="gray", linestyle=":", linewidth=1)
    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel("estimated heading (deg)")
    axes[1].set_title("Heading over time (from commanded turns)")

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=140)
    plt.close(fig)
    return out_png_path
"""
One entry point for the evaluation.

  python run_eval.py unit               fast offline checks (no LLM, no hardware); this is what CI runs
  python run_eval.py alive [options]    the conversational scenario suite (needs Ollama + llama3.2);
                                        options are passed to buddy_alive_test_runner.py, e.g.
                                        --isolate  --trace-prompts  --category memory_management  --output out.json
  python run_eval.py results            print the committed results (results/stress_runs.json)
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UNIT_SCRIPTS = ["eval/test_context_guards.py", "eval/test_save_target.py"]


def unit() -> int:
    failed = 0
    for script in UNIT_SCRIPTS:
        print(f"\n=== {script}")
        failed += subprocess.call([sys.executable, str(ROOT / script)], cwd=ROOT) != 0
    print("\nALL UNIT CHECKS PASSED" if not failed else f"\n{failed} UNIT SCRIPT(S) FAILED")
    return 1 if failed else 0


def alive(extra: list) -> int:
    args = [sys.executable, str(ROOT / "buddy_alive_test_runner.py")]
    if "--test-file" not in extra:
        args += ["--test-file", str(ROOT / "scenarios" / "buddy_tony_stark_tests.json")]
    return subprocess.call(args + extra, cwd=ROOT)


def results() -> int:
    data = json.loads((ROOT / "results" / "stress_runs.json").read_text(encoding="utf-8"))
    print(f"{'run':34s}{'mode':12s}{'passed':>8s}{'rate':>6s}{'alive':>7s}{'bleed':>6s}  note")
    for r in data["runs"]:
        print(f"{r['run']:34s}{r['mode']:12s}{r['passed']:>4d}/{r['total']:<3d}{r['pass_rate']:>6s}{r['alive_score']:>7.2f}"
              f"{str(r['context_bleed_turns']) if r['context_bleed_turns'] is not None else '-':>6s}  {r['note']}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "unit":
        sys.exit(unit())
    if cmd == "alive":
        sys.exit(alive(sys.argv[2:]))
    if cmd == "results":
        sys.exit(results())
    print(__doc__)
    sys.exit(2)

"""
eval/test_save_target.py

Regression test for score_save_target(): a bare "save it" must not be hijacked into
"save the rover telemetry log" when the rover has no telemetry to save (rover offline,
or the topic merely mentions "rover"). Found by the alive stress test:

    memory_save_recall_flow, turn 3:  "save it"  ->  "I don't have any recent telemetry to save yet."

Lightweight on purpose: it extracts only the functions under test from
brain/conversation_manager.py (a ~8000-line file that imports the whole Buddy stack),
so it needs no Ollama, no audio, no rover.

Run with:
    pytest eval/test_save_target.py -v
    python eval/test_save_target.py
"""
import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "brain" / "conversation_manager.py"
_WANTED_FUNCS = {"_rover_has_telemetry", "score_save_target"}
_WANTED_ASSIGNS = {"ROVER_SAVE_SIGNALS", "CONVERSATION_SAVE_SIGNALS"}


def _load_under_test() -> dict:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED_FUNCS:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in _WANTED_ASSIGNS for t in node.targets
        ):
            keep.append(node)
    ns: dict = {"re": re}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SRC), "exec"), ns)
    missing = (_WANTED_FUNCS | _WANTED_ASSIGNS) - set(ns)
    assert not missing, f"could not find in {SRC.name}: {missing}"
    return ns


NS = _load_under_test()
score_save_target = NS["score_save_target"]


class FakeMM:
    def __init__(self, last_topic=None):
        self._t = last_topic

    def get_last_topic(self):
        return self._t


def rover(has_telemetry: bool, session_active: bool = False):
    if has_telemetry:
        return SimpleNamespace(
            session_active=session_active,
            _last_telemetry_verdict=SimpleNamespace(samples=[1, 2, 3]),
            _session_samples=[1, 2, 3],
        )
    return SimpleNamespace(session_active=session_active, _last_telemetry_verdict=None, _session_samples=[])


class SaveTargetTests(unittest.TestCase):
    # ── the bug ────────────────────────────────────────────────────────────────
    def test_bare_save_in_rover_topic_without_telemetry_is_not_rover_log(self):
        r = score_save_target("save it", FakeMM("rover:session"), rover(False))
        self.assertEqual(r["target"], "conversation")

    def test_bare_save_with_rover_session_active_but_no_samples(self):
        r = score_save_target("save it", FakeMM("rover:session"), rover(False, session_active=True))
        self.assertEqual(r["target"], "conversation")

    def test_vague_weak_signal_without_telemetry_is_conversation_not_ambiguous(self):
        # one weak rover word ("data") and nothing to save: don't ask "rover log or conversation?"
        r = score_save_target("save the data", FakeMM(None), rover(False))
        self.assertEqual(r["target"], "conversation")

    # ── behaviour that must NOT change ─────────────────────────────────────────
    def test_bare_save_in_rover_topic_with_telemetry_still_rover_log(self):
        r = score_save_target("save it", FakeMM("rover:session"), rover(True))
        self.assertEqual(r["target"], "rover_log")

    def test_explicit_rover_log_request_without_telemetry_still_routes_to_rover(self):
        # the user asked for the log by name: let the rover handler answer honestly
        r = score_save_target("save the rover telemetry log", FakeMM(None), rover(False))
        self.assertEqual(r["target"], "rover_log")

    def test_explicit_conversation_save_unchanged(self):
        for has in (True, False):
            r = score_save_target("save our conversation", FakeMM("rover:session"), rover(has))
            self.assertEqual(r["target"], "conversation")

    def test_no_rover_manager_at_all(self):
        r = score_save_target("save it", FakeMM("rover:session"), None)
        self.assertEqual(r["target"], "conversation")

    def test_with_telemetry_decision_matches_original_logic_exactly(self):
        """When there IS telemetry the new code must behave exactly like the old scoring."""
        def old(user_input, mm, rm):
            text = user_input.lower(); lt = (mm.get_last_topic() or "").strip()
            rh = sum(1 for w in NS["ROVER_SAVE_SIGNALS"] if w in text)
            ch = sum(1 for w in NS["CONVERSATION_SAVE_SIGNALS"] if w in text)
            prior = 0.35 if (lt.startswith("rover:") or getattr(rm, "session_active", False)) else 0.0
            rs, cs = min(1.0, rh * 0.3 + prior), min(1.0, ch * 0.3)
            if rs >= 0.5 and rs - cs >= 0.2: return "rover_log"
            if cs >= 0.5 and cs - rs >= 0.2: return "conversation"
            if prior > 0 and rh == 0 and ch == 0: return "rover_log"
            return "ambiguous"

        phrases = ["save it", "save this", "save the log", "store this data", "save our conversation",
                   "save the notes", "export the run", "save the telemetry csv", "keep this chat", "save"]
        for topic in (None, "rover:session", "kalman filter"):
            for active in (False, True):
                for p in phrases:
                    rm = rover(True, session_active=active)
                    self.assertEqual(score_save_target(p, FakeMM(topic), rm)["target"], old(p, FakeMM(topic), rm),
                                     f"changed behaviour with telemetry: {p!r} topic={topic!r} active={active}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

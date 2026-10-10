"""
buddy_alive_test_runner.py

Runs the alive test suite against Buddy's ConversationManager and
produces a score report showing exactly where Buddy feels dead.

Usage:
    python buddy_alive_test_runner.py

    # Run specific category only:
    python buddy_alive_test_runner.py --category vision_awareness

    # Run with vision awareness scorer debug output:
    python buddy_alive_test_runner.py --debug-vision

    # Save report to file:
    python buddy_alive_test_runner.py --output report.json

    # Diagnostic: wipe all session memory between scenarios. Run it next to a normal run -
    # a scenario that fails in the normal run but passes isolated was hit by context bleed.
    python buddy_alive_test_runner.py --isolate --output report_isolated.json

    # Diagnostic: capture the exact LLM prompts. Every context-bleed turn then states which
    # prompt section (belief block, conversation-so-far, background context...) held the leak.
    python buddy_alive_test_runner.py --trace-prompts --output report_traced.json

Scenario JSON extras (all optional):
    "must_not_contain": [...]        scenario level, applied to every turn
    "bleed_allow": [...]             scenario or turn level: terms allowed to appear even though
                                     another scenario owns them (see context-bleed detection)

The runner does NOT start audio/VAD/TTS — it talks directly to
ConversationManager.handle_request() so tests run fast and silent.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import re
import sys
import json
import time
import argparse
import traceback
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from brain.conversation_manager import _seed_session_context

# ── Patch sys.path so imports resolve the same way main.py does ──────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Lazy import of Buddy internals (so we can report import errors cleanly) ──
def _context_guards_state() -> str:
    """Recorded in every report so A/B runs (BUDDY_CONTEXT_GUARDS=0 vs default) are distinguishable afterwards."""
    try:
        from brain.utils.context_guards import guards_enabled
        return "on" if guards_enabled() else "off"
    except Exception:
        return "unknown"


def _import_buddy():
    try:
        from brain.core.brain import MemoryManager
        from brain.core.devices.device_handler import DeviceHandlerFSM
        from brain.identity.identity_manager import IdentityManager
        from brain.core.internet_access.internet_service import get_explorer
        from brain.conversation_manager import ConversationManager
        return MemoryManager, IdentityManager, get_explorer, ConversationManager
    except Exception as e:
        print(f"[IMPORT ERROR] Could not import Buddy internals: {e}")
        traceback.print_exc()
        sys.exit(1)

# ── Vision awareness scorer (optional — skip if not yet present) ─────────────
def _import_vision_scorer():
    try:
        from brain.adaptive.vision.vision_awareness import score_vision_need
        return score_vision_need
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn:            int
    input:           str
    response:        str
    response_time_ms: float
    checks_passed:   list[str] = field(default_factory=list)
    checks_failed:   list[str] = field(default_factory=list)
    warnings:        list[str] = field(default_factory=list)
    vision_score:    Optional[float] = None
    alive_score:     float = 0.0        # 0.0 – 1.0 for this turn
    prompts:         list = field(default_factory=list)   # LLM prompts sent this turn (--trace-prompts)


@dataclass
class ScenarioResult:
    id:           str
    category:     str
    description:  str
    turns:        list[TurnResult] = field(default_factory=list)
    passed:       bool = False
    alive_score:  float = 0.0           # average across turns
    fatal_fails:  list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Context-bleed detection
# ─────────────────────────────────────────────────────────────────────────────
# "Bleed" = Buddy answers with ANOTHER scenario's topic (stale last_topic / short-term
# memory), e.g. a reply about MPPI/Nav2 to "difference between RL and imitation learning".
# Keyword checks can't see it: the reply can still contain an expected word like "rover"
# and pass. So we look for terms that belong to exactly ONE other scenario's inputs.
#
# Precision matters more than recall here (a noisy detector would make the harness less
# trustworthy), so:
#   * only entity-like terms count (acronyms, names, anything with a digit);
#   * GENERIC_TECH_TERMS are ignored - any good answer may mention them unprompted;
#   * recall questions ("what did we discuss...") are exempt, they legitimately list topics;
#   * a turn/scenario can whitelist a term with  "bleed_allow": ["term", ...]  in its checks.

_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_COMMON_CAPITALISED = {
    "i", "buddy", "friday", "monday", "tuesday", "wednesday", "thursday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "sir", "okay", "ok", "yes", "no", "hey", "actually",
    "oh", "so", "the", "and", "but",
}
GENERIC_TECH_TERMS = {
    "kalman", "pid", "slam", "ekf", "mpc", "lqr", "ros", "lidar", "imu", "gps", "pwm",
    "rover", "odometry", "controller", "navigation",
}
_RECALL_QUESTION_RE = re.compile(
    r"\bwhat (did|have) we (discuss|talk|cover|say)|\bwhat do you remember\b|\bremind me what we\b",
    re.IGNORECASE,
)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _entity_tokens(text: str) -> set:
    """Acronyms, names and anything with a digit. Capitalised words only count when they are
    NOT the first word of a sentence ("However", "Specifically" are just grammar)."""
    out = set()
    for sentence in _SENTENCE_SPLIT_RE.split(text or ""):
        toks = _TERM_RE.findall(sentence)
        for i, t in enumerate(toks):
            low = t.lower()
            if len(low) < 3 or low in _COMMON_CAPITALISED:
                continue
            if any(c.isdigit() for c in t) or (t.isupper() and len(t) >= 3) or (t.istitle() and i > 0):
                out.add(low)
    return out


def learn_reply_terms(replies_by_scenario: dict, min_replies: int = 2) -> dict:
    """
    Terms that BUDDY ITSELF introduced (the inputs never said them) and keeps coming back to
    in exactly one finished scenario, e.g. "DDPG" after an RL scenario. If one of these
    shows up in a different scenario's reply, an old topic is leaking in.
    min_replies guards against one-off incidental words: the term must appear in at least
    that many replies of its owner scenario.
    Returns {term: owning_scenario_id}.
    """
    owned = {}
    for sid, replies in replies_by_scenario.items():
        counts = Counter()
        for reply in replies:
            counts.update(_entity_tokens(reply) - GENERIC_TECH_TERMS)
        owned[sid] = {t for t, n in counts.items() if n >= min_replies}
    df = Counter(t for terms in owned.values() for t in terms)
    return {t: sid for sid, terms in owned.items() for t in terms if df[t] == 1}


def _scenario_text(s: dict) -> str:
    parts = [s.get("description", "")]
    parts += s.get("bleed_allow", [])
    for t in s.get("turns", []):
        parts.append(t.get("input", ""))
        ac = t.get("alive_checks", {})
        parts += ac.get("must_contain_one_of", []) + [ac.get("note", "")] + ac.get("bleed_allow", [])
    return " ".join(parts)


def build_bleed_terms(scenarios: list) -> dict:
    """scenario id -> {term: owning_scenario_id} for terms that belong to exactly one OTHER scenario."""
    own = {}
    for s in scenarios:
        toks = set()
        for t in s.get("turns", []):
            toks |= _entity_tokens(t.get("input", ""))
        own[s["id"]] = toks - GENERIC_TECH_TERMS
    df = Counter(tok for toks in own.values() for tok in toks)
    foreign = {}
    for s in scenarios:
        mine = set(_TERM_RE.findall(_scenario_text(s).lower()))
        foreign[s["id"]] = {
            tok: owner
            for owner, toks in own.items() if owner != s["id"]
            for tok in toks if df[tok] == 1 and tok not in mine
        }
    return foreign


def find_bleed(response: str, user_input: str, forbidden: dict) -> list:
    if not forbidden or _RECALL_QUESTION_RE.search(user_input or ""):
        return []
    resp = set(_TERM_RE.findall((response or "").lower()))
    inp = set(_TERM_RE.findall((user_input or "").lower()))
    return sorted((t, forbidden[t]) for t in forbidden if t in resp and t not in inp)


_SECTION_RE = re.compile(r"^\[[^\]\n]{2,120}\]", re.MULTILINE)


def locate_in_prompts(prompts: list, term: str) -> list:
    """
    Where in the prompts sent to the LLM does `term` appear? Returns the section headers
    that contain it, e.g. ['[Conversation so far]', '[BACKGROUND CONTEXT: x | similarity: 0.54]'].
    This is what turns "Buddy leaked topic X" into "...because X was in the prompt section Y".
    """
    pat = re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", re.IGNORECASE)
    found = set()
    for p in prompts:
        heads = [(m.start(), m.group(0)) for m in _SECTION_RE.finditer(p)]
        for m in pat.finditer(p):
            header = "[prompt start]"
            for pos, h in heads:
                if pos > m.start():
                    break
                header = h
            found.add(header)
    return sorted(found)


def _has_phrase(text: str, phrase: str) -> bool:
    """Whole-word phrase match. A plain substring test let "ah" match inside "ahead"/"yeah"."""
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Core checker
# ─────────────────────────────────────────────────────────────────────────────

class AliveChecker:
    """
    Evaluates a single response against the alive_checks spec from the JSON.
    Returns (passed_list, failed_list, warnings_list, score 0-1).
    """

    # Phrases that indicate Buddy is dead / stuck
    DEAD_PHRASES = [
        "let's start over",
        "i'm having trouble with that",
        "having trouble with that",
        "i don't understand",
        "i'm not sure what you mean",
        "could you rephrase",
        "i got confused",
        "something went wrong",
    ]

    # Phrases that indicate natural alive engagement
    ALIVE_PHRASES = [
        "ah", "i see", "makes sense", "got it", "understood",
        "based on what i can see", "looking at your setup",
        "from what i'm seeing", "i can see that",
        "taking a look", "let me check",
    ]

    def check(
        self,
        response: str,
        checks: dict,
        vision_score: Optional[float] = None,
    ) -> tuple[list, list, list, float]:
        passed = []
        failed = []
        warnings = []
        score_parts = []

        resp_lower = response.lower().strip()
        if resp_lower.startswith("[error:") or "traceback" in resp_lower:
            failed.append("Runtime error returned as response")
            score_parts.append(0.0)

        # ── must_not_contain ─────────────────────────────────────────────────
        for phrase in checks.get("must_not_contain", []):
            if phrase.lower() in resp_lower:
                failed.append(f"DEAD_PHRASE found: '{phrase}'")
                score_parts.append(0.0)
            else:
                passed.append(f"No dead phrase: '{phrase}'")
                score_parts.append(1.0)

        # ── must_contain_one_of ──────────────────────────────────────────────
        required_one = checks.get("must_contain_one_of", [])
        if required_one:
            found = [p for p in required_one if p.lower() in resp_lower]
            if found:
                passed.append(f"Expected phrase found: {found[0]!r}")
                score_parts.append(1.0)
            else:
                failed.append(f"None of expected phrases found: {required_one}")
                score_parts.append(0.0)

        # ── response length check ────────────────────────────────────────────
        max_len = checks.get("max_response_length")
        if max_len:
            if len(response) <= max_len:
                passed.append(f"Response length OK ({len(response)} <= {max_len})")
                score_parts.append(1.0)
            else:
                failed.append(f"Response too long ({len(response)} > {max_len})")
                score_parts.append(0.3)

        # ── vision awareness score checks ────────────────────────────────────
        if vision_score is not None:
            min_vs = checks.get("score_vision_awareness_min")
            max_vs = checks.get("score_vision_awareness_max")

            if min_vs is not None:
                if vision_score >= min_vs:
                    passed.append(f"Vision score {vision_score:.2f} >= {min_vs}")
                    score_parts.append(1.0)
                else:
                    failed.append(
                        f"Vision score too low: {vision_score:.2f} < {min_vs} "
                        f"(camera should have been offered/triggered)"
                    )
                    score_parts.append(0.0)

            if max_vs is not None:
                if vision_score <= max_vs:
                    passed.append(f"Vision score {vision_score:.2f} <= {max_vs} (correctly suppressed)")
                    score_parts.append(1.0)
                else:
                    failed.append(
                        f"Vision score too high: {vision_score:.2f} > {max_vs} "
                        f"(camera should NOT have triggered here)"
                    )
                    score_parts.append(0.0)

        # ── should_offer_camera ──────────────────────────────────────────────
        offer_camera = checks.get("should_offer_camera")
        if offer_camera is True:
            CAMERA_OFFER_SIGNALS = [
                "take a look", "have a look", "look at your",
                "check what", "see your", "want me to look",
                "should i look", "want me to see",
                "based on what i can see", "looking at your setup",
                "from what i'm seeing",
            ]
            found = any(s in resp_lower for s in CAMERA_OFFER_SIGNALS)
            if found:
                passed.append("Camera offer or vision response detected")
                score_parts.append(1.0)
            else:
                # Not a hard fail — might be silent grab
                warnings.append("Camera offer expected but not detected in response text")
                score_parts.append(0.5)

        elif offer_camera is False:
            CAMERA_OFFER_SIGNALS = [
                "take a look", "have a look", "look at your",
                "want me to look", "should i look",
            ]
            found = any(s in resp_lower for s in CAMERA_OFFER_SIGNALS)
            if not found:
                passed.append("Camera correctly NOT offered")
                score_parts.append(1.0)
            else:
                failed.append("Camera was offered but should not have been")
                score_parts.append(0.0)

        # ── implicit alive signal check (always run) ─────────────────────────
        has_dead = any(p in resp_lower for p in self.DEAD_PHRASES)
        has_alive = any(_has_phrase(resp_lower, p) for p in self.ALIVE_PHRASES)

        if has_dead:
            warnings.append("Response contains a dead/stuck phrase even if not in explicit checks")
        if has_alive:
            passed.append("Response contains natural alive engagement phrase")
            score_parts.append(1.0)

        # ── compute turn score ───────────────────────────────────────────────
        alive_score = (sum(score_parts) / len(score_parts)) if score_parts else 0.5

        # Apply penalty for dead phrases regardless of other scores
        if has_dead and not checks.get("must_not_contain"):
            alive_score = max(0.0, alive_score - 0.25)

        return passed, failed, warnings, round(alive_score, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

class BuddyAliveTestRunner:

    trace_prompts = False  # default; __init__ overrides it (also keeps __init__-less test doubles working)

    def __init__(self, debug_vision: bool = False, isolate: bool = False, trace_prompts: bool = False):
        self.debug_vision = debug_vision
        self.trace_prompts = trace_prompts
        self._prompt_log: list = []
        if trace_prompts:
            self._install_prompt_tracer()
        # isolate=False ("continuous", the default) keeps one long session across scenarios,
        # which is how Buddy is really used and is what exposes context bleed.
        # isolate=True wipes ALL session memory between scenarios: a diagnostic to tell
        # "this scenario is broken" apart from "an earlier topic leaked into it".
        self.isolate = isolate
        self._bleed_terms: dict = {}
        self._replies: dict = {}   # scenario id -> Buddy's replies, for learning leaked terms
        self._bleed_events: list = []
        self._current_bleed_terms: dict = {}
        self._scenario_bleed_allow: set = set()
        self._scenario_must_not: list = []
        self.checker = AliveChecker()
        self.score_vision_need = _import_vision_scorer()
        import unittest.mock as mock
        # No real camera in tests. conversation_manager imports the capture functions by name,
        # so they are patched there too; patching only vision.webcam_capture does not reach it.
        self._webcam_patchers = [
            mock.patch(f"{module}.{name}", return_value=value)
            for module in ("vision.webcam_capture", "brain.conversation_manager")
            for name, value in (("capture_single_frame", None), ("capture_burst_frames", []))
        ]
        for p in self._webcam_patchers:
            p.start()
        self._mock = mock
        self._active_patchers = []

        MemoryManager, IdentityManager, get_explorer, ConversationManager = _import_buddy()

        print("[SETUP] Initialising Buddy ConversationManager...")
        try:
            identity = IdentityManager()
            explorer = get_explorer(enabled=False)  # disable real web calls during test
            mm = MemoryManager()
            self.cm = ConversationManager(
                mm,
                tello_client=None,
                explorer=explorer,
                identity=identity,
            )
            print("[SETUP] Ready.\n")
        except Exception as e:
            print(f"[SETUP ERROR] {e}")
            traceback.print_exc()
            sys.exit(1)

    def _install_prompt_tracer(self):
        """Record every prompt Buddy sends to Ollama (module-level ollama.generate / ollama.chat)."""
        import ollama
        orig_generate, orig_chat = ollama.generate, ollama.chat

        def generate(*a, **k):
            self._prompt_log.append(str(k.get("prompt", a[1] if len(a) > 1 else "")))
            return orig_generate(*a, **k)

        def chat(*a, **k):
            msgs = k.get("messages", a[1] if len(a) > 1 else [])
            self._prompt_log.append("\n".join(f"<{m.get('role')}>\n{m.get('content')}" for m in msgs))
            return orig_chat(*a, **k)

        ollama.generate, ollama.chat = generate, chat

    # Topics that scenario setups seed into temp_topic_memory; dropped between scenarios.
    # (This set used to be a literal `{...}` placeholder, so nothing but "test:"-prefixed
    # keys was ever cleared and every scenario inherited the previous ones' topics.)
    _SEEDED_TOPICS = {
        "llm episodic memory", "robotic arm", "sql optimization", "rover:session",
        "web:research", "vision:bench", "session:weather", "session:news", "session:schedule",
    }

    def _reset_session(self):
        try:
            mm = self.cm.memory_manager
            if self.isolate:
                mm.reset_session_state()  # pending, last_topic, flow stack, web contexts, ...
            else:
                mm.set_pending_topic(None)
                mm.pending_payload = None
                mm.set_last_topic(None)
            mm._stuck_count = 0
            mm._last_pending = None

            # ── Reset rover state so each scenario starts from a clean session ──
            rover = getattr(self.cm.device_handler, "rover_manager", None)
            if rover is not None:
                try:
                    rover.watcher.stop()
                except Exception:
                    pass
                rover.session_active = False
                rover.current_direction = "stop"
                rover.current_speed = 0
                rover.last_updated = 0.0

            from brain.core.brain import temp_topic_memory, sensory_memory
            if self.isolate:
                temp_topic_memory.clear()
                sensory_memory.clear()
                try:
                    from brain.belief_state import BeliefState
                    self.cm.belief.state = BeliefState()
                except Exception:
                    pass
            else:
                for key in list(temp_topic_memory.keys()):
                    if key.startswith("test:") or key in self._SEEDED_TOPICS:
                        temp_topic_memory.pop(key, None)

            self.cm._shadow_prev_turn = None  # shadow classifier's previous-turn context
        except Exception as e:
            print(f"    [RESET WARNING] session reset incomplete: {type(e).__name__}: {e}")

    def _setup_scenario(self, scenario: dict):
        for p in self._active_patchers:
            try:
                p.stop()
            except Exception:
                pass
        self._active_patchers = []

        setup = scenario.get("setup", {})
        if not setup:
            return

        # ── seed_temp_topic_memory ─────────────────────────────────────────────
        seed_memory = setup.get("seed_temp_topic_memory")
        if seed_memory:
            from brain.core.brain import temp_topic_memory

            for topic, exchanges in seed_memory.items():
                temp_topic_memory[topic] = exchanges

            print(f"    [SETUP] Seeded temp_topic_memory topics: {list(seed_memory.keys())}")

        # ── set_last_topic ────────────────────────────────────────────────────
        if setup.get("set_last_topic") is not None:
            last_topic = setup.get("set_last_topic")
            self.cm.memory_manager.set_last_topic(last_topic)
            print(f"    [SETUP] last_topic set to: {last_topic!r}")

        # ── seed_session_context ─────────────────────────────────────────────
        if setup.get("seed_session_context"):
            from brain.core.brain import temp_topic_memory, store_short_term_memory
            mock_intro = (
                "Good morning sir. As of Monday, April 10, 2026 — 08:30. "
                "Weather status: Clear, 12°C now. Range: 8–15°C. "
                "Here are the latest developments: "
                "ECB holds rates steady amid eurozone slowdown. "
                "New EU AI Act enforcement guidelines published. "
                "Standing by."
            )
            _seed_session_context(mock_intro, self.cm.memory_manager)
            print(f"    [SETUP] Session context seeded from mock intro")

        # ── mock_note_result ─────────────────────────────────────────────────
        note_cfg = setup.get("mock_note_result")
        if note_cfg:
            from pathlib import Path

            # Build a proper result dict matching find_latest_note_smart output
            path_val = Path(note_cfg["path"]) if note_cfg.get("path") else None

            mock_result = {
                "path":           path_val,
                "confidence":     note_cfg.get("confidence", 1.0),
                "file_date":      note_cfg.get("file_date"),
                "spoken_name":    note_cfg.get("spoken_name", "Daily Note"),
                "ask_user":       note_cfg.get("ask_user", False),
                "open_explorer":  note_cfg.get("open_explorer", False),
                "all_candidates": [],
            }

            # Patch find_latest_note_smart in the project_handler module
            p1 = self._mock.patch(
                "brain.projects.project_handler.find_latest_note_smart",
                return_value=mock_result,
            )
            p1.start()
            self._active_patchers.append(p1)

            # Also patch read_note_sections and summarize_note so we don't
            # need real files on disk
            p2 = self._mock.patch(
                "brain.projects.project_handler.read_note_sections",
                return_value={
                    "date":        note_cfg.get("file_date", "2026-04-01"),
                    "objectives":  "Evaluate bandpass filter pipeline on 200ps dataset.",
                    "methodology": "Apply best hyperparameter setup from 100ps reflection as baseline.",
                    "notes":       "Key finding: single-shot deployment validated. Mark2 deferred.",
                },
            )
            p2.start()
            self._active_patchers.append(p2)

            p3 = self._mock.patch(
                "brain.projects.project_handler.summarize_note",
                return_value=(
                    "Your last session focused on the bandpass filter pipeline. "
                    "You were evaluating the 200ps dataset using the best setup "
                    "from 100ps reflection as baseline. "
                    "Key takeaway: single-shot deployment is validated."
                ),
            )
            p3.start()
            self._active_patchers.append(p3)

            # Patch NoteAnalyzer.run so it doesn't hit the real DB or PDF
            p4 = self._mock.patch(
                "brain.projects.note_analyzer.NoteAnalyzer.run",
                return_value=([], []),   # (todos, short_term_ctx)
            )
            p4.start()
            self._active_patchers.append(p4)

            p5 = self._mock.patch(
                "brain.projects.note_analyzer.NoteAnalyzer.build_session_briefing",
                return_value=(
                    "You have 7 open action items. "
                    "Top priorities: test model with single real shot signal, "
                    "document hyperparameter setup, and run transmission dataset analysis."
                ),
            )
            p5.start()
            self._active_patchers.append(p5)

            print(f"    [SETUP] Note mock: conf={mock_result['confidence']:.2f} "
                  f"ask_user={mock_result['ask_user']} "
                  f"open_explorer={mock_result['open_explorer']}")

        # ── mock_matlab_state ────────────────────────────────────────────────
        matlab_state = setup.get("mock_matlab_state")
        if matlab_state:
            p6 = self._mock.patch(
                "brain.projects.project_handler.check_matlab_state",
                return_value=matlab_state,
            )
            p6.start()
            self._active_patchers.append(p6)
            print(f"    [SETUP] MATLAB state mock: {matlab_state}")

        # ── mock_matlab_open ─────────────────────────────────────────────────
        matlab_open = setup.get("mock_matlab_open")
        if matlab_open is not None:
            p7 = self._mock.patch(
                "brain.projects.project_handler.open_matlab_at_folder",
                return_value=matlab_open,
            )
            p7.start()
            self._active_patchers.append(p7)

        # ── mock open_file_explorer (always mock — never open real explorer) ─
        p8 = self._mock.patch(
            "brain.projects.project_handler.open_file_explorer",
            return_value=True,
        )
        p8.start()
        self._active_patchers.append(p8)

    def _get_vision_score(self, user_input: str) -> Optional[float]:
        if not self.score_vision_need:
            return None
        try:
            result = self.score_vision_need(
                user_text=user_input,
                last_topic=self.cm.memory_manager.get_last_topic() or "",
                pending_state=self.cm.memory_manager.get_pending_topic() or "",
            )
            if self.debug_vision:
                print(f"    [VisionScore] {result.score:.2f} "
                      f"({result.decision.value}) — {result.reason}")
            return result.score
        except Exception as e:
            if self.debug_vision:
                print(f"    [VisionScore ERROR] {e}")
            return None

    def run_turn(self, turn_spec: dict, scenario_id: str) -> TurnResult:

        turn_num = turn_spec["turn"]
        user_input = turn_spec["input"]
        checks = dict(turn_spec.get("alive_checks", {}))
        # scenario-level "must_not_contain" applies to every turn of that scenario
        if self._scenario_must_not:
            checks["must_not_contain"] = list(checks.get("must_not_contain", [])) + self._scenario_must_not
        note = checks.get("note", "")

        print(f"  Turn {turn_num}: {user_input!r}")
        if note:
            print(f"    Note: {note}")

        self._prompt_log = []  # prompts sent to the LLM during THIS turn (--trace-prompts)
        if user_input == "__STARTUP_GREETING__":
            response = "[STARTUP GREETING — not testable via handle_request, check manually]"
            response_time = 0.0
            vision_score = None
        else:
            # Vision score BEFORE handle_request — captures correct context
            vision_score = self._get_vision_score(user_input)

            t0 = time.perf_counter()
            try:
                response = self.cm.handle_request(user_input)
                response = (response or "").strip()
            except Exception as e:
                response = f"[ERROR: {e}]"
                traceback.print_exc()
            response_time = (time.perf_counter() - t0) * 1000
        turn_prompts = list(self._prompt_log) if self.trace_prompts else []

        # Debug state AFTER handle_request
        pending_payload = getattr(self.cm.memory_manager, "pending_payload", None)

        if isinstance(pending_payload, dict):
            failure_ctx = pending_payload.get("failure_context")
        else:
            failure_ctx = getattr(pending_payload, "failure_context", None)

        print(
            f"    [DEBUG] last_topic={self.cm.memory_manager.get_last_topic()!r} "
            f"pending={self.cm.memory_manager.get_pending_topic()!r} "
            f"failure_ctx={failure_ctx!r}"
        )
        passed, failed, warnings, alive_score = self.checker.check(
            response, checks, vision_score
        )

        # ── Context bleed: did the reply talk about another scenario's topic? ──
        if user_input != "__STARTUP_GREETING__":
            allow = {a.lower() for a in checks.get("bleed_allow", [])} | self._scenario_bleed_allow
            bleed = [(t, o) for t, o in find_bleed(response, user_input, self._current_bleed_terms)
                     if t not in allow]
            if bleed:
                terms = ", ".join(f"{t!r} (from {o})" for t, o in bleed)
                # With --trace-prompts we can say WHERE the leaked term entered the LLM prompt.
                sources = {t: locate_in_prompts(turn_prompts, t) for t, _ in bleed} if turn_prompts else {}
                where = ""
                if sources:
                    flat = sorted({h for hs in sources.values() for h in hs})
                    where = (f"; the term was in the prompt under {', '.join(flat)}" if flat
                             else "; the term was NOT in any prompt (the LLM produced it on its own)")
                failed.append(f"CONTEXT_BLEED: reply talks about {terms}, never mentioned in this scenario{where}")
                alive_score = max(0.0, alive_score - 0.5)
                self._bleed_events.append({
                    "scenario": scenario_id, "turn": turn_num, "input": user_input,
                    "terms": [{"term": t, "from_scenario": o, **({"in_prompt_sections": sources[t]} if sources else {})}
                              for t, o in bleed],
                    "response_preview": response[:160],
                })

        # Debug state AFTER handle_request
        pending_payload = getattr(self.cm.memory_manager, "pending_payload", None)

        if isinstance(pending_payload, dict):
            failure_ctx = pending_payload.get("failure_context")
        else:
            failure_ctx = getattr(pending_payload, "failure_context", None)

        print(
            f"    [DEBUG] last_topic={self.cm.memory_manager.get_last_topic()!r} "
            f"pending={self.cm.memory_manager.get_pending_topic()!r} "
            f"failure_ctx={failure_ctx!r}"
        )

        # ── Runtime memory-state checks ───────────────────────────────────
        actual_last_topic = self.cm.memory_manager.get_last_topic() or ""

        actual_pending = self.cm.memory_manager.get_pending_topic()

        if "expected_pending" in checks:
            expected_pending = checks.get("expected_pending")

            if actual_pending == expected_pending:
                checks.setdefault("_runtime_passed", []).append(
                    f"pending matched expected: {expected_pending!r}"
                )
            else:
                checks.setdefault("_runtime_failed", []).append(
                    f"pending mismatch: expected {expected_pending!r}, got {actual_pending!r}"
                )

        expected_last_topic = checks.get("expected_last_topic")
        if expected_last_topic:
            if actual_last_topic.strip().lower() == expected_last_topic.strip().lower():
                checks.setdefault("_runtime_passed", []).append(
                    f"last_topic matched expected: {expected_last_topic!r}"
                )
            else:
                checks.setdefault("_runtime_failed", []).append(
                    f"last_topic mismatch: expected {expected_last_topic!r}, got {actual_last_topic!r}"
                )

        # passed, failed, warnings, alive_score = self.checker.check(
        #     response, checks, vision_score
        # )

        runtime_passed = checks.get("_runtime_passed", [])
        runtime_failed = checks.get("_runtime_failed", [])

        passed.extend(runtime_passed)
        failed.extend(runtime_failed)

        if runtime_failed:
            alive_score = max(0.0, alive_score - 0.35)
        # Print compact result
        status = "✓" if not failed else "✗"
        print(f"    {status} Response ({response_time:.0f}ms): {response[:120]!r}{'...' if len(response) > 120 else ''}")
        if vision_score is not None:
            print(f"    Vision score: {vision_score:.2f}")
        for f in failed:
            print(f"    ✗ FAIL: {f}")
        for w in warnings:
            print(f"    ⚠ WARN: {w}")
        for p in passed[:3]:  # only show first 3 passes to keep output clean
            print(f"    ✓ {p}")
        print(f"    Alive score: {alive_score:.2f}")

        return TurnResult(
            turn=turn_num,
            input=user_input,
            response=response,
            response_time_ms=round(response_time, 1),
            checks_passed=passed,
            checks_failed=failed,
            warnings=warnings,
            vision_score=vision_score,
            alive_score=alive_score,
            prompts=turn_prompts,
        )

    def _teardown_scenario(self):
        for p in self._active_patchers:
            try:
                p.stop()
            except Exception:
                pass
        self._active_patchers = []

    def run_scenario(self, scenario: dict) -> ScenarioResult:
        sid = scenario["id"]
        category = scenario["category"]
        desc = scenario["description"]

        print(f"\n{'='*60}")
        print(f"SCENARIO: {sid}")
        print(f"Category: {category}")
        print(f"Description: {desc}")
        print('='*60)

        self._reset_session()
        self._setup_scenario(scenario)

        # Terms from the scenarios' inputs, plus topics Buddy itself keeps bringing up in
        # earlier scenarios (e.g. "DDPG"), minus anything this scenario legitimately mentions.
        mine = set(_TERM_RE.findall(_scenario_text(scenario).lower()))
        learned = {t: o for t, o in learn_reply_terms(self._replies).items() if t not in mine}
        self._current_bleed_terms = {**learned, **self._bleed_terms.get(sid, {})}
        self._scenario_bleed_allow = {a.lower() for a in scenario.get("bleed_allow", [])}
        self._scenario_must_not = list(scenario.get("must_not_contain", []))

        result = ScenarioResult(id=sid, category=category, description=desc)

        for turn_spec in scenario["turns"]:
            tr = self.run_turn(turn_spec, sid)
            result.turns.append(tr)
            if tr.checks_failed:
                result.fatal_fails.extend(tr.checks_failed)

        result.passed = len(result.fatal_fails) == 0
        result.alive_score = round(
            sum(t.alive_score for t in result.turns) / len(result.turns), 3
        ) if result.turns else 0.0
        self._replies[sid] = [t.response for t in result.turns
                              if not t.response.startswith("[STARTUP GREETING")]

        status = "PASS" if result.passed else "FAIL"
        print(f"\n  → Scenario {status} | Alive score: {result.alive_score:.2f}")
        self._teardown_scenario()
        return result

    def run_all(
        self,
        test_file: str = "scenarios/buddy_tony_stark_tests.json",
        #test_file: str = "rover_stress_test.json",
        category_filter: Optional[str] = None,
    ) -> dict:
        with open(test_file, encoding="utf-8") as f:
            scenarios = json.load(f)

        # Bleed terms are derived from ALL scenarios, so build them before any filtering.
        self._bleed_terms = build_bleed_terms([s for s in scenarios if "id" in s and not s.get("skip")])
        self._bleed_events = []
        self._replies = {}

        if category_filter:
            scenarios = [s for s in scenarios if s.get("category") == category_filter]
            print(f"[FILTER] Running category: {category_filter} ({len(scenarios)} scenarios)")

        results = []
        for scenario in scenarios:
            # Skip metadata block and any non-scenario entries
            if "id" not in scenario:
                continue
            if scenario.get("skip"):
                print(f"\n[SKIP] {scenario['id']} — {scenario.get('description', '')}")
                continue

            sr = self.run_scenario(scenario)
            results.append(sr)

        return self._build_report(results)

    def _build_report(self, results: list[ScenarioResult]) -> dict:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        overall_alive = round(sum(r.alive_score for r in results) / total, 3) if total else 0.0

        # Category breakdown
        categories: dict[str, list[float]] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r.alive_score)
        category_scores = {
            cat: round(sum(scores) / len(scores), 3)
            for cat, scores in categories.items()
        }

        # Worst turns (most dead)
        all_turns = [
            (r.id, t)
            for r in results
            for t in r.turns
        ]
        worst_turns = sorted(all_turns, key=lambda x: x[1].alive_score)[:5]

        # All failures
        all_failures = []
        for r in results:
            for turn in r.turns:
                for fail in turn.checks_failed:
                    all_failures.append({
                        "scenario": r.id,
                        "turn": turn.turn,
                        "input": turn.input,
                        "failure": fail,
                        "response_preview": turn.response[:150],
                    })

        report = {
            "summary": {
                "total_scenarios": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": f"{passed/total*100:.0f}%" if total else "0%",
                "overall_alive_score": overall_alive,
                "alive_rating": self._alive_rating(overall_alive),
                "mode": "isolated" if self.isolate else "continuous",
                "context_bleed_turns": len(self._bleed_events),
                "context_guards": _context_guards_state(),
            },
            "context_bleed": self._bleed_events,
            "category_scores": category_scores,
            "worst_turns": [
                {
                    "scenario": sid,
                    "turn": t.turn,
                    "input": t.input,
                    "alive_score": t.alive_score,
                    "failures": t.checks_failed,
                    "response_preview": t.response[:200],
                }
                for sid, t in worst_turns
            ],
            "all_failures": all_failures,
            "scenario_results": [
                {
                    "id": r.id,
                    "category": r.category,
                    "passed": r.passed,
                    "alive_score": r.alive_score,
                    "fatal_fails": r.fatal_fails,
                    "turns": [
                        {
                            "turn": t.turn,
                            "input": t.input,
                            "response": t.response,
                            "response_time_ms": t.response_time_ms,
                            "alive_score": t.alive_score,
                            "passed": t.checks_passed,
                            "failed": t.checks_failed,
                            "warnings": t.warnings,
                            "vision_score": t.vision_score,
                            **({"llm_prompts": t.prompts} if t.prompts else {}),
                        }
                        for t in r.turns
                    ],
                }
                for r in results
            ],
        }

        self._print_summary(report)
        return report

    def _alive_rating(self, score: float) -> str:
        if score >= 0.85: return "JARVIS — fully alive"
        if score >= 0.70: return "MOSTLY ALIVE — a few dead spots"
        if score >= 0.55: return "HALF ALIVE — significant dead zones"
        if score >= 0.40: return "MOSTLY DEAD — feels like a chatbot"
        return "SIRI CIRCA 2012 — needs major work"

    def _print_summary(self, report: dict):
        s = report["summary"]
        print(f"\n{'='*60}")
        print(f"BUDDY ALIVE TEST REPORT")
        print(f"{'='*60}")
        print(f"Scenarios:  {s['passed']}/{s['total_scenarios']} passed ({s['pass_rate']})")
        print(f"Alive Score: {s['overall_alive_score']:.2f} — {s['alive_rating']}")
        print(f"Mode: {s['mode']}  |  context-bleed turns: {s['context_bleed_turns']}")
        for b in report.get("context_bleed", []):
            terms = ", ".join(
                f"{t['term']} (from {t['from_scenario']}"
                + (f"; in prompt: {', '.join(t['in_prompt_sections']) or 'not in any prompt'}" if "in_prompt_sections" in t else "")
                + ")"
                for t in b["terms"]
            )
            print(f"  BLEED {b['scenario']} turn {b['turn']}: {b['input']!r} -> mentions {terms}")
        print()

        print("Category Breakdown:")
        for cat, score in sorted(report["category_scores"].items(), key=lambda x: x[1]):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            flag = " ← FIX THIS" if score < 0.55 else ""
            print(f"  {cat:<35} {bar} {score:.2f}{flag}")

        print()
        print("Worst Turns (most dead):")
        for wt in report["worst_turns"]:
            print(f"  [{wt['alive_score']:.2f}] {wt['scenario']} turn {wt['turn']}: {wt['input']!r}")
            for f in wt["failures"][:2]:
                print(f"         → {f}")

        if report["all_failures"]:
            print(f"\nTotal failures: {len(report['all_failures'])}")
            print("Top 5 failure patterns:")
            failure_types: dict[str, int] = {}
            for f in report["all_failures"]:
                key = f["failure"][:60]
                failure_types[key] = failure_types.get(key, 0) + 1
            for msg, count in sorted(failure_types.items(), key=lambda x: -x[1])[:5]:
                print(f"  x{count}  {msg}")

        print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Buddy Alive Test Runner")
    parser.add_argument(
        "--category",
        help="Run only scenarios matching this category",
        default=None,
    )
    parser.add_argument(
        "--debug-vision",
        action="store_true",
        help="Print vision awareness scores for each turn",
    )
    parser.add_argument(
        "--output",
        help="Save full report to this JSON file",
        default=None,
    )
    parser.add_argument(
        "--test-file",
        help="Path to test JSON file",
        default="scenarios/buddy_tony_stark_tests.json",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="Wipe ALL session memory between scenarios (diagnostic). Default is one continuous "
             "session, which is realistic and is what exposes context bleed.",
    )
    parser.add_argument(
        "--trace-prompts",
        action="store_true",
        help="Record the exact prompts sent to the LLM each turn (saved in the report as "
             "llm_prompts) and, for every context-bleed turn, say which prompt section the "
             "leaked term was in. Makes the report large.",
    )
    args = parser.parse_args()

    runner = BuddyAliveTestRunner(
        debug_vision=args.debug_vision, isolate=args.isolate, trace_prompts=args.trace_prompts,
    )
    report = runner.run_all(
        test_file=args.test_file,
        category_filter=args.category,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Full report saved to: {args.output}")


if __name__ == "__main__":
    main()
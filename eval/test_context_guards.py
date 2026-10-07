"""
Tests for the four context-bleed guards (brain/utils/context_guards.py, belief_state.py,
prompts.py, rag_core.py). No Ollama / torch needed.

Run from the project root:   python eval/test_context_guards.py
"""
import ast
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.utils import context_guards as cg
from brain.belief_state import BeliefStateManager, Domain, Phase

FAILED = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILED.append(name)


def set_guards(on: bool):
    os.environ[cg.ENV_SWITCH] = "1" if on else "0"


class MM:
    def __init__(self, topic=None, pending=None):
        self.topic, self.pending = topic, pending

    def get_last_topic(self):
        return self.topic

    def get_pending_topic(self):
        return self.pending


# ── switch ───────────────────────────────────────────────────────────────────
for v, expect in [("1", True), ("", True), ("0", False), ("false", False), ("OFF", False), ("no", False)]:
    os.environ[cg.ENV_SWITCH] = v
    check(f"switch {v!r} -> {expect}", cg.guards_enabled() is expect)
del os.environ[cg.ENV_SWITCH]
check("switch default on", cg.guards_enabled())

# ── guard 1: follow-up detection ─────────────────────────────────────────────
for t in ["yes", "yeah take a look", "how does it handle obstacles though", "is it better than a simple PID for this",
          "back to what we were discussing — is particle filter SLAM better", "do that again", "tell me more"]:
    check(f"followup: {t!r}", cg.is_followup_like(t))
for t in ["activate the rover", "what's the difference between reinforcement learning and imitation learning",
          "explain model predictive control for a ground rover", "search for latest ROS2 Humble navigation updates",
          "can you explain how kalman filters work in detail please"]:
    check(f"NOT followup: {t!r}", not cg.is_followup_like(t))

# ── guard 1: buffer selection ────────────────────────────────────────────────
buf = ["User: explain MPC\nBuddy: MPC optimises a trajectory over a horizon..."]
sim_lo = lambda a, b: 0.20
sim_hi = lambda a, b: 0.60
set_guards(True)
check("unrelated question -> buffer dropped", cg.select_recent_context("what is imitation learning vs RL", buf, sim_lo) == [])
check("similar question -> buffer kept", cg.select_recent_context("what is imitation learning vs RL", buf, sim_hi) == buf)
check("follow-up -> buffer kept even if dissimilar", cg.select_recent_context("how does it handle obstacles though", buf, sim_lo) == buf)
check("empty buffer stays empty", cg.select_recent_context("yes", [], sim_hi) == [])
def boom(a, b): raise RuntimeError("embedder down")
check("scoring error fails OPEN", cg.select_recent_context("what is imitation learning vs RL", buf, boom) == buf)
check("best exchange wins (not diluted)", cg.select_recent_context(
    "q", ["a", "b", "c"], lambda a, b: 0.9 if b == "c" else 0.1) == ["a", "b", "c"])
set_guards(False)
check("guards off -> buffer always kept", cg.select_recent_context("what is imitation learning vs RL", buf, sim_lo) == buf)

# ── guard 3: bare acknowledgements ───────────────────────────────────────────
set_guards(True)
cg.note_last_reply("Got it — saved 'RL basics' to long-term memory.")
check("bare yes after non-question -> not stored", not cg.should_store_exchange("yes"))
check("'ok sure' after non-question -> not stored", not cg.should_store_exchange("ok sure"))
check("real sentence after non-question -> stored", cg.should_store_exchange("yes and also explain DDPG"))
cg.note_last_reply("Do you want me to save our previous topic?")
check("bare yes after question -> stored", cg.should_store_exchange("yes"))
cg.note_last_reply("")
check("bare no after empty reply -> not stored", not cg.should_store_exchange("no"))
set_guards(False)
check("guards off -> always stored", cg.should_store_exchange("yes"))
set_guards(True)

# ── guard 2: belief state ────────────────────────────────────────────────────
def run(turns, guards=True):
    set_guards(guards)
    m = BeliefStateManager()
    out = []
    for text, topic in turns:
        out.append((m.update(text, MM(topic)), m))
    return out

tech = [("how does the lidar pointnet pipeline work", "pointnet"),
        ("what about the voxel grid sampling", "pointnet")]
r = run(tech + [("what's the difference between reinforcement learning and imitation learning", "reinforcement learning")])
st = r[-1][0]
check("topic change drops old entities", "voxel" not in st.active_entities and "lidar" not in st.active_entities)
check("thread summary follows new topic", "pointnet" not in st.thread_summary)
r = run(tech + [("what's the difference between reinforcement learning and imitation learning", "reinforcement learning")], guards=False)
check("guards off: old entities survive (old behaviour)", "voxel" in r[-1][0].active_entities or "lidar" in r[-1][0].active_entities)

r = run([("hello there", None), ("how does the lidar pointnet pipeline work", "pointnet")])
check("None -> topic does NOT reset", "lidar" in r[-1][0].active_entities or "pointnet" in r[-1][0].active_entities or "pipeline" in r[-1][0].active_entities)

r = run([("how does the lidar pointnet pipeline work", "vision_awareness_111"),
         ("tell me what the lidar sees", "vision_awareness_222")])
check("timestamped topic suffix is the same thread", "lidar" in r[-1][0].active_entities and "pointnet" in r[-1][0].active_entities)

r = run([("how does the lidar pointnet pipeline work", "pointnet"), ("ok", None)])
check("topic cleared -> stale Thread line removed", r[-1][0].thread_summary == "")

r = run([("how does the lidar pointnet pipeline work", "pointnet"),
         ("actually let's talk about the weather today", "pointnet")])
check("explicit pivot resets entities", "lidar" not in r[-1][0].active_entities and "pointnet" not in r[-1][0].active_entities)

set_guards(True)
m = BeliefStateManager()
m.update("look at my setup with the webcam", MM("vision"))
m.report_tool_result("webcam", True, "saw a rover")
check("fresh tool shown", "Last tool: webcam" in m.state.to_prompt_block())
for i in range(5):
    m.update(f"explain kalman filter step {i}", MM("vision"))
check("tool result expires after a few turns", "Last tool" not in m.state.to_prompt_block())
check("expired tool still available to is_vision_active logic", m.state.last_tool_result is not None)
set_guards(False)
check("guards off: tool never expires", "Last tool: webcam" in m.state.to_prompt_block())

set_guards(True)
m = BeliefStateManager()
m.update("explain the kalman filter and ekf slam sensor model", MM("slam"))
check("technical domain set", m.state.domain == Domain.TECHNICAL)
m.update("hmm", MM("slam"))
check("short message inherits domain", m.state.domain == Domain.TECHNICAL)
m.update("what is the capital of france and why was it chosen as the capital originally", MM("slam"))
check("long unrelated message does NOT inherit domain", m.state.domain == Domain.UNKNOWN)

# ── guard 4: prompt wording ──────────────────────────────────────────────────
import importlib.util
src = (ROOT / "brain/core/prompts.py").read_text(encoding="utf-8")
tree = ast.parse(src)
ns = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("_answer_rules", "generate_response"):
        exec(compile(ast.Module([node], []), "prompts", "exec"), ns)
set_guards(True)
p = ns["generate_response"]("what is RL", "General Conversation", "No relevant past discussion.")
check("guards on: new wording", "Answer the user's latest question directly first" in p and "connect the ideas smoothly" not in p
      and "Ask a follow-up question to keep the conversation alive" not in p)
set_guards(False)
p = ns["generate_response"]("what is RL", "General Conversation", "No relevant past discussion.")
check("guards off: original wording", "connect the ideas smoothly" in p and "keep the conversation alive" in p)

# ── RagCore passes the real user text through ────────────────────────────────
rc_src = (ROOT / "brain/core/RAG/rag_core.py").read_text(encoding="utf-8")
rc_mod = types.ModuleType("rag_core_under_test")
rc_mod.__dict__.update({"__name__": "rag_core_under_test"})
# stub the helper imports it needs
helper_names = [n for n in ("shorten", "safe_join")]
rc_clean = "\n".join(l for l in rc_src.splitlines() if not l.startswith(("from brain", "import brain")))
rc_mod.__dict__["shorten"] = lambda s, n: s[:n]
rc_mod.__dict__["safe_join"] = lambda parts: "\n\n".join(parts)
exec(compile(rc_clean, "rag_core", "exec"), rc_mod.__dict__)
seen = {}
def llm_with(prompt, mode, system_context="", override_sensory=None, user_text=None):
    seen["u"] = user_text; return "ok"
def llm_plain(prompt, mode):
    seen["plain"] = True; return "ok"
ctx = rc_mod.RagContext(topic="t")
rc_mod.RagCore(llm_with).answer(user_input="hi there", ctx=ctx)
check("RagCore passes user_text when supported", seen.get("u") == "hi there")
rc_mod.RagCore(llm_plain).answer(user_input="hi there", ctx=ctx)
check("RagCore still works with a plain llm_fn", seen.get("plain") is True)

print()
print("ALL PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)

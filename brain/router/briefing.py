from brain.adaptive.vision.vision_awareness import score_vision_need
from brain.core.brain import sensory_memory
import ollama
import json
import re

BUDDY_TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "save_schedule", "description": "Save a reminder, appointment, event, deadline, or calendar item.", "parameters": {"type": "object", "properties": {"raw_text": {"type": "string"}, "title": {"type": "string"}, "date_time_text": {"type": "string"}}, "required": ["raw_text"]}}},
    {"type": "function", "function": {"name": "recall_schedule", "description": "Recall upcoming reminders, calendar items, appointments, or plans.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "time_range": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "delete_schedule", "description": "Delete, cancel, or remove a saved reminder or schedule item.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}}},
    {"type": "function", "function": {"name": "save_topic_memory", "description": "Save the current conversation, note, topic, or personal memory for later.", "parameters": {"type": "object", "properties": {"topic": {"type": "string"}, "raw_text": {"type": "string"}}, "required": ["raw_text"]}}},
    {"type": "function", "function": {"name": "recall_memory", "description": "Recall a previous conversation, saved note, memory, or topic.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "topic": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "delete_memory", "description": "Delete, forget, erase, or remove a saved memory/conversation/topic.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the web for current, latest, external, or factual information.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "web_followup", "description": "Follow up on web search results or saved links: open, show, read, summarize, or compare a result, source, article, link, website, first/second item, or numbered result.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "show", "read", "summarize", "compare", "other"]}, "index": {"type": "integer"}, "target": {"type": "string"}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "move_rover", "description": "Move or stop the rover: forward, backward, left, right, stop.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["forward", "backward", "left", "right", "stop"]}, "raw_text": {"type": "string"}}, "required": ["direction"]}}},
    {"type": "function", "function": {"name": "connect_rover", "description": "Connect, wake, activate, or bring the rover online.", "parameters": {"type": "object", "properties": {"raw_text": {"type": "string"}}, "required": ["raw_text"]}}},
    {"type": "function", "function": {"name": "rover_status", "description": "Check rover status, telemetry, connection, battery, or readiness.", "parameters": {"type": "object", "properties": {"raw_text": {"type": "string"}}, "required": ["raw_text"]}}},
    {"type": "function", "function": {"name": "tello_takeoff", "description": "Start a Tello takeoff and hover flow. Requires safety confirmation before takeoff.", "parameters": {"type": "object", "properties": {"raw_text": {"type": "string"}}, "required": ["raw_text"]}}},
    {"type": "function", "function": {"name": "tello_land", "description": "Land the Tello drone.", "parameters": {"type": "object", "properties": {"raw_text": {"type": "string"}}, "required": ["raw_text"]}}},
    {"type": "function", "function": {"name": "tello_stream", "description": "Start or open the Tello camera stream.", "parameters": {"type": "object", "properties": {"raw_text": {"type": "string"}}, "required": ["raw_text"]}}},
    {
        "type": "function",
        "function": {
            "name": "system_query",
            "description": (
                "ONLY for explicit requests about: current time, current date, "
                "battery percentage, current local weather, GPS location, or CPU load. "
                "NEVER use this for technical/conceptual questions, robotics/engineering "
                "discussion, or anything not literally asking about device/system status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {"type": "function", "function": {"name": "friday_task", "description": "Ask Friday, the background agent, to do a delegated task.", "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}}},
    {"type": "function", "function": {"name": "job_watchdog", "description": "Run or read Friday's job watchdog scan/recap.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["scan", "read", "other"]}, "raw_text": {"type": "string"}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "run_rover_simulation", "description": "Run the LQR rover tracking simulation with a given scenario (straight or sinusoidal path).", "parameters": {"type": "object", "properties": {"scenario": {"type": "string", "enum": ["straight", "sinusoidal"]}, "amplitude": {"type": "number"}, "wavelength": {"type": "number"}}, "required": ["scenario"]}}},
    {"type": "function", "function": {"name": "report_last_simulation_result", "description": "Report the verdict and metrics from the most recently run rover simulation.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "show_simulation_plot", "description": "Generate and open the diagnostic plot for the most recent rover simulation run.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "report_rover_log_result", "description": "Report the step-response result (rise time, overshoot, settling) of the last saved rover telemetry log.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "plot_rover_log", "description": "Plot the velocity and PWM response of the last saved rover telemetry log.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "where_was_that_saved", "description": "Report the file path of the most recently saved rover log or simulation run.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "plot_estimated_trajectory", "description": "Plot the rover's estimated turning direction and path based on commanded wheel setpoints, useful for validating command logic with wheels off.", "parameters": {"type": "object", "properties": {}, "required": []}}},
]

BUDDY_TOOL_TO_INTENT = {
    "save_schedule": "schedule_save",
    "recall_schedule": "schedule_recall",
    "delete_schedule": "schedule_delete",
    "save_topic_memory": "save_memory",
    "recall_memory": "recall_memory",
    "delete_memory": "delete_memory",
    "search_web": "web_search",
    "web_followup": "web_followup",
    "move_rover": "device_control",
    "connect_rover": "device_control",
    "rover_status": "device_control",
    "tello_takeoff": "device_control",
    "tello_land": "device_control",
    "tello_stream": "device_control",
    "system_query": "system_query",
    "friday_task": "friday_task",
    "job_watchdog": "job_watchdog",
    "run_rover_simulation": "device_control",
    "report_last_simulation_result": "device_control",
    "show_simulation_plot": "device_control",
    "report_rover_log_result": "device_control", 
    "plot_rover_log": "device_control",            
    "where_was_that_saved": "device_control",
    "plot_estimated_trajectory": "device_control",       
}

_TIME_RE = re.compile(
    r"""
    \b(
        \d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?
        |
        \d{1,2}\.\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?
        |
        \d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)?
        |
        noon|midnight
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

def user_mentioned_time(text: str) -> bool:
    return bool(_TIME_RE.search(text or ""))

_RELATIVE_RE = re.compile(r"\bin\s+\d+\s+(day|days|week|weeks|month|months|year|years)\b", re.IGNORECASE)
_ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DMY_SLASH_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_DMY_DOT_RE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b")
_MONTHNAME_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.IGNORECASE)
_DATE_WORDS = {
    "today","tomorrow","yesterday",
    "this week","next week","last week",
    "this month","next month","last month",
    "this year","next year","last year",
    "weekend","this weekend","next weekend",
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday"
}

# ── Signal groups, reused across grounding checks ──────────────────────

SAVE_TOPIC_SIGNALS = [
    "save this", "save that", "save it", "store this", "store that",
    "note this", "take a note", "remember this", "keep this in mind",
    "bookmark", "write this down", "jot this down",
]

RECALL_SIGNALS = [
    "recall", "remember", "what did", "read", "show me", "tell me what",
    "summary", "summarize", "saved", "previous", "earlier", "last time",
    "did you save", "did we", "what have we",
]

MEMORY_OBJECT_WORDS = [
    "conversation", "discussion", "memory", "topic", "notes",
    "chat", "history", "about",
]

SCHEDULE_OBJECT_WORDS = [
    "schedule", "reminder", "reminders", "appointment", "appointments",
    "meeting", "meetings", "event", "calendar", "plans",
]

DELETE_SIGNALS = ["delete", "remove", "forget", "erase", "wipe", "clear", "cancel"]

SCHEDULE_RECALL_SIGNALS = [
    "what's my schedule", "what is my schedule", "what do i have",
    "what have i got", "upcoming", "next meeting", "any meetings",
    "any appointments", "my calendar", "what's on my calendar",
    "am i free", "do i have anything",
]

WEB_SEARCH_SIGNALS = [
    "search", "google", "look up", "find me", "latest", "news",
    "current price", "what's happening",
]

JOB_WATCHDOG_SIGNALS = [
    "job", "jobs", "watchdog", "recap", "scan for jobs", "job search",
]


TOOL_GROUNDING_KEYWORDS = {
    "web_followup": ["open", "summarize", "summary", "compare", "result", "article", "link", "source"],
    "move_rover": ["rover", "move", "turn", "go", "stop"],
    "connect_rover": ["rover", "connect", "activate"],
    "tello_takeoff": ["tello", "drone", "takeoff", "take off", "hover"],
    "tello_land": ["tello", "drone", "land"],
    "tello_stream": ["tello", "drone", "stream", "camera"],
    "friday_task": ["friday"],
    "system_query": ["time", "date", "battery", "weather", "location", "cpu", "clock"],
    "save_schedule": ["remind", "reminder", "schedule", "appointment"],
    "recall_memory": RECALL_SIGNALS,
    "rover_status": ["rover", "status", "connected", "battery", "ready", "telemetry", "online"],
    "search_web": WEB_SEARCH_SIGNALS,
    "job_watchdog": JOB_WATCHDOG_SIGNALS,
    "run_rover_simulation": ["simulation", "sim", "simulate"],
    "report_last_simulation_result": ["simulation", "sim", "result", "verdict"],
    "show_simulation_plot": ["simulation", "sim", "plot", "graph", "chart"],
    "report_last_simulation_result": ["simulation", "sim", "result", "verdict"],
    "show_simulation_plot": ["simulation", "sim", "plot", "graph", "chart"],
    "report_rover_log_result": ["result", "response", "log", "movement", "behavior"],   # ← add
    "plot_rover_log": ["plot", "graph", "chart", "log", "behavior", "movement"],         # ← add
    "where_was_that_saved": ["where", "path", "location", "save", "saved", "file"],
    "plot_estimated_trajectory": ["direction", "path", "trajectory", "turn", "curve", "heading"],
}


def _passes_grounding(name, user_input, mm):
    text = user_input.lower()

    # ── save_topic_memory: recall-shaped phrasing beats a bare "save" ──
    if name == "save_topic_memory":
        has_recall_shape = any(s in text for s in RECALL_SIGNALS)
        has_save_shape = any(s in text for s in SAVE_TOPIC_SIGNALS)
        if has_recall_shape and not has_save_shape:
            return False
        return has_save_shape

    # ── delete_memory vs delete_schedule: same verbs, different object.
    # A bare "delete/remove/forget" isn't enough — must name a memory
    # object, and must NOT be more clearly about a schedule object. ──
    if name == "delete_memory":
        has_delete_verb = any(s in text for s in DELETE_SIGNALS)
        has_memory_object = any(w in text for w in MEMORY_OBJECT_WORDS)
        has_schedule_object = any(w in text for w in SCHEDULE_OBJECT_WORDS)
        if not has_delete_verb:
            return False
        if has_schedule_object and not has_memory_object:
            return False
        return has_memory_object or True  # verb alone still weakly grounded; tighten below
        # NOTE: if this proves too loose in testing, require has_memory_object strictly.

    if name == "delete_schedule":
        has_delete_verb = any(s in text for s in DELETE_SIGNALS)
        has_schedule_object = any(w in text for w in SCHEDULE_OBJECT_WORDS)
        return has_delete_verb and has_schedule_object

    # ── recall_memory vs recall_schedule: same trap as save — a bare
    # "recall"/"remember" isn't enough if the utterance is clearly about
    # a schedule instead of a topic/conversation. ──
    if name == "recall_schedule":
        has_recall_shape = any(s in text for s in SCHEDULE_RECALL_SIGNALS)
        has_save_shape = (
            "remind me" in text
            or any(s in text for s in SAVE_TOPIC_SIGNALS)
            or user_mentioned_time(user_input)
            or user_mentioned_date(user_input)
        )
        # Save-shaped phrasing wins — a bare object word like "meeting"
        # is not enough evidence to ground a recall tool.
        if has_save_shape and not has_recall_shape:
            return False
        return has_recall_shape

    if name == "recall_memory":
        has_recall_signal = any(s in text for s in RECALL_SIGNALS)
        has_schedule_object = any(w in text for w in SCHEDULE_OBJECT_WORDS)
        # if it's clearly schedule-shaped and NOT memory-shaped, don't ground here
        if has_schedule_object and not any(w in text for w in MEMORY_OBJECT_WORDS):
            return False
        return has_recall_signal

    keywords = TOOL_GROUNDING_KEYWORDS.get(name)
    if keywords and not any(k in text for k in keywords):
        return False
    if name == "web_followup" and not mm.get_active_web_topic():
        return False
    if name == "save_schedule" and not (user_mentioned_time(user_input) or user_mentioned_date(user_input)):
        return False
    return True

def user_mentioned_date(text: str) -> bool:
    t = (text or "").lower()
    if any(w in t for w in _DATE_WORDS):
        return True
    if _RELATIVE_RE.search(t):
        return True
    if _ISO_RE.search(t) or _DMY_SLASH_RE.search(t) or _DMY_DOT_RE.search(t):
        return True
    if _MONTHNAME_RE.search(t):
        return True
    return False

def _tool_call_name(tool_call) -> str:
    function = getattr(tool_call, "function", None)
    return (getattr(function, "name", "") if function is not None else "").strip()

def _tool_call_args(tool_call) -> dict:
    function = getattr(tool_call, "function", None)
    args = getattr(function, "arguments", {}) if function is not None else {}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return args if isinstance(args, dict) else {}

def _has_required_tool_args(name: str, args: dict) -> bool:
    required = TOOL_CALL_REQUIRED_ARGS.get(name, set())
    return all(args.get(key) not in (None, "") for key in required)

TOOL_CALL_REQUIRED_ARGS = {
    tool["function"]["name"]: set(tool["function"]["parameters"].get("required", []))
    for tool in BUDDY_TOOL_SCHEMAS
}

def build_briefing(user_input, mm, belief_state, episodic_router, tello_client=None, rover_client=None) -> str:
    lines = []
    last_topic = mm.get_last_topic() or "none"
    lines.append(f"Current topic: {last_topic}")

    pending = mm.get_pending_topic()
    if pending:
        lines.append(f"Pending confirmation: {_humanize_pending(pending, mm.pending_payload)}")

    try:
        matches = episodic_router._score_episodes(user_input, restrict_topic=None)[:2]
        for m in matches:
            lines.append(f"Related past topic: '{m.episode.topic}' (similarity {m.score:.2f})")
    except Exception:
        pass

    try:
        recent = list(sensory_memory)[-2:] if sensory_memory else []
        awareness = score_vision_need(user_input, last_topic, pending or "", recent, "")
        if awareness.score > 0.3:
            lines.append(f"Vision relevance: {awareness.score:.2f} — {awareness.context_hint}")
    except Exception:
        pass

    return "\n".join(lines) if lines else "No prior context."

_PENDING_HUMAN = {
    "Awaiting_Topic": "waiting for the user to name a topic to save",
    "Awaiting_Delete_Confirm": "waiting for yes/no to confirm a delete",
    "Awaiting_ConversationSave_Confirm": "waiting for yes/no to confirm saving this conversation",
    "Awaiting_Schedule_Details": "waiting for missing schedule details (title/time)",
}
def _humanize_pending(state, payload):
    return _PENDING_HUMAN.get(state, state)

def route_via_llm(user_input, mm, belief_state, episodic_router, **device_clients):
    briefing = build_briefing(user_input, mm, belief_state, episodic_router, **device_clients)
    system = (
        "You are Buddy's routing layer. Choose exactly one tool when the user asks Buddy "
        "to perform a capability. If the user is only chatting or no tool fits, answer in "
        "plain text without a tool call."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{briefing}"},
        {"role": "user", "content": user_input},
    ]
    try:
        response = ollama.chat(model="llama3.2:latest", messages=messages,
                                tools=BUDDY_TOOL_SCHEMAS, options={"temperature": 0})
    except Exception as e:
        print(f"[Router] error: {e}")
        return None

    message = getattr(response, "message", None)
    tool_calls = getattr(message, "tool_calls", None) if message else None
    if not tool_calls:
        raw_reply = getattr(message, "content", "") if message else ""
        filtered_reply = postprocess_chat_response(raw_reply)
        return {"intent": "chat", "tool_name": None, "tool_args": {}, "response": filtered_reply}

    call = tool_calls[0]
    name = _tool_call_name(call)
    args = _tool_call_args(call)

    if name not in BUDDY_TOOL_TO_INTENT:
        return None
    
    # For Device Control: Always ensure raw_text is the actual original utterance, never trust the LLM to fill it
    if "raw_text" in TOOL_CALL_REQUIRED_ARGS.get(name, set()) or "raw_text" in args or name in (
        "move_rover", "connect_rover", "tello_takeoff", "tello_land", "tello_stream"
    ):
        args["raw_text"] = user_input

    if not _passes_grounding(name, user_input, mm):
        print(f"[Router] rejected ungrounded {name} for: {user_input[:50]!r}")
        return {"intent": "chat", "tool_name": None, "tool_args": {}, "response": ""}

    # ── NEW: grounding check — reject system_query if it isn't actually one ──
    if name == "web_followup":
        has_keyword = any(w in user_input.lower() for w in
            ["open", "summarize", "summary", "compare", "result",
            "article", "link", "source", "browser", "website"])
        has_active_session = bool(mm.get_active_web_topic())
        if not (has_keyword and has_active_session):
            print(f"[Router] rejected ungrounded web_followup for: {user_input[:50]!r}")
            return {"intent": "chat", "tool_name": None, "tool_args": {}, "response": ""}

    if name == "save_schedule":
        if not (user_mentioned_time(user_input) or user_mentioned_date(user_input) or "remind" in user_input.lower()):
            print(f"[Router] rejected ungrounded save_schedule for: {user_input[:50]!r}")
            return {"intent": "chat", "tool_name": None, "tool_args": {}, "response": ""}
        
    if name in ("move_rover", "connect_rover", "tello_takeoff", "tello_land", "tello_stream"):
        if not any(w in user_input.lower() for w in
                ["rover", "tello", "drone", "robot", "move", "turn", "go",
                    "stop", "connect", "activate", "takeoff", "take off", "land", "hover"]):
            print(f"[Router] rejected hallucinated device_control for: {user_input[:50]!r}")
            return {"intent": "chat", "tool_name": None, "tool_args": {}, "response": ""}

    if name == "friday_task":
        if not any(w in user_input.lower() for w in
                ["friday", "ask friday", "tell friday"]):
            print(f"[Router] rejected hallucinated friday_task for: {user_input[:50]!r}")
            return {"intent": "chat", "tool_name": None, "tool_args": {}, "response": ""}
        
    if name == "system_query":
        if not any(w in user_input.lower() for w in
                   ["time", "date", "battery", "weather", "location", "cpu", "clock"]):
            print(f"[Router] rejected hallucinated system_query for: {user_input[:50]!r}")
            return {"intent": "chat", "tool_name": None, "tool_args": {}, "response": ""}

    if name in ("run_rover_simulation", "report_last_simulation_result", "show_simulation_plot"):
        if not any(w in user_input.lower() for w in ["simulation", "sim", "simulate", "plot", "graph"]):
            print(f"[Router] rejected hallucinated {name} for: {user_input[:50]!r}")
            return {"intent": "chat", "tool_name": None, "tool_args": {}, "response": ""}
        
    return {"intent": BUDDY_TOOL_TO_INTENT[name], "tool_name": name, "tool_args": args}

def postprocess_chat_response(text, max_sentences=2, max_chars=300):
    if not text:
        return ""
    text = " ".join(text.split()) 
    sentence_endings = {".", "?", "!"}
    sentences, current = [], ""
    for char in text:
        current += char
        if char in sentence_endings:
            sentences.append(current.strip())
            current = ""
            if len(sentences) >= max_sentences:
                break
    if not sentences and current:
        sentences.append(current.strip())

    short_text = " ".join(sentences).strip()
    if len(short_text) > max_chars:
        short_text = short_text[:max_chars].rsplit(" ", 1)[0].strip() + "..."
    return short_text

def _system_query_dispatch(args, cm):
    raw = args.get("query_type") or args.get("query") or args.get("raw_text") or ""
    resolved = cm._resolve_system_query(raw)
    output = cm.intent_handler.handle_intent(resolved)
    return output or "Sorry, I couldn't get that — try asking again."

# DESTRUCTIVE_TOOLS = {"delete_memory", "delete_schedule", "move_rover", "tello_takeoff"}
DESTRUCTIVE_TOOLS = {"delete_memory", "delete_schedule", "tello_takeoff"}

def filter_tool_call(tool_name, args):
    if not _has_required_tool_args(tool_name, args):
        return "reject"
    if tool_name in DESTRUCTIVE_TOOLS:
        return "confirm"
    return "allow"

def make_dispatch_table(cm):
    return {
        "recall_memory":     lambda a: cm.memory_handler.handle_recall(a.get("query",""), routed_intent="recall_memory"),
        "delete_memory":     lambda a: cm.memory_handler.execute_delete_from_router(a.get("target","")),
        "search_web":         lambda a: cm.handle_internet_request(a.get("query","")),
        "web_followup":       lambda a: cm.answer_from_web_followup(a.get("target","")),
        "move_rover":         lambda a: cm.device_handler.handle(a.get("raw_text","")),
        "connect_rover":      lambda a: cm.device_handler.handle(a.get("raw_text","")),
        "system_query":       lambda a: _system_query_dispatch(a, cm),
        "save_schedule":      lambda a: cm.memory_handler.save_memory(a.get("raw_text",""), routed_intent="schedule_save"),
        "save_topic_memory": lambda a: cm.memory_handler.save_from_router(a.get("topic",""), a.get("raw_text","")),
        "recall_schedule":    lambda a: cm.memory_handler.handle_recall(a.get("query",""), routed_intent="schedule_recall"),
        "run_rover_simulation":          lambda a: cm.sim_manager.run_simulation(scenario=a.get("scenario"), amplitude=a.get("amplitude"), wavelength=a.get("wavelength")),
        "report_last_simulation_result": lambda a: cm.sim_manager.report_last_result(),
        "show_simulation_plot":          lambda a: cm.sim_manager.show_plot(cm._open_file_for_user),
        "report_rover_log_result": lambda a: cm.device_handler.rover_manager.report_last_log_result(),
        "plot_rover_log":          lambda a: cm.device_handler.rover_manager.plot_last_log(cm._open_file_for_user),
        "where_was_that_saved":    lambda a: cm.report_last_saved_location(),
        "plot_estimated_trajectory": lambda a: cm.device_handler.rover_manager.plot_last_trajectory_estimate(cm._open_file_for_user),
    }


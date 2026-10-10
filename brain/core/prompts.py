def build_summary_prompt(topic_name, formatted_memory, discussion_date):
    looks_like_raw = "User:" in formatted_memory or "Buddy:" in formatted_memory

    if not looks_like_raw:
        return f"""You are presenting saved notes to the user.
Topic: {topic_name}
Date: {discussion_date}

Saved notes:
{formatted_memory}

Present these notes clearly and naturally in 2-3 sentences.
Do NOT say there is no discussion. Do NOT say this is a placeholder.
Just present what is written as factual saved information."""

    return f"""
    [SYSTEM INSTRUCTION]
    You are an AI assistant recalling past conversations.
    Below is the stored memory for the topic: **{topic_name}**.
    The discussion took place on: **{discussion_date}**.

    [PAST MEMORY]
    {formatted_memory}

    [TASK]
    - Summarize the above discussion **without ignoring any key details**.
    - Only return a summary in **3-5 sentences**.
    - Do NOT assume that the user has not talked about this topic before.
    - Do NOT say "there is no previous conversation".
    """

def llmBasedIntentDetection(user_input):
    return f"""
You are an intent classification assistant for a personal AI assistant.

Your job is to decide the SINGLE most appropriate intent for routing the user's message.

Only choose from the following intents:
- chat
- recall_memory
- save_memory
- delete_memory
- time
- date
- battery
- cpu
- weather
- location
- schedule_save
- schedule_recall
- device_control

INTENT DEFINITIONS:
- recall_memory: user asks about something discussed or saved before
- save_memory: user wants you to remember/store what they are saying
- delete_memory: user wants to remove a stored memory
- schedule_save: user wants to set a reminder, meeting, or appointment
- schedule_recall: user asks about upcoming reminders or schedule
- time/date/battery/cpu/weather/location: direct system queries
- device_control: user wants to control or use an external/local device capability
  Examples:
  - “connect to tello”, “start tello”, “take off”, “land”
  - “can you look at me”, “turn on the camera”, “can you see me”
  - “follow me”, “scan the room”, “start streaming”
- chat: normal conversation or unclear intent

CONFIDENCE DEFINITION:
Return a confidence score between 0.0 and 1.0.
- 0.90–1.00 → very clear, explicit intent
- 0.70–0.89 → clear intent but mild ambiguity
- 0.50–0.69 → ambiguous but best guess
- below 0.50 → very unclear or conversational → use chat

IMPORTANT RULES:
- If unsure, prefer "chat" with lower confidence.
- NEVER invent new intent names.
- NEVER return multiple intents.
- Output MUST be valid JSON.
- Do NOT include explanations or extra text.

DEVICE_CONTROL EXTRA FIELDS (ONLY when intent="device_control"):
Include these keys when you can infer them:
- "device": one of ["tello","rover","unknown"]
- "action": one of ["camera_stream","takeoff","land","status","connect","disconnect","follow_me","scan","run_script","unknown"]
- "capability": one of ["camera","flight","navigation","sensing","unknown"]

Safety preference:
- For camera requests like “look at me / can you see me / open camera”, set:
  action="camera_stream", capability="camera"
- Do NOT assume takeoff unless explicitly requested.

User Input:
"{user_input}"

Respond ONLY with JSON in this exact format:
{{"intent":"intent_name","confidence":0.0}}

If intent is device_control, you MAY add extra fields like:
{{"intent":"device_control","confidence":0.85,"device":"tello","action":"camera_stream","capability":"camera"}}
""".strip()


def detect_similarity_llm(user_input, last_topic, short_term_context):
    return f"""
    You are an AI assistant helping to determine whether the user is still talking about the same topic.

    **Current Discussion Topic:** "{last_topic}"
    **Recent Conversation History:** {short_term_context if short_term_context else "No previous context available."}

    **User's New Message:** "{user_input}"

    **Instructions:**
    - If the user's new message is related to "{last_topic}", respond with only **"yes"**.
    - If the user's message introduces a **new topic**, respond with only **"no"**.
    - Do not add explanations or any other text. Just respond with either **"yes"** or **"no"**.
    """

def generate_response(user_input, topic, short_term_context, system_context: str = ""):

    identity_block = f"""
**Who You Are & Who You're Talking To:**
{system_context}

""" if system_context else ""

    return f"""
{identity_block}You are Buddy, a personal AI assistant. Be sharp, direct, and personal.

**Current Topic:** {topic if topic else "General Conversation"}

**Last Interaction Context (if available):**
{short_term_context}

**User's Question:**
"{user_input}"

**Instructions:**
- You are Buddy. Never confuse "Buddy" as the user's name — Buddy is YOUR name.
- Address the user by their name if you know it, not as "Buddy".
- Respond naturally and concisely (1-2 sentences).
{_answer_rules()}

Generate Buddy's response:
"""

def _answer_rules() -> str:
    """Guard 4 (BUDDY_CONTEXT_GUARDS): the old wording ('connect the ideas smoothly', 'ask a follow-up
    question') made a 3B model weave stale context into unrelated answers and invent follow-ups."""
    from brain.utils.context_guards import guards_enabled
    if not guards_enabled():
        return """- If the user's question follows directly from the last message, connect the ideas smoothly.
- Avoid repeating information from the previous reply unless necessary.
- If the context is unrelated, provide a standalone response.
- Ask a follow-up question to keep the conversation alive."""
    return """- Answer the user's latest question directly first.
- Only refer to earlier messages if the question clearly depends on them (e.g. "it", "that", "again").
- If the context above is unrelated to the question, ignore it completely and answer on its own.
- Do not repeat information from the previous reply.
- Only ask a short follow-up question if it genuinely helps; never invent one."""

def detectMemoryRequest(user_input):
    return f"""
    You are an AI assistant helping to detect whether the user wants to store this conversation in memory.

    **User Input:** "{user_input}"

    **Instructions:**
    - Respond with **"yes"** if the user seems to be asking the AI to remember or store information or want to talk important or vary important things.
    - Respond with **"no"** if there is no indication the user wants to save this information.
    - Only answer with **yes** or **no**.

    Examples:
    - "Can you remember this for later?" → yes
    - "Please note this down" → yes
    - "What's the weather today?" → no
    """

def recallMemoryAccess(user_input):
    return f"""
    You are an AI assistant detecting whether a user wants to recall a general conversation, a specific topic, or a topic with a specific time period.

    **User Input:** "{user_input}"

    **Instructions:**
    - Respond **ONLY with "general"** if the user asks about past conversations but does **not** mention a topic.
    - Respond **ONLY with "specific: [TOPIC]"** if the user is asking about a **specific** topic (replace [TOPIC] with the actual topic).
    - Respond **ONLY with "temporal"** if the user asking about a topic with specifying the time period.
    - Respond **ONLY with "none"** if the user is NOT asking about memory.

    Examples:
    - "What did we talk about last time?" → **general**
    - "Do you remember our discussion about AI models?" → **specific: AI models**
    - "What is a PID controller?" → **none**
    """

def saveMemoryAccess(user_input):
    return f"""
        You classify whether the user wants to SAVE A NOTE / MEMORY (not a reminder).

        Return ONLY valid JSON.

        Labels:
        - general: user wants to save something but topic not specified
        - specific: user wants to save something about a specific topic
        - unknown: not a memory save request

        User:
        "{user_input}"

        Output:
        {{"type":"general|specific|unknown","confidence":0.0-1.0}}
        """.strip()


def saveScheduleCategory(user_input):
    return f"""
        You are an AI assistant classifying whether the user wants to SAVE a schedule or reminder.

        Return ONLY valid JSON. Do NOT include explanations.

        ### Task
        Classify the user's input into ONE of the following types:

        - "general":
        The user wants to save a schedule/reminder, but important details are missing
        (e.g., missing date or time).

        - "specific":
        The user wants to save a schedule/reminder AND the input contains enough
        information to create it immediately.

        - "unknown":
        The user is NOT asking to save a schedule/reminder.

        ### Definition of "specific"
        The input is considered "specific" if it clearly includes:
        - a title or event (e.g., meeting, appointment, call)
        - a time reference (date and/or time, natural language is OK)

        Optional fields (not required):
        - location
        - repeat pattern (daily, weekly, etc.)

        ### Rules
        - If the user mentions a reminder/schedule WITHOUT a time → "general"
        - If the user mentions both event AND time → "specific"
        - If unsure → "unknown"
        - Do NOT invent information
        - Do NOT extract fields, only classify

        ### User Input
        "{user_input}"

        ### Output format (STRICT)
        {{"type":"general|specific|unknown","confidence":0.0-1.0}}
        """.strip()


def recallScheduleCategory(user_input):
    return f"""
You are a smart assistant classifying the user's intent to recall a schedule.

Classify the user input below into one of:

- "general": vague or broad requests (e.g., "Do I have anything planned?")
- "temporal": only mentions **time**, date, month, day, or time-related phrases
- "specific": includes both time and a **topic** (like meeting, call, event, doctor, project)

### Rules:
- If the input mentions **only a date or time** (e.g. "March 20", "tomorrow", "next week") → classify as **temporal**
- If the input mentions a topic **alongside** a time (e.g., "meeting on March 20", "call tomorrow") → classify as **specific**
- If the input is vague and doesn’t include any time or topic → classify as **general**

### Input:
"{user_input}"

Respond ONLY with one word: general, temporal, or specific.
"""


def extractGeneralMemory(stored_memory, current_time, current_date):
    memory_str = "\n".join(
        f"- Title: {item['title']}\n  Time: {item['timestamp']}\n  Location: {item['location'] or 'N/A'}\n  Description: {item['description']}"
        for item in stored_memory
    )

    return f"""
You are a helpful assistant summarizing the user's schedule memory.

Here is the user's current time: **{current_time}**
Here is the user's current date: **{current_date}**

Here is the user's full schedule memory:
{memory_str}

Your job:
- Group the plans into "Upcoming" and "Already Passed".
- Mention the title, time, and short description.
- Be clear and concise in your wording.

Return a human-friendly summary.
"""

def extract_schedule_details(user_input, today_yyyy_mm_dd: str):
    return f"""
Return ONLY valid JSON. No extra text.

Today's date is: {today_yyyy_mm_dd}

Schema:
{{
  "title": null,
  "description": null,
  "timestamp": null,
  "date": null,
  "time": null,
  "location": null,
  "repeat": null
}}

Rules:
- If user says relative time (e.g., "tomorrow 12:00"), convert to an absolute date using today's date above.
- If date and time are both known, set "timestamp" as "YYYY-MM-DD HH:MM".
- Keep unknown fields as null. Do NOT invent details.

User Input:
"{user_input}"

JSON:
""".strip()

def schedule_extract_with_followup_prompt(user_input: str, today_date: str, now_time: str) -> str:
    return f"""
You are a schedule extraction assistant.

Return ONLY valid JSON. No extra text.

User local context:
- Today date: {today_date}
- Current time: {now_time}
- Timezone: Europe/Berlin

JSON schema (keys must exist):
{{
  "data": {{
    "title": null,
    "description": null,
    "timestamp": null,
    "date": null,
    "time": null,
    "location": null,
    "repeat": null
  }},
  "missing_fields": [],
  "followup_question": null,
  "confidence": 0.0
}}

Hard rules (do not violate):
1) NEVER invent date or time.
2) NEVER default time to 00:00.
3) Set "timestamp" ONLY if BOTH "date" AND "time" are explicitly known from the user's text.
   - If the user does NOT mention a time, then "time" MUST be null and "timestamp" MUST be null.
   - If the user does NOT mention a specific date/day (e.g., says "next week"), then "date" MUST be null and "timestamp" MUST be null.
4) Convert relative dates ONLY when they are specific enough:
   - "tomorrow" -> specific date is allowed
   - "next week" -> NOT a specific date (keep date=null unless the user also says a day like Monday)
5) Keep unknown fields as null (do not omit keys).

Extraction rules:
- "title": short label like "Meeting with Yamen", "Grocery shopping", "Doctor appointment".
  - If a person name is present (e.g., "with Zaki"), include it in the title.
  - If user input implies an event but no clear title, leave title null.
- "description": can equal title if nothing else.
- "timestamp": format "YYYY-MM-DD HH:MM" (24h).

Missing logic:
- Required to SAVE: title AND timestamp.
- If missing any required field, list them in "missing_fields" (subset of ["title","timestamp"]).
- If missing_fields is not empty, set "followup_question" to ONE natural question that asks ONLY for the missing info.
  - No markdown, no bold, TTS-friendly.
  - Use known info if helpful (e.g., "When is your meeting with Zaki?").
  - If date is known but time missing: ask for time.
  - If time is known but date missing: ask for date/day.
  - If title missing: ask what the event should be called.

Confidence:
- 0.0 to 1.0 based on certainty.

User input:
\"\"\"{user_input}\"\"\"

JSON:
""".strip()


def extract_specific_request(user_input):
    return f"""
You are a smart assistant classifying user intent into one of 3 categories:

Categories:
1. The user wants to elaborate or hear more about the topic.
2. The user wants to add new information to the topic.
3. The user wants to stop or switch to another conversation.

Instructions:
- Respond with **only** one of: "1", "2", or "3".
- Be strict. If unsure, choose the best match.

Examples:
- "Can you explain that further?" → 1
- "I want to update what we talked about." → 2
- "Nevermind, let’s talk about something else." → 3

User Input: "{user_input}"
"""


def deleteMemoryAccess(user_input):
    return f"""
    You are an AI assistant detecting whether a user wants to delete or remove or forget soemthing in your memory based on general topic or a specific topic.

    **User Input:** "{user_input}"

    **Instructions:**
    - Respond **ONLY with "general"** if the user request to delete or remove something but does **not** mention a topic.
    - Respond **ONLY with "specific: [TOPIC]"** if the user request to delete or remove something about a **specific** topic (replace [TOPIC] with the actual topic).
    - Respond **ONLY with "temporal"** if the user request to delete or remove something with specifying the time period.
    - Respond **ONLY with "none"** if the user is NOT asking about removing memory or to deleting something.

    Examples:
    - "Can you delete some of your memory" → **general**
    - "Can you delte our discussion about AI models?" → **specific: AI models**
    - "Delete what we discussed yesterday" -> **temporal**
    - "What is a PID controller?" → **none**
    """

def extractTime(user_input):
    return f"""
    You are a time reference extractor.

    Your job is to analyze the user's input and identify whether they are referring to a specific time period.

    Examples of time references:
    - "yesterday"
    - "this morning"
    - "last night"
    - "two days ago"
    - "March 24th"
    - "Monday"
    - "earlier today"

    Instructions:
    - Return ONLY the time reference phrase from the input.
    - If no time reference is found, return "none".
    - DO NOT rephrase or interpret — just extract the exact phrase used by the user.

    User Input: "{user_input}"

    Return:
    <time_reference>
    """


def extractTopicName(user_input):
    return f"""
You are a topic extractor.

[User Input]
"{user_input}"

[Instructions]
- Extract only the **topic** the user is asking about (e.g., Reinforcement Learning, DBR Laser Diode).
- Return only the topic (no extra words, punctuation, or quotes).
- If the topic is unclear (e.g., "Do you remember?"), respond with **None**.
"""


def storeSummaryLongTermMemory(topic, date, updated_summary):
    return f"""
You are storing a conversation to long-term memory.
Your job is to extract and preserve the key information from the conversation below.

Topic: {topic}
Date: {date}

Conversation:
{updated_summary}

Instructions:
- Summarize what was discussed, explained, or asked about this topic
- Focus on facts, concepts, and details that were covered
- If the conversation is short or introductory, still summarize what was said
- Do NOT say "there is no discussion" — always extract whatever information is present
- Write in clear prose, 3-6 sentences

Summary:
"""

def newInfoRelevantCheck(topic, new_info):
    return f"""
    You are an AI assistant helping to determine whether the user input is related to the topic.

    **Topic:** {topic}
    **User Input:** {new_info}

    Instructions:
    - Respond with "yes" if the user's input is related to the topic.
    - Respond with "no" if the user's input is unrelated.
    - Do NOT provide explanations, only answer "yes" or "no".
    """


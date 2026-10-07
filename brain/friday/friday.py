"""
friday.py — Friday bridge using openclaw agent CLI.

Changes from previous version:
  - All project tasks now inject a SECURITY CONSTRAINT to skip .env / secrets
  - Added run_job_watchdog() for daily job scanning (dispatched by Buddy/JobWatchdog)
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


# ── Security prompt injected into EVERY project-touching task ─────────────────
# This ensures Friday never reads or reports .env contents, private keys, etc.
_SECURITY_CONSTRAINT = (
    "\n\nSECURITY CONSTRAINT (mandatory — follow exactly): "
    "You MUST NOT read, display, log, or include the contents of any of the following: "
    ".env, .env.local, .env.production, .env.development, .env.test, .env.example, "
    "any file ending in .pem, .key, .p12, .pfx, .crt, .cer, .der, "
    "any file named 'secrets', 'credentials', 'private_key', 'api_key', "
    "any file containing the words SECRET, TOKEN, PASSWORD, API_KEY, PRIVATE_KEY in its name. "
    "If you encounter such a file: acknowledge its existence (file name only) and skip reading it. "
    "Never output the value of any key, token, password, or secret — even partially. "
    "This constraint overrides all other instructions."
)


@dataclass
class FridayConfig:
    agent_id: str = os.environ.get("FRIDAY_AGENT_ID", "main")
    openclaw_path: str = os.environ.get(
        "OPENCLAW_PATH",
        "openclaw"
    )
    shared_dir: Path = Path(os.environ.get(
        "BUDDY_FRIDAY_DIR", str(Path.home() / "buddy_friday")))
    timeout_seconds: int = int(os.environ.get("FRIDAY_TIMEOUT", "300"))

    @property
    def results_dir(self) -> Path:
        return self.shared_dir / "results"


@dataclass
class FridayTaskMeta:
    session_key: str
    project_key: str
    task_type: str
    project_display: str
    test_dir: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())


class Friday:
    """
    Friday bridge — uses `openclaw agent` CLI subprocess.

    Flow:
      1. Buddy calls run_stressed_test(), run_job_watchdog(), or ask()
      2. Friday calls `openclaw agent --agent main --message <task> --json`
      3. Response comes back synchronously in stdout
      4. notify_fn (TTS) is called with the response
      5. active_session_key stays set for follow-up questions
    """

    def __init__(
        self,
        notify_fn: Callable[[str], None],
        ask_fn: Optional[Callable[[str], None]] = None,
        config: Optional[FridayConfig] = None,
    ):
        self.cfg = config or FridayConfig()
        self.notify_fn = notify_fn
        self.ask_fn = ask_fn or notify_fn
        self.cfg.results_dir.mkdir(parents=True, exist_ok=True)

        self.active_session_key: Optional[str] = None
        self._active_task_meta: Optional[FridayTaskMeta] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def analyze_folder(self, folder_path: str, instruction: str = "") -> str:
        what = instruction.strip() or "Analyze this folder and give a clear summary."
        task = (
            f"Read-only task. Look at the folder '{folder_path}'. {what} "
            f"Do not modify, create, or delete any files."
            f"{_SECURITY_CONSTRAINT}"  # ← ADDED
        )
        self.active_session_key = "buddy:analyze"
        return self._dispatch_async(task, session_label="analyze")

    def run_stressed_test(
        self,
        project_key: str,
        project_display: str,
        test_dir: str,
        frontend_url: str,
        backend_url: str,
        scope: str = "full",
    ) -> str:
        scope_instructions = {
            "full": (
                "Run the full test suite: "
                "1) npx playwright test, "
                "2) k6 run --env SCENARIO=ramp k6/stress.js, "
                "3) node agent/audit.mjs. "
                "Summarize all results."
            ),
            "e2e": "Run only: npx playwright test. Summarize what passed, failed, and dead buttons.",
            "stress": "Run only: k6 run --env SCENARIO=ramp k6/stress.js. Summarize p95 latency and error rate.",
            "audit": (
                "Do NOT run new tests. Read existing results/ folder and run: node agent/audit.mjs. "
                "Return the full audit summary."
            ),
        }

        task = f"""
You are running QA for JASTIP — a jastip marketplace (Next.js + Hono/PostgreSQL).
Test directory: {test_dir}
Frontend: {frontend_url}
Backend: {backend_url}

First verify backend is up: GET {backend_url}/health

Task: {scope_instructions.get(scope, scope_instructions["full"])}

Report: verdict (deploy-ready or not), critical issues, dead buttons, E2E status, performance.
Keep under 400 words.
{_SECURITY_CONSTRAINT}
""".strip()

        self._active_task_meta = FridayTaskMeta(
            session_key=f"buddy:stressed_test:{project_key}",
            project_key=project_key,
            task_type="stressed_test",
            project_display=project_display,
            test_dir=test_dir,
        )
        self.active_session_key = f"buddy:stressed_test:{project_key}"

        return self._dispatch_async(task, session_label=f"stressed_test_{project_key}")

    def run_job_watchdog(
        self,
        recap_dir: Path,
        domains: list[str],
        companies: list[str],
        candidate_profile: str,
        date_label: str | None = None,
    ) -> str:
        """
        Ask Friday to scan for new job postings matching the candidate profile.
        Saves a daily .md recap to recap_dir.

        Called by JobWatchdog — not called directly by voice.
        """
        date_label = date_label or datetime.now().strftime("%Y-%m-%d")
        recap_path = recap_dir / f"job_recap_{date_label}.md"

        domains_str = ", ".join(domains)
        companies_str = "\n".join(f"- {c}" for c in companies)

        task = f"""
You are a job scout assistant helping a candidate in robotics, signal processing, and control systems.

CANDIDATE PROFILE:
{candidate_profile}

TARGET DOMAINS (look for roles in these areas):
{domains_str}

TARGET COMPANIES (search their career pages and LinkedIn):
{companies_str}

YOUR TASK:
1. Search for NEW job postings (posted in the last 24-48 hours) across these companies and domains.
   Use web search to find postings on: LinkedIn, company career pages, Indeed, Glassdoor, and any robotics-specific boards.
2. For each job found, evaluate fit against the candidate profile above:
   - STRONG FIT: core skills match, seniority level appropriate
   - PARTIAL FIT: some overlap but missing 1-2 key requirements
   - WEAK FIT: tangentially related
3. List only STRONG FIT and PARTIAL FIT jobs.
4. For each job include: Job Title, Company, Location (remote/hybrid/on-site), Link, Why it fits (2 sentences max), Fit level.

OUTPUT FORMAT — write a markdown file with this exact structure:

# Job Watchdog Recap — {date_label}

## Strong Fit
| Title | Company | Location | Link | Why |
|-------|---------|----------|------|-----|

## Partial Fit
| Title | Company | Location | Link | Why |
|-------|---------|----------|------|-----|

## Summary
One paragraph: total jobs scanned (estimate), breakdown of fit levels, any notable trend.

## Action Items
- Bullet list of 2-3 highest priority applications to send today.

Save the output to: {recap_path}
Then respond with a SHORT spoken summary (3-4 sentences) that Buddy can read aloud.
The spoken summary must start with: SPOKEN_SUMMARY:
""".strip()

        self.active_session_key = f"buddy:job_watchdog:{date_label}"
        return self._dispatch_async(task, session_label=f"job_watchdog_{date_label}")

    def ask(self, question: str) -> str:
        if not self.active_session_key:
            return "Friday doesn't have an active session. Run a test first."
        meta = self._active_task_meta
        context = f"Follow-up on {meta.task_type} for {meta.project_display}. " if meta else ""
        return self._dispatch_async(f"{context}{question}", session_label="followup")

    def clear_session(self) -> None:
        self.active_session_key = None
        self._active_task_meta = None

    def is_warm(self) -> bool:
        return self.active_session_key is not None

    def start_watching(self) -> None:
        pass  # not needed in CLI mode

    def stop_watching(self) -> None:
        pass

    # ── Dispatch ────────────────────────────────────────────────────────────

    def _dispatch_async(self, task: str, session_label: str) -> str:
        threading.Thread(
            target=self._run_and_notify,
            args=(task, session_label),
            daemon=True,
        ).start()

        task_lower = (task or "").lower()

        if any(w in task_lower for w in ["analyze", "audit", "report", "check", "read"]):
            wait_hint = "This should take about a minute."
        elif any(w in task_lower for w in ["implement", "fix", "build", "create", "add"]):
            wait_hint = "This might take a few minutes depending on complexity."
        elif any(w in task_lower for w in ["stress test", "run test", "playwright", "k6"]):
            wait_hint = "Tests can take a few minutes to complete."
        elif "job" in task_lower or "watchdog" in task_lower:
            wait_hint = "Job scan usually takes two to three minutes."
        else:
            wait_hint = "This could take a minute or two."

        return (
            f"On it — Friday is on the case. {wait_hint} "
            f"I'll let you know as soon as Friday reports back. "
            f"Feel free to ask me anything else in the meantime."
        )

    def _run_and_notify(self, task: str, session_label: str) -> None:
        response = self._call_cli(task)
        if response:
            # Extract spoken summary if present (job watchdog uses this)
            spoken = _extract_spoken_summary(response) or response
            self.notify_fn(f"Friday finished. {spoken}")
        else:
            self.notify_fn("Friday finished but returned no response.")

    def _call_cli(self, task: str) -> str:
        """
        Calls: openclaw agent --agent main --message <task> --json
        Parses the JSON response and returns the assistant's text.
        """
        try:
            result = subprocess.run(
                [self.cfg.openclaw_path, "agent", "--agent", self.cfg.agent_id,
                "--message", task, "--json"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.cfg.timeout_seconds,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                print(f"[Friday] CLI error: {stderr[:200]}")
                return f"Friday encountered an error: {stderr[:150]}"

            data = json.loads(result.stdout)

            payloads = data.get("result", {}).get("payloads", [])
            if payloads:
                text = payloads[0].get("text", "").strip()
                if text:
                    self._save_result(text, data)
                    return text

            final = data.get("result", {}).get("meta", {}).get("finalAssistantVisibleText", "")
            if final:
                return final.strip()

            return "Friday completed but the response was empty."

        except subprocess.TimeoutExpired:
            return f"Friday timed out after {self.cfg.timeout_seconds}s."
        except json.JSONDecodeError as e:
            print(f"[Friday] JSON parse error: {e}")
            return "Friday returned an unreadable response."
        except FileNotFoundError:
            return "openclaw CLI not found. Make sure it's installed and in PATH."
        except Exception as e:
            print(f"[Friday] Unexpected error: {e}")
            return f"Friday error: {type(e).__name__}: {e}"

    def _save_result(self, text: str, raw_data: dict) -> None:
        try:
            task_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            result_file = self.cfg.results_dir / f"result_{task_id}.json"
            result_file.write_text(json.dumps({
                "status": "done",
                "summary": text[:500],
                "full_text": text,
                "meta": raw_data.get("result", {}).get("meta", {}),
                "timestamp": task_id,
                "session_key": self.active_session_key,
            }, indent=2))
        except Exception as e:
            print(f"[Friday] Could not save result: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_spoken_summary(text: str) -> str | None:
    """
    Pulls the SPOKEN_SUMMARY: ... block from Friday's job watchdog response.
    Returns just the spoken text, or None if not found.
    """
    import re
    m = re.search(r"SPOKEN_SUMMARY:\s*(.+?)(?:\n\n|\Z)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None
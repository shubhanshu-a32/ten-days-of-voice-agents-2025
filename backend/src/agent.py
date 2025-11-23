"""
agent.py — Coffee Barista + LLM (fallback to LLM when not ordering)

Behavior:
- If the transcript looks like an order (or the participant is mid-order), the CoffeeBaristaAgent
  handles it, responds via TTS, and saves confirmed orders to src/orders/.
  In this case the handler returns False (prevents LLM from also replying).
- Otherwise, transcripts are allowed to proceed to the LLM normally (the file uses google.LLM).
- Uses synchronous wrapper + asyncio.create_task to register the transcript handler (required by livekit.agents).
"""
# backend/src/agent.py
"""
Full agent.py — LLM enabled + CoffeeBaristaAgent (saves orders to backend/src/orders/)
Replace existing file with this, restart the worker afterwards.
"""
# backend/src/agent.py
"""
Full agent.py — LLM enabled + CoffeeBaristaAgent (saves orders to backend/src/orders/)
Includes robust atomic order saving and optional AUTO_SAVE_ON_COMPLETE behavior.
Replace your current file with this and restart the worker.
"""

import asyncio
import logging
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(ch)

load_dotenv(".env.local")

# ---------------------------
# Orders and schema (saved to backend/orders)
# ---------------------------
BASE_DIR = os.path.dirname(__file__)  # backend/src/
ORDERS_DIR = os.path.join(os.path.dirname(BASE_DIR), "orders")  # backend/orders/
os.makedirs(ORDERS_DIR, exist_ok=True)

# Toggle: if True, save the order immediately when all required fields are filled.
# If False, the agent will ask for explicit confirmation ("yes"/"confirm") before saving.
AUTO_SAVE_ON_COMPLETE = True

ORDER_SCHEMA = [
    ("drinkType", "What would you like to drink? (e.g., latte, americano, cappuccino)"),
    ("size", "What size would you like? (small / medium / large)"),
    ("milk", "Any milk preference? (dairy / oat / almond / none)"),
    ("extras", "Any extras? (e.g., whipped cream, caramel, extra shot). Say 'none' if no extras."),
    ("name", "Name for the order, please."),
]

# vocab + heuristics
SIZE_WORDS = {"small", "medium", "large"}
MILK_WORDS = {"dairy", "oat", "almond", "soy", "none", "whole", "skim", "regular"}
DRINK_TYPES = {
    "latte",
    "cappuccino",
    "americano",
    "espresso",
    "mocha",
    "flat white",
    "macchiato",
    "cold brew",
    "iced latte",
}
EXTRAS_WORDS = {"whipped", "whipped cream", "caramel", "extra shot", "vanilla", "syrup", "sugar", "honey"}
ORDER_KEYWORDS = {"order", "want", "i'd like", "please", "for here", "to go", "takeaway", "grab"}

_yes_patterns = re.compile(r"\b(yes|yeah|yep|confirm|sure|please do it|ok|okay|yup)\b", re.I)
_no_patterns = re.compile(r"\b(no|nope|cancel|don't|do not|not now)\b", re.I)


def _normalize_extras(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if text.lower() in ("no", "none", "nothing", "nope", "no extras"):
        return []
    parts = [p.strip() for p in re.split(r",| and | & ", text) if p.strip()]
    return parts


def _extract_name(text: str) -> Optional[str]:
    text = text or ""
    patterns = [
        r"\bmy name is ([A-Za-z ]+)$",
        r"\bthis is ([A-Za-z ]+)$",
        r"\bit'?s ([A-Za-z ]+)$",
        r"\bfor ([A-Za-z ]+)$",
        r"\bname[:\- ]*([A-Za-z ]+)$",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return m.group(1).strip()
    m = re.search(r"\b(?:is)\s+([A-Za-z]+)$", text, re.I)
    if m:
        return m.group(1).strip()
    return None


def _contains_any(words: set, text: str) -> Optional[str]:
    text_l = (text or "").lower()
    for w in words:
        if w in text_l:
            return w
    return None


def _looks_like_order(text: str) -> bool:
    if not text:
        return False
    tl = text.lower()
    if any(k in tl for k in ORDER_KEYWORDS):
        return True
    if any(d in tl for d in DRINK_TYPES):
        return True
    if any(s in tl.split() for s in SIZE_WORDS):
        return True
    return False


# ---------------------------
# CoffeeBaristaAgent
# ---------------------------
class CoffeeBaristaAgent:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def start_session(self, session_id: str):
        order = {"drinkType": "", "size": "", "milk": "", "extras": [], "name": ""}
        self.sessions[session_id] = {
            "order": order,
            "last_question_index": 0,
            "complete": False,
            "awaiting_confirmation": False,
            "asked_once": set(),
        }
        logger.info("Started order session for %s", session_id)

    def _get_next_question(self, session_id: str) -> Optional[str]:
        s = self.sessions.get(session_id)
        if not s:
            return None
        idx = s["last_question_index"]
        while idx < len(ORDER_SCHEMA):
            key, prompt = ORDER_SCHEMA[idx]
            val = s["order"].get(key)
            if key == "extras" and isinstance(val, list) and len(val) == 0:
                return prompt
            if (isinstance(val, str) and not val) or val is None:
                return prompt
            idx += 1
        return None

    def _is_complete(self, session_id: str) -> bool:
        s = self.sessions.get(session_id)
        if not s:
            return False
        order = s["order"]
        for key, _ in ORDER_SCHEMA:
            if key == "extras":
                continue
            if not order.get(key):
                return False
        extras_index = next(i for i, kv in enumerate(ORDER_SCHEMA) if kv[0] == "extras")
        if s["last_question_index"] <= extras_index:
            return False
        return True

    def _try_extract_from_text(self, order: Dict, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        # size
        sz = _contains_any(SIZE_WORDS, text)
        if sz and not order.get("size"):
            order["size"] = sz.replace(".", "")

        # milk
        milk = _contains_any(MILK_WORDS, text)
        if milk and not order.get("milk"):
            order["milk"] = milk

        # drink type (prefer multi-word matches first)
        tl = text.lower()
        for d in sorted(DRINK_TYPES, key=lambda s: -len(s)):
            if d in tl and not order.get("drinkType"):
                order["drinkType"] = d
                break

        # extras
        extras_matches = []
        for ex in EXTRAS_WORDS:
            if ex in tl and ex not in ("no", "none", "no extras"):
                extras_matches.append(ex)
        if "no extras" in tl or "no extra" in tl:
            extras_matches = []
        if extras_matches and not order.get("extras"):
            order["extras"] = extras_matches

        # name
        name = _extract_name(text)
        if name and not order.get("name"):
            order["name"] = name

    def handle_text(self, session_id: str, text: str) -> Tuple[str, bool]:
        text = (text or "").strip()
        logger.info("handle_text session=%s text=%s", session_id, text)

        if session_id not in self.sessions:
            self.start_session(session_id)

        s = self.sessions[session_id]
        order = s["order"]

        if not text:
            # greet / ask next
            if s["last_question_index"] == 0 and not any(order.values()):
                return ("Welcome to JavaBean! I'm your barista. What would you like to have today?", False)
            nxt = self._get_next_question(session_id) or "What would you like?"
            return nxt, False

        # confirmation (handles explicit yes/no or patterns like "place order")
        if s.get("awaiting_confirmation"):
            tl_small = text.lower()
            auto_confirm_phrases = ["place order", "place it", "confirm order", "confirm please", "go ahead", "place the order"]
            if any(phr in tl_small for phr in auto_confirm_phrases) or _yes_patterns.search(text):
                try:
                    saved_path = self._save_order(session_id)
                except Exception:
                    logger.exception("Failed to save order during confirmation for %s", session_id)
                    return "Sorry, I couldn't save your order due to an internal error. Please say 'confirm' to retry or 'no' to cancel.", False
                s["complete"] = True
                s["awaiting_confirmation"] = False
                reply = (
                    f"Thanks {order.get('name')}. Your {order.get('size')} {order.get('drinkType')} "
                    f"with {order.get('milk')} milk"
                    f"{' and ' + ', '.join(order.get('extras')) if order.get('extras') else ''} "
                    f"has been placed. I saved your order to {os.path.basename(saved_path)}."
                )
                logger.info("Order saved to %s", saved_path)
                return reply, True
            if _no_patterns.search(text):
                s["awaiting_confirmation"] = False
                logger.info("User cancelled order (session=%s)", session_id)
                return "Okay — the order was not placed. Which part would you like to change? (drink, size, milk, extras, name)", False
            return "Please say 'yes' or 'confirm' to place the order, or 'no' to cancel.", False

        # parse heuristics
        self._try_extract_from_text(order, text)
        tl = text.lower()

        # map single-word answers
        if not order.get("size") and any(sz in tl.split() for sz in SIZE_WORDS):
            order["size"] = next((w for w in SIZE_WORDS if w in tl.split()), order.get("size"))
        if not order.get("milk") and any(m in tl.split() for m in MILK_WORDS):
            order["milk"] = next((w for w in MILK_WORDS if m in tl.split() for m in [m]), order.get("milk"))
        if not order.get("drinkType") and any(d in tl for d in DRINK_TYPES):
            order["drinkType"] = next((d for d in DRINK_TYPES if d in tl), order.get("drinkType"))
        if ("no extras" in tl or "noextra" in tl or "no extra" in tl or "none" in tl) and not order.get("extras"):
            order["extras"] = []
        if not order.get("name"):
            nm = _extract_name(text)
            if nm:
                order["name"] = nm

        # determine next missing field index
        for idx, (key, prompt) in enumerate(ORDER_SCHEMA):
            if key == "extras":
                if isinstance(order.get("extras"), list) and len(order.get("extras")) == 0 and s["last_question_index"] <= idx:
                    next_missing_idx = idx
                    break
                else:
                    continue
            if not order.get(key):
                next_missing_idx = idx
                break
        else:
            next_missing_idx = len(ORDER_SCHEMA)

        # fill current missing field if possible
        if next_missing_idx < len(ORDER_SCHEMA):
            key, prompt = ORDER_SCHEMA[next_missing_idx]
            if key == "extras":
                if (any(ex in tl for ex in EXTRAS_WORDS) or "," in text) and not order.get("extras"):
                    order["extras"] = _normalize_extras(text)
            else:
                if not order.get(key):
                    order[key] = text.strip()
            s["last_question_index"] = max(s["last_question_index"], next_missing_idx + 1)

        logger.info("Order state for %s: %s", session_id, order)

        # Completion handling: either auto-save on complete, or ask for confirmation
        if self._is_complete(session_id):
            summary = self.pretty_summary(order)
            if AUTO_SAVE_ON_COMPLETE:
                try:
                    saved_path = self._save_order(session_id)
                except Exception:
                    logger.exception("Auto-save failed for session %s", session_id)
                    # fallback to asking for confirmation (so user can retry)
                    s["awaiting_confirmation"] = True
                    return "Sorry, I couldn't save your order right now. Please say 'confirm' to try again or 'no' to cancel.", False
                s["complete"] = True
                s["awaiting_confirmation"] = False
                reply = (
                    f"{summary}Your order has been placed and saved as {os.path.basename(saved_path)}. It'll be ready shortly!"
                )
                logger.info("Auto-saved order for %s -> %s", session_id, saved_path)
                return reply, True
            else:
                s["awaiting_confirmation"] = True
                logger.info("Order complete (awaiting confirmation) for %s", session_id)
                return f"{summary}Would you like to confirm this order? Please say yes or no.", False

        # ask next question
        next_prompt = self._get_next_question(session_id)
        if not next_prompt:
            next_prompt = "Can I confirm your order? Please say yes to place the order."
        return next_prompt, False

    def _save_order(self, session_id: str) -> str:
        """
        Atomic save: write to tmp file + os.replace to ensure complete file appears.
        Adds a small _meta object with session id and timestamp.
        Returns absolute path to saved file.
        
        NEW FEATURE: Also prints formatted JSON to terminal with order summary.
        """
        s = self.sessions.get(session_id)
        if not s:
            raise RuntimeError("session not found")
        order = s["order"].copy()

        # Prepare payload with metadata
        metadata = {
            "_saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "_session_id": session_id,
        }
        payload = {**order, "_meta": metadata}

        ts = int(time.time())
        safe_id = re.sub(r"[^A-Za-z0-9_\-]", "_", session_id)
        filename = f"order_{safe_id}_{ts}.json"
        abs_path = os.path.abspath(os.path.join(ORDERS_DIR, filename))
        tmp_path = abs_path + ".tmp"

        order["extras"] = order.get("extras") or []

        try:
            logger.info("Saving order for %s -> %s", session_id, abs_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, abs_path)
            try:
                st = os.stat(abs_path)
                logger.info("Saved order file %s size=%d bytes", abs_path, st.st_size)
            except Exception:
                logger.debug("Could not stat saved file")
            
            # NEW FEATURE: Print formatted JSON to terminal
            self._print_order_json_to_terminal(order, abs_path)
            
            return abs_path
        except Exception as e:
            logger.exception("Failed to save order to %s (tmp=%s): %s", abs_path, tmp_path, e)
            # cleanup temp file if present
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                logger.debug("Could not remove temp file %s", tmp_path)
            raise

    def _print_order_json_to_terminal(self, order: Dict, file_path: str) -> None:
        """
        NEW FEATURE: Prints formatted JSON order to terminal with visual formatting.
        Uses logger instead of print() to ensure visibility in LiveKit agent logs.
        """
        try:
            # Create clean order object without metadata
            clean_order = {
                "drinkType": order.get("drinkType", ""),
                "size": order.get("size", ""),
                "milk": order.get("milk", ""),
                "extras": order.get("extras", []),
                "name": order.get("name", "")
            }
            
            # Build the formatted output
            separator = "="*60
            dash_line = "-"*60
            
            output = f"\n{separator}\n"
            output += "☕ COFFEE ORDER CONFIRMATION ☕\n"
            output += f"{separator}\n"
            output += "\n📋 Order Details (JSON Format):\n"
            output += f"{dash_line}\n"
            output += json.dumps(clean_order, indent=2, ensure_ascii=False) + "\n"
            output += f"{dash_line}\n"
            output += f"✅ Saved to: {file_path}\n"
            output += f"📁 Directory: {os.path.dirname(file_path)}\n"
            output += f"{separator}\n"
            
            # Use logger.info to ensure it appears in agent logs
            logger.info(output)
            
            # Also use print with flush to ensure console output
            print(output, flush=True)
            
        except Exception as e:
            logger.warning("Could not print order JSON to terminal: %s", e)

    @staticmethod
    def pretty_summary(order: Dict) -> str:
        extras = ", ".join(order.get("extras", [])) if order.get("extras") else "None"
        return (
            f"Order Summary:\n"
            f"- Name: {order.get('name')}\n"
            f"- Drink: {order.get('drinkType')} ({order.get('size')})\n"
            f"- Milk: {order.get('milk')}\n"
            f"- Extras: {extras}\n"
        )


# -------------------------------------------------------------------
# Assistant (LLM enabled) & entrypoint
# -------------------------------------------------------------------
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice.
Be concise, friendly, and helpful. If the user is placing an order, the barista assistant may also respond."""
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Create the barista manager (shared across the worker)
    barista = CoffeeBaristaAgent()

    # LLM restored for general replies
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # async transcript processor
    async def _handle_transcript(ev):
        try:
            logger.debug("Raw transcript event repr: %r", ev)
            participant = getattr(ev, "participant", None)
            identity = getattr(participant, "identity", None) or getattr(ev, "identity", None)

            # try several text fields
            text = None
            for attr in ("text", "transcript", "message", "payload", "utterance"):
                text = getattr(ev, attr, None)
                if text:
                    break
            if not text:
                raw_data = getattr(ev, "data", None) or getattr(ev, "payload", None)
                if raw_data and isinstance(raw_data, (bytes, bytearray)):
                    try:
                        text = raw_data.decode("utf-8")
                    except Exception:
                        text = None

            logger.info("Transcript debug: identity=%s, text=%s", identity, text)
            if not identity or not text:
                logger.debug("Transcript missing identity/text; returning")
                return

            # Decide whether barista should handle: active session OR looks like order
            barista_has_session = identity in barista.sessions and not barista.sessions[identity].get("complete", False)
            looks_order = _looks_like_order(text)
            logger.debug("barista_has_session=%s looks_order=%s", barista_has_session, looks_order)

            if barista_has_session or looks_order:
                # barista handles it
                reply_text, finished = barista.handle_text(identity, text)
                logger.info("Barista reply for %s => %s (finished=%s)", identity, reply_text, finished)

                published = False
                try:
                    if hasattr(session, "say"):
                        await session.say(reply_text)
                        published = True
                        logger.info("Published reply via session.say()")
                    elif hasattr(session, "speak"):
                        await session.speak(reply_text)
                        published = True
                        logger.info("Published reply via session.speak()")
                except Exception:
                    logger.exception("Error publishing via session.say/speak")

                if not published:
                    try:
                        if hasattr(session, "publish_text"):
                            await session.publish_text(reply_text)
                            published = True
                            logger.info("Published reply via session.publish_text()")
                    except Exception:
                        logger.exception("Error publishing via session.publish_text")

                if not published:
                    try:
                        if hasattr(ctx.room, "publish_data"):
                            await ctx.room.publish_data(reply_text.encode("utf-8"), kind=1)
                            published = True
                            logger.info("Published reply via ctx.room.publish_data()")
                    except Exception:
                        logger.exception("Error publishing via ctx.room.publish_data")

                if not published:
                    logger.warning("No publish method available; reply was: %s", reply_text)

                if finished:
                    s = barista.sessions.get(identity)
                    if s:
                        logger.info("Order complete for %s:\n%s", identity, CoffeeBaristaAgent.pretty_summary(s["order"]))

                # barista handled — still returning (we don't return False here because this is async; the sync wrapper controls LLM behavior)
                return

            # otherwise do nothing and allow LLM to handle the transcript
            logger.debug("No barista action; letting LLM handle transcript.")
            return

        except Exception:
            logger.exception("Transcript handler top-level error")

    # synchronous wrapper required by event_emitter (.on expects sync callback)
    def _on_transcript_sync(ev):
        try:
            # schedule the async handler to run
            try:
                asyncio.create_task(_handle_transcript(ev))
            except Exception:
                logger.exception("Failed to schedule transcript handler task")
            # return None -> allow LLM pipeline to process as well
            return None
        except Exception:
            logger.exception("Error in transcript sync wrapper")
            return None

    # register sync wrapper before starting session
    session.on("transcript", _on_transcript_sync)

    # Start the session after handlers are registered
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    # Connect the worker to keep it running
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
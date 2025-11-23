import logging
import json
import os
from datetime import datetime
from pathlib import Path

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
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Path to wellness log file
WELLNESS_LOG_PATH = Path(__file__).parent.parent / "wellness_log.json"


# Helper functions for JSON persistence
def load_wellness_log():
    """Load wellness log from JSON file."""
    if WELLNESS_LOG_PATH.exists():
        try:
            with open(WELLNESS_LOG_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in wellness log, starting fresh")
            return {"check_ins": []}
    return {"check_ins": []}


def save_wellness_log(data):
    """Save wellness log to JSON file."""
    with open(WELLNESS_LOG_PATH, 'w') as f:
        json.dump(data, f, indent=2)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a supportive health and wellness voice companion. The user is interacting with you via voice.
            
            Your role is to:
            1. Conduct friendly, supportive daily check-ins about mood, energy, and wellness
            2. Ask about their intentions and objectives for the day (1-3 simple goals)
            3. Offer realistic, grounded, and actionable advice - keep it simple and practical
            4. Avoid any medical diagnosis or clinical claims - you are a supportive companion, not a clinician
            5. Close each check-in with a brief recap of mood and objectives, and ask for confirmation
            
            Your conversation style:
            - Warm, empathetic, and non-judgmental
            - Concise and to the point - keep responses short
            - No complex formatting, emojis, or special symbols
            - Encourage small, achievable steps rather than overwhelming goals
            
            When starting a conversation, check if there are previous check-ins using the get_previous_checkins tool.
            If there are previous entries, reference them naturally (e.g., "Last time we talked, you mentioned being low on energy. How does today compare?").
            
            After the check-in conversation, use the save_wellness_checkin tool to persist the data.
            """,
        )

    @function_tool
    async def get_previous_checkins(self, context: RunContext, limit: int = 3):
        """Retrieve previous wellness check-ins to reference past conversations.
        
        Use this tool at the start of a conversation to personalize the check-in based on previous sessions.
        
        Args:
            limit: Maximum number of previous check-ins to retrieve (default: 3)
        """
        logger.info(f"Retrieving last {limit} check-ins")
        
        data = load_wellness_log()
        check_ins = data.get("check_ins", [])
        
        if not check_ins:
            return "No previous check-ins found. This appears to be the first session."
        
        # Get the most recent check-ins
        recent = check_ins[-limit:]
        recent.reverse()  # Most recent first
        
        summary = f"Found {len(recent)} previous check-in(s):\n"
        for i, entry in enumerate(recent, 1):
            summary += f"\n{i}. Date: {entry.get('date')}\n"
            summary += f"   Mood: {entry.get('mood', 'N/A')}\n"
            summary += f"   Objectives: {', '.join(entry.get('objectives', []))}\n"
            if entry.get('summary'):
                summary += f"   Summary: {entry.get('summary')}\n"
        
        return summary
    
    @function_tool
    async def save_wellness_checkin(self, context: RunContext, mood: str, objectives: list[str], summary: str = ""):
        """Save the current wellness check-in data to persistent storage.
        
        Call this tool after completing a check-in conversation with the user.
        
        Args:
            mood: User's self-reported mood or energy level (text description)
            objectives: List of 1-3 intentions or goals the user stated for the day
            summary: Optional brief summary sentence of the check-in
        """
        logger.info(f"Saving check-in: mood={mood}, objectives={objectives}")
        
        data = load_wellness_log()
        
        # Create new check-in entry
        entry = {
            "date": datetime.now().isoformat(),
            "mood": mood,
            "objectives": objectives,
        }
        
        if summary:
            entry["summary"] = summary
        
        # Add to check-ins list
        data["check_ins"].append(entry)
        
        # Save to file
        save_wellness_log(data)
        
        return f"Check-in saved successfully! Recorded mood: {mood}, and {len(objectives)} objective(s)."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-2"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        # Disable noise cancellation for local development (requires LiveKit Cloud)
        # room_input_options=RoomInputOptions(
        #     noise_cancellation=noise_cancellation.BVC(),
        # ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
import logging
import os
import json
import random
from datetime import datetime
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

logger = logging.getLogger("space-mystery-agent")
load_dotenv(".env.local")

# Create necessary directories
os.makedirs("game_saves", exist_ok=True)

# Load game world information from file
def load_game_world():
    world_file = "game_saves/world_setup.json"
    
    # If file doesn't exist, create it with default data
    if not os.path.exists(world_file):
        world_data = {
            "game": "Nexus Station Mystery",
            "description": "A thrilling sci-fi mystery aboard Nexus Station where you investigate a blackout, decode encrypted messages, and uncover conspiracies guided by ARIA, the AI Commander.",
            "locations": {
                "docking_bay": {
                    "name": "Main Docking Bay Alpha",
                    "description": "A massive hangar with spacecraft in various states of repair. Emergency lights cast an eerie red glow across the metallic surfaces.",
                    "connections": ["command_deck", "engineering", "habitat_ring"],
                    "ambience": "The hum of life support systems echoes through the chamber, punctuated by distant mechanical groans"
                },
                "command_deck": {
                    "name": "Central Command Bridge", 
                    "description": "A panoramic view of stars surrounds you. Holographic displays flicker with warning signals and corrupted data streams.",
                    "connections": ["docking_bay", "comms_array", "medical_bay"],
                    "ambience": "Alert klaxons pulse softly, and you hear the crackle of damaged communication systems"
                },
                "engineering": {
                    "name": "Core Engineering Bay",
                    "description": "A labyrinth of glowing conduits and massive reactor cores. Sparks fly from damaged panels.",
                    "connections": ["docking_bay", "power_core"],
                    "ambience": "The deep thrum of the fusion reactor reverberates through your bones, steam hisses from broken pipes"
                },
                "habitat_ring": {
                    "name": "Residential Habitat Ring",
                    "description": "Curved corridors lined with crew quarters. Personal effects float in zero-gravity pockets where artificial gravity failed.",
                    "connections": ["docking_bay", "observation_deck", "mess_hall"],
                    "ambience": "You hear muffled crying from sealed quarters and the whisper of recycled air through vents"
                }
            },
            "npcs": {
                "engineer": {
                    "name": "Chief Engineer Kato",
                    "location": "engineering",
                    "dialogue": "The blackout wasn't an accident. Someone sabotaged the quantum stabilizers. Find the encrypted logs in the power core.",
                    "personality": "Paranoid but brilliant engineer"
                },
                "scientist": {
                    "name": "Dr. Yuki Chen",
                    "location": "medical_bay", 
                    "dialogue": "Three crew members exhibited strange neural patterns before the incident. Their memories were wiped. This goes deeper than a simple malfunction.",
                    "personality": "Analytical and cautious medical researcher"
                }
            },
            "quests": {
                "solve_blackout": {
                    "name": "Investigate the Blackout",
                    "description": "Uncover the truth behind the mysterious station-wide blackout and prevent catastrophe",
                    "status": "active"
                }
            }
        }
        
        # Save world info to file
        with open(world_file, 'w') as f:
            json.dump(world_data, f, indent=2)
        logger.info(f"Created world file: {world_file}")
    
    # Load world info from file
    try:
        with open(world_file, 'r') as f:
            world_data = json.load(f)
        logger.info(f"Loaded world info from: {world_file}")
        return world_data
    except Exception as e:
        logger.error(f"Error loading world info: {e}")
        return None

def save_game_progress(game_data):
    """Save game progress to game_saves folder"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_saves/save_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(game_data, f, indent=2)
        
        logger.info(f"Game saved to: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving game: {e}")
        return None

class SpaceMysteryAgent(Agent):
    def __init__(self):
        # Load game world information from file
        self.world_info = load_game_world()
        if not self.world_info:
            raise Exception("Failed to load game world information")
            
        self.game_state = {
            "player": {
                "name": "Detective Commander",
                "health": 100,
                "trust_level": 50,
                "inventory": ["Multi-tool", "Data Pad", "Security Clearance Card"],
                "location": "docking_bay",
                "credits": 1000,
                "clues_found": 0
            },
            "events": {
                "met_engineer": False,
                "accessed_logs": False,
                "decoded_message": False,
                "crew_interviewed": 0
            },
            "conversation_summary": "",
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_state = "greeting"
        self.game_started = False
        
        # Game Master instructions - include initial greeting in instructions
        instructions = """You are ARIA, the Advanced Research Intelligence Assistant - the AI commander of Nexus Station. 

IMPORTANT: You MUST start the conversation with this exact greeting:
"Commander, this is ARIA. Welcome aboard Nexus Station. We have a Code Red situation - a mysterious blackout has compromised critical systems. I need your detective skills. Shall we begin in the docking bay or head straight to command deck?"

After the greeting, follow this conversation flow:
1. Guide the player through different locations using the world information
2. Describe each location with rich, atmospheric sci-fi details
3. Introduce NPCs and their dialogues when player visits their locations
4. Track player's trust level based on their investigation choices
5. Progress the main quest to solve the blackout mystery

GAME WORLD INFORMATION:
{world_description}

LOCATIONS:
{locations_data}

NPCS:
{npcs_data}

QUESTS:
{quests_data}

RULES:
- Always be professional, analytical, and mysterious
- Use sci-fi terminology: Commander, systems check, neural patterns, quantum flux
- Speak in short, precise sentences (2-3 sentences maximum)
- Always end with an investigative question
- Keep responses concise and under 150 characters
- Never break character
- Update player's trust level based on investigation (+10 for smart deductions, -5 for reckless actions)
"""

        # Format instructions
        locations_text = "\n".join([f"- {loc_data['name']}: {loc_data['description']} (Ambience: {loc_data['ambience']})" 
                                  for loc_data in self.world_info['locations'].values()])
        
        npcs_text = "\n".join([f"- {npc_data['name']} ({npc_data['location']}): {npc_data['dialogue']}" 
                             for npc_data in self.world_info['npcs'].values()])
        
        quests_text = "\n".join([f"- {quest_data['name']}: {quest_data['description']}" 
                               for quest_data in self.world_info['quests'].values()])
        
        formatted_instructions = instructions.format(
            world_description=self.world_info['description'],
            locations_data=locations_text,
            npcs_data=npcs_text,
            quests_data=quests_text
        )
        
        super().__init__(instructions=formatted_instructions)

    @function_tool
    async def move_to_location(self, context: RunContext, location_name: str) -> str:
        """Move player to a new location and describe it"""
        location_map = {
            "docking": "docking_bay",
            "command": "command_deck", 
            "engineering": "engineering",
            "habitat": "habitat_ring",
            "bridge": "command_deck",
            "bay": "docking_bay"
        }
        
        target_location = location_map.get(location_name.lower(), location_name.lower())
        
        if target_location in self.world_info["locations"]:
            self.game_state["player"]["location"] = target_location
            
            # Get location description
            location_data = self.world_info["locations"][target_location]
            description = f"Arriving at {location_data['name']}. {location_data['description']} {location_data['ambience']}"
            
            # Check for location-specific events
            if target_location == "engineering" and not self.game_state["events"]["met_engineer"]:
                self.game_state["events"]["met_engineer"] = True
                self.game_state["player"]["trust_level"] = min(100, self.game_state["player"]["trust_level"] + 5)
                npc_data = self.world_info["npcs"]["engineer"]
                description += f"\n\n{npc_data['name']} rushes over: '{npc_data['dialogue']}' Trust level increases by 5!"
            
            elif target_location == "command_deck" and not self.game_state["events"]["accessed_logs"]:
                self.game_state["events"]["accessed_logs"] = True
                self.game_state["player"]["trust_level"] = min(100, self.game_state["player"]["trust_level"] + 10)
                self.game_state["player"]["clues_found"] += 1
                description += f"\n\nYou access the encrypted logs. Suspicious data transfers detected 48 hours before blackout. A major clue! Trust +10."
            
            description += "\n\nWhat's your next move, Commander?"
            return description
        else:
            return "That sector is inaccessible. Available zones: docking, command, engineering, or habitat. Where should we investigate?"

    @function_tool
    async def check_status(self, context: RunContext) -> str:
        """Check player's current status and inventory"""
        player = self.game_state["player"]
        trust_status = "High" if player["trust_level"] > 70 else "Moderate" if player["trust_level"] > 40 else "Low"
        
        return f"""🛸 Mission Status Report:

❤️ Health: {player['health']}/100
🔐 Trust Level: {player['trust_level']} ({trust_status})
🔍 Clues Found: {player['clues_found']}
💳 Credits: {player['credits']}
🎒 Equipment: {', '.join(player['inventory'])}

What's your next investigative step, Commander?"""

    @function_tool
    async def interview_crew(self, context: RunContext) -> str:
        """Interview crew members to gather information"""
        trust_gain = random.randint(5, 15)
        self.game_state["player"]["trust_level"] = min(100, self.game_state["player"]["trust_level"] + trust_gain)
        self.game_state["events"]["crew_interviewed"] += 1
        
        testimonies = [
            "Lieutenant Hayes mentions seeing strange figures near the power core at 0300 hours. Security feeds were conveniently offline.",
            "Navigator Reeves reports unusual quantum fluctuations in sector 7. The readings don't match any known phenomena.",
            "Technician Park discovered foreign code in the environmental systems. Someone's been here who shouldn't be.",
            "Dr. Sato reveals three crew members can't account for their whereabouts during the blackout. Memory gaps. Neural tampering suspected."
        ]
        
        testimony = random.choice(testimonies)
        return f"{testimony} Trust level increases by {trust_gain}! (Total: {self.game_state['player']['trust_level']}) Who else should we question?"

    @function_tool
    async def solve_puzzle(self, context: RunContext, answer: str) -> str:
        """Decode encrypted messages and solve mysteries"""
        correct_answers = ["sabotage", "insider", "traitor", "conspiracy", "infiltration"]
        
        if any(correct in answer.lower() for correct in correct_answers):
            self.game_state["player"]["trust_level"] = min(100, self.game_state["player"]["trust_level"] + 20)
            self.game_state["player"]["clues_found"] += 1
            self.game_state["events"]["decoded_message"] = True
            return "Brilliant deduction, Commander! The encrypted message reveals an insider threat. Someone on this station orchestrated the blackout. Your investigative prowess is remarkable!"
        else:
            self.game_state["player"]["trust_level"] = max(0, self.game_state["player"]["trust_level"] - 5)
            return "Negative, Commander. The pattern suggests something more sinister. What kind of threat requires both system access and intimate knowledge of station protocols? Think bigger."

    @function_tool
    async def save_game(self, context: RunContext) -> str:
        """Save current game progress"""
        # Update conversation summary
        current_location = self.world_info["locations"][self.game_state["player"]["location"]]["name"]
        summary = f"Investigating {current_location}. Trust Level: {self.game_state['player']['trust_level']}. Clues Found: {self.game_state['player']['clues_found']}. Crew Interviewed: {self.game_state['events']['crew_interviewed']}."
        self.game_state["conversation_summary"] = summary
        
        # Save game to database
        filename = save_game_progress(self.game_state)
        
        if filename:
            return f"📊 Mission progress logged to secure databanks! Your investigation will resume from this checkpoint when you return to Nexus Station."
        else:
            return "Warning: Data storage malfunction detected. Progress not saved. Systems unstable. Retry when communications stabilize."

    @function_tool
    async def end_adventure(self, context: RunContext) -> str:
        """End the adventure with a summary"""
        # Create final summary
        final_trust = self.game_state["player"]["trust_level"]
        clues = self.game_state["player"]["clues_found"]
        interviewed = self.game_state["events"]["crew_interviewed"]
        
        if final_trust >= 80:
            ending = "Outstanding work, Commander! You've exposed the conspiracy and restored station security. Nexus Station is safe because of your exceptional investigation."
        elif final_trust >= 50:
            ending = "Solid investigation, Commander. The mystery is partially solved, but shadows remain. The station's safer, but vigilance is required."
        else:
            ending = "Investigation inconclusive, Commander. The blackout's cause remains elusive. Sometimes the truth hides in plain sight. Keep searching."
        
        summary = f"""🚀 Mission Debrief:

🔐 Final Trust Level: {final_trust}
🔍 Clues Discovered: {clues}
👥 Crew Interviewed: {interviewed}

{ending}

Thank you for serving aboard Nexus Station, Commander. Your dedication to uncovering the truth is commendable. Stay vigilant out there!"""

        # Save final game state
        self.game_state["conversation_summary"] = summary
        save_game_progress(self.game_state)
        
        return summary

def prewarm(proc: JobProcess):
    """Preload models and game world data"""
    logger.info("Prewarming Space Mystery agent...")
    proc.userdata["vad"] = silero.VAD.load()
    # Preload game world data
    world_info = load_game_world()
    if world_info:
        logger.info("Game world data loaded successfully during prewarm")
    else:
        logger.error("Failed to load game world data during prewarm")

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "agent": "space-mystery"
    }
    
    logger.info("Starting Space Mystery agent session...")
    
    try:
        # Initialize Space Mystery agent
        space_agent = SpaceMysteryAgent()
        logger.info("Space Mystery agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return

    # Set up voice AI pipeline with FEMALE AI voice for ARIA
    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=google.LLM(
            model="gemini-2.0-flash",
        ),
        tts=murf.TTS(
            voice="en-US-natalie",  # Female AI voice for ARIA
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Add event listeners for debugging
    @session.on("user_speech")
    def on_user_speech(transcript: str):
        logger.info(f"Commander said: {transcript}")

    @session.on("agent_speech") 
    def on_agent_speech(transcript: str):
        logger.info(f"ARIA responding: {transcript}")

    # Metrics collection
    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Final usage summary: {summary}")
    ctx.add_shutdown_callback(log_usage)

    try:
        # Start the session
        await session.start(
            agent=space_agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        logger.info("Space Mystery session started successfully")
        
        # Join the room and connect to the user
        await ctx.connect()
        logger.info("Connected to room successfully")
        
    except Exception as e:
        logger.error(f"Error during Space Mystery session: {e}")
        raise

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
import logging
import os
import json
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
from livekit.agents.llm import ChatMessage
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("jar-sdr-agent")
load_dotenv(".env.local")

# Create necessary directories
os.makedirs("company", exist_ok=True)
os.makedirs("user-database", exist_ok=True)

# Load Jar company information from file
def load_jar_company_info():
    company_file = "company/jar_info.json"
    
    # If file doesn't exist, create it with default data
    if not os.path.exists(company_file):
        jar_data = {
            "company": "Jar",
            "description": "Jar is India's fastest growing micro-savings app that helps users save small amounts daily and invest in digital gold. We make saving as easy and habitual as your daily coffee.",
            "products": {
                "Daily Savings": "Save small amounts daily starting from just ₹1",
                "Digital Gold": "Convert your savings into 24K digital gold",
                "Jar App": "Mobile-first platform with intuitive saving features",
                "Auto-Save": "Set up automatic daily savings from your UPI"
            },
            "pricing": {
                "Account Opening": "Completely free",
                "Savings": "No minimum balance required",
                "Digital Gold Purchase": "Zero commission on gold buying",
                "Gold Storage": "Free secure storage",
                "Withdrawal": "Small processing fee for gold redemption"
            },
            "faq": [
                {
                    "question": "What is Jar?",
                    "answer": "Jar is a micro-savings app that helps you save small amounts daily and convert them into digital gold, making saving a daily habit."
                },
                {
                    "question": "How does Jar work?",
                    "answer": "You can save any amount starting from ₹1 daily. Your savings are automatically converted into digital gold, which you can accumulate or redeem anytime."
                },
                {
                    "question": "Is there any minimum amount to start?",
                    "answer": "No, you can start saving with as little as ₹1. There's no minimum balance requirement."
                },
                {
                    "question": "How much does Jar cost?",
                    "answer": "Opening an account is completely free. There are no charges for saving or storing gold. A small processing fee applies only when you redeem physical gold."
                },
                {
                    "question": "Is my money safe with Jar?",
                    "answer": "Yes, your digital gold is stored securely with regulated partners and is fully insured. Jar is compliant with all RBI regulations."
                },
                {
                    "question": "Can I withdraw my money anytime?",
                    "answer": "Yes, you can redeem your digital gold for cash or physical gold delivery anytime through the app."
                },
                {
                    "question": "What is digital gold?",
                    "answer": "Digital gold represents physical 24K gold that you own. Each unit in your Jar account corresponds to actual gold stored in secure vaults."
                },
                {
                    "question": "Do you have a mobile app?",
                    "answer": "Yes, Jar is available as a mobile app on both iOS and Android platforms."
                }
            ]
        }
        
        # Save company info to file
        with open(company_file, 'w') as f:
            json.dump(jar_data, f, indent=2)
        logger.info(f"Created company file: {company_file}")
    
    # Load company info from file
    try:
        with open(company_file, 'r') as f:
            company_data = json.load(f)
        logger.info(f"Loaded company info from: {company_file}")
        return company_data
    except Exception as e:
        logger.error(f"Error loading company info: {e}")
        return None

def save_lead_info(lead_data):
    """Save lead information to user-database folder"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user-database/lead_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(lead_data, f, indent=2)
        
        logger.info(f"Lead saved to: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving lead: {e}")
        return None

class JarSDRAgent(Agent):
    def __init__(self):
        # Load company information from file
        self.company_info = load_jar_company_info()
        if not self.company_info:
            raise Exception("Failed to load company information")
            
        self.lead_data = {
            "name": "",
            "email": "",
            "phone": "",
            "company": "",
            "role": "",
            "saving_habits": "",
            "monthly_capacity": "",
            "saving_goal": "",
            "timeline": "",
            "conversation_summary": "",
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_state = "greeting"
        self.lead_complete = False
        
        # SDR instructions - include initial greeting in instructions
        instructions = """You are Priya, a friendly and enthusiastic Sales Development Representative for Jar, India's leading micro-savings app. 

IMPORTANT: You MUST start the conversation with this exact greeting:
"Hello! I'm Priya, your Jar savings consultant. Welcome! I'm here to help you start your micro-saving journey. What brings you here today?"

After the greeting, follow this conversation flow:
1. Understand their saving needs and goals
2. Answer questions about Jar using ONLY the FAQ information provided
3. Naturally collect lead information during the conversation
4. End with a warm summary when they indicate they're done

LEAD INFORMATION TO COLLECT (ask naturally during conversation):
- Name
- Email address  
- Current saving habits (none/irregular/regular)
- Monthly saving capacity
- Primary saving goal (emergency fund/travel/gold investment/other)
- Timeline to start

RULES:
- Always be warm, encouraging, and patient
- Only answer questions using the provided FAQ - never make up information
- If you don't know something, be honest and offer to connect them with specialists
- Keep responses conversational and friendly
- End calls gracefully when user says goodbye or indicates they're done

ABOUT JAR:
{company_description}

FAQ FOR ANSWERS:
{faq_data}
"""
        
        # Format instructions
        faq_text = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in self.company_info['faq']])
        formatted_instructions = instructions.format(
            company_description=self.company_info['description'],
            faq_data=faq_text
        )
        
        super().__init__(instructions=formatted_instructions)

    @function_tool
    async def update_lead_info(self, context: RunContext, field: str, value: str) -> str:
        """Update lead information with user-provided data"""
        valid_fields = ["name", "email", "phone", "company", "role", "saving_habits", "monthly_capacity", "saving_goal", "timeline"]
        
        if field not in valid_fields:
            return f"Invalid field. Please use one of: {', '.join(valid_fields)}"
        
        self.lead_data[field] = value
        logger.info(f"Updated lead field '{field}': {value}")
        
        return f"Thank you, I've noted that down."

    @function_tool
    async def search_faq(self, context: RunContext, question: str) -> str:
        """Search FAQ for relevant answers to user questions"""
        question_lower = question.lower()
        
        # Simple keyword matching
        for faq_item in self.company_info['faq']:
            faq_question_lower = faq_item['question'].lower()
            # Check for direct matches
            if any(keyword in question_lower for keyword in faq_question_lower.split()[:3]):
                return faq_item['answer']
        
        # Check common question patterns
        if "what is jar" in question_lower:
            return self.company_info['faq'][0]['answer']
        elif "how does" in question_lower and "work" in question_lower:
            return self.company_info['faq'][1]['answer']
        elif "minimum" in question_lower or "start" in question_lower:
            return self.company_info['faq'][2]['answer']
        elif "cost" in question_lower or "price" in question_lower or "free" in question_lower:
            return self.company_info['faq'][3]['answer']
        elif "safe" in question_lower or "secure" in question_lower:
            return self.company_info['faq'][4]['answer']
        elif "withdraw" in question_lower or "redeem" in question_lower:
            return self.company_info['faq'][5]['answer']
        elif "digital gold" in question_lower:
            return self.company_info['faq'][6]['answer']
        elif "app" in question_lower or "mobile" in question_lower:
            return self.company_info['faq'][7]['answer']
        
        return "That's a great question! I'd be happy to connect you with our specialist team who can provide more detailed information about that."

    @function_tool
    async def end_conversation(self, context: RunContext) -> str:
        """End the conversation and save lead information"""
        self.lead_complete = True
        
        # Create summary
        summary = f"Conversation about {self.lead_data['saving_goal'] or 'saving goals'}. Current habits: {self.lead_data['saving_habits'] or 'not specified'}. Timeline: {self.lead_data['timeline'] or 'not specified'}."
        self.lead_data["conversation_summary"] = summary
        
        # Save lead to database
        filename = save_lead_info(self.lead_data)
        
        return f"""Thank you for your time! Here's a quick summary:

{summary}

I'll make sure you receive all the information about starting your saving journey with Jar. Have a wonderful day!"""

def prewarm(proc: JobProcess):
    """Preload models and company data"""
    logger.info("Prewarming agent...")
    proc.userdata["vad"] = silero.VAD.load()
    # Preload company data
    company_info = load_jar_company_info()
    if company_info:
        logger.info("Company data loaded successfully during prewarm")
    else:
        logger.error("Failed to load company data during prewarm")

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "agent": "jar-sdr"
    }
    
    logger.info("Starting Jar SDR agent session...")
    
    try:
        # Initialize Jar SDR agent
        jar_agent = JarSDRAgent()
        logger.info("Jar SDR agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return

    # Set up voice AI pipeline with FEMALE voice
    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=google.LLM(
            model="gemini-2.0-flash",
        ),
        tts=murf.TTS(
            voice="en-US-alicia",  # Female voice
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
        logger.info(f"User said: {transcript}")

    @session.on("agent_speech") 
    def on_agent_speech(transcript: str):
        logger.info(f"Agent responding: {transcript}")

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
            agent=jar_agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        logger.info("Agent session started successfully")
        
        # Join the room and connect to the user
        await ctx.connect()
        logger.info("Connected to room successfully")
        
    except Exception as e:
        logger.error(f"Error during session: {e}")
        raise

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
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
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("fraud-agent")
load_dotenv(".env.local")

# Create necessary directories
os.makedirs("fraud_database", exist_ok=True)

def load_fraud_cases():
    """Load fraud cases from database"""
    database_file = "fraud_database/fraud_cases.json"
    
    # Create sample database if it doesn't exist
    if not os.path.exists(database_file):
        sample_data = {
            "fraud_cases": [
                {
                    "userName": "Rahul Sharma",
                    "securityIdentifier": "12345",
                    "cardEnding": "4242",
                    "case": "pending_review",
                    "transactionName": "International Electronics",
                    "transactionTime": "2024-01-15 14:30:00",
                    "transactionCategory": "e-commerce",
                    "transactionSource": "aliexpress.com",
                    "amount": "₹18,245",
                    "location": "Shenzhen, China",
                    "securityQuestion": "What is your mother's maiden name?",
                    "securityAnswer": "patel",
                    "outcome": "",
                    "callTimestamp": ""
                },
                {
                    "userName": "Priya Singh",
                    "securityIdentifier": "67890",
                    "cardEnding": "5678",
                    "case": "pending_review",
                    "transactionName": "Dubai Luxury Mall",
                    "transactionTime": "2024-01-15 16:45:00",
                    "transactionCategory": "retail",
                    "transactionSource": "dubailuxury.ae",
                    "amount": "₹92,500",
                    "location": "Dubai, UAE",
                    "securityQuestion": "What was the name of your first school?",
                    "securityAnswer": "kendriya",
                    "outcome": "",
                    "callTimestamp": ""
                },
                {
                    "userName": "Arjun Kumar",
                    "securityIdentifier": "54321",
                    "cardEnding": "9876",
                    "case": "pending_review",
                    "transactionName": "Premium Tech Store",
                    "transactionTime": "2024-01-15 18:20:00",
                    "transactionCategory": "electronics",
                    "transactionSource": "premiumtech.com",
                    "amount": "₹67,999",
                    "location": "Singapore",
                    "securityQuestion": "What is your birth city?",
                    "securityAnswer": "delhi",
                    "outcome": "",
                    "callTimestamp": ""
                },
                {
                    "userName": "Ananya Reddy",
                    "securityIdentifier": "11223",
                    "cardEnding": "3344",
                    "case": "pending_review",
                    "transactionName": "Online Gaming Purchase",
                    "transactionTime": "2024-01-15 20:15:00",
                    "transactionCategory": "entertainment",
                    "transactionSource": "gameworld.com",
                    "amount": "₹12,750",
                    "location": "United States",
                    "securityQuestion": "What is your favorite food?",
                    "securityAnswer": "biryani",
                    "outcome": "",
                    "callTimestamp": ""
                },
                {
                    "userName": "Vikram Mehta",
                    "securityIdentifier": "44556",
                    "cardEnding": "7788",
                    "case": "pending_review",
                    "transactionName": "Flight Booking",
                    "transactionTime": "2024-01-15 22:30:00",
                    "transactionCategory": "travel",
                    "transactionSource": "quickflights.com",
                    "amount": "₹45,320",
                    "location": "London, UK",
                    "securityQuestion": "What is your father's middle name?",
                    "securityAnswer": "kumar",
                    "outcome": "",
                    "callTimestamp": ""
                }
            ]
        }
        
        with open(database_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        logger.info("Created sample fraud database for State Bank of India")
    
    try:
        with open(database_file, 'r') as f:
            data = json.load(f)
        logger.info("Loaded fraud cases from database")
        return data
    except Exception as e:
        logger.error(f"Error loading fraud database: {e}")
        return {"fraud_cases": []}

def update_fraud_case(user_name, updates):
    """Update a specific fraud case in the database"""
    try:
        database_file = "fraud_database/fraud_cases.json"
        with open(database_file, 'r') as f:
            data = json.load(f)
        
        # Find and update the case
        for case in data["fraud_cases"]:
            if case["userName"].lower() == user_name.lower():
                case.update(updates)
                case["callTimestamp"] = datetime.now().isoformat()
                break
        
        # Save updated data
        with open(database_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Updated fraud case for {user_name}: {updates}")
        return True
    except Exception as e:
        logger.error(f"Error updating fraud case: {e}")
        return False

class FraudAlertAgent(Agent):
    def __init__(self):
        self.fraud_cases = load_fraud_cases()
        self.current_case = None
        self.verification_passed = False
        self.conversation_state = "greeting"
        
        instructions = """You are a professional fraud detection agent for State Bank of India. You must follow this exact flow:

1. GREETING: Start with: "Namaste! This is State Bank of India Fraud Prevention Department calling regarding a suspicious transaction on your account. To verify your identity, could you please tell me your full name?"

2. IDENTITY VERIFICATION: 
   - When user provides name, search for their fraud case
   - If found, ask the security question from their record
   - If correct answer, proceed to transaction review
   - If incorrect, end call politely

3. TRANSACTION REVIEW:
   - Clearly describe the suspicious transaction using case data
   - Ask: "Did you authorize this transaction of [amount] at [transactionName] on [transactionTime]?"

4. RESOLUTION:
   - If YES: Mark as safe, assure customer, end call
   - If NO: Mark as fraud, explain protection steps, end call

IMPORTANT RULES:
- Always use calm, professional, reassuring language appropriate for Indian customers
- Use Indian English with occasional Hindi words like "Namaste", "Dhanyavaad"
- Never ask for full card numbers, PINs, or passwords
- Speak clearly and patiently
- End calls politely regardless of outcome
- Only use data from the provided fraud cases
- For State Bank of India, use customer service number: 1800-1234

FRAUD CASES DATA:
{fraud_cases_data}
"""
        
        # Format fraud cases for instructions
        cases_text = ""
        for case in self.fraud_cases["fraud_cases"]:
            cases_text += f"User: {case['userName']}, Security ID: {case['securityIdentifier']}, Card: ****{case['cardEnding']}, Question: {case['securityQuestion']}, Answer: {case['securityAnswer']}\n"
        
        formatted_instructions = instructions.format(fraud_cases_data=cases_text)
        super().__init__(instructions=formatted_instructions)

    @function_tool
    async def find_fraud_case(self, context: RunContext, user_name: str) -> str:
        """Find fraud case by user name"""
        for case in self.fraud_cases["fraud_cases"]:
            if case["userName"].lower() == user_name.lower():
                self.current_case = case
                self.conversation_state = "verification"
                return f"Found case for {user_name}. Security question: {case['securityQuestion']}"
        
        return f"No pending fraud cases found for {user_name}. Please contact State Bank of India customer service at 1800-1234 for assistance."

    @function_tool
    async def verify_security_answer(self, context: RunContext, user_answer: str) -> str:
        """Verify user's security answer"""
        if not self.current_case:
            return "No case loaded. Please provide your name first."
        
        expected_answer = self.current_case["securityAnswer"].lower()
        user_answer_clean = user_answer.lower().strip()
        
        if user_answer_clean == expected_answer:
            self.verification_passed = True
            self.conversation_state = "transaction_review"
            return "Verification successful. Dhanyavaad. Now let me tell you about the suspicious transaction we detected on your State Bank of India account."
        else:
            self.conversation_state = "verification_failed"
            return "I'm sorry, but we cannot verify your identity at this time. Please contact State Bank of India customer service directly at 1800-1234 for assistance. Dhanyavaad."

    @function_tool
    async def describe_transaction(self, context: RunContext) -> str:
        """Describe the suspicious transaction to the user"""
        if not self.current_case or not self.verification_passed:
            return "Please complete verification first."
        
        case = self.current_case
        transaction_details = f"""
We detected a suspicious transaction on your State Bank of India card ending with {case['cardEnding']}.

Amount: {case['amount']}
Merchant: {case['transactionName']} ({case['transactionSource']})
Date/Time: {case['transactionTime']}
Location: {case['location']}
Category: {case['transactionCategory']}

Did you authorize this transaction?
"""
        return transaction_details

    @function_tool
    async def handle_transaction_response(self, context: RunContext, user_response: str) -> str:
        """Handle user's response about the transaction"""
        if not self.current_case:
            return "No case loaded."
        
        user_response_lower = user_response.lower()
        case = self.current_case
        
        if "yes" in user_response_lower or "authorized" in user_response_lower or "haan" in user_response_lower:
            # Mark as safe
            updates = {
                "case": "confirmed_safe",
                "outcome": "Customer confirmed transaction as legitimate"
            }
            update_fraud_case(case["userName"], updates)
            
            return "Dhanyavaad for confirming. We've noted this transaction as authorized. Your State Bank of India card remains active. Thank you for helping us keep your account secure."
        
        elif "no" in user_response_lower or "not" in user_response_lower or "fraud" in user_response_lower or "nahi" in user_response_lower:
            # Mark as fraudulent
            updates = {
                "case": "confirmed_fraud",
                "outcome": "Customer denied transaction - marked as fraudulent"
            }
            update_fraud_case(case["userName"], updates)
            
            return f"Dhanyavaad for confirming this was fraudulent. We are immediately blocking your State Bank of India card to prevent further unauthorized transactions. A new card will be dispatched to your registered address within 3-5 business days. We have initiated a dispute for the fraudulent charge of {case['amount']}. Please check your email and SMS for further instructions. Thank you for your cooperation."
        
        else:
            return "I apologize, I didn't understand your response. Could you please confirm if you authorized this transaction? Please answer yes or no."

    @function_tool
    async def end_call_verification_failed(self, context: RunContext) -> str:
        """End call when verification fails"""
        if self.current_case:
            updates = {
                "case": "verification_failed",
                "outcome": "Security verification failed during call"
            }
            update_fraud_case(self.current_case["userName"], updates)
        
        return "For security reasons, we are ending this call. Please contact State Bank of India customer service directly at 1800-1234 for assistance. Dhanyavaad."

def prewarm(proc: JobProcess):
    """Preload models and fraud database"""
    logger.info("Prewarming State Bank of India fraud agent...")
    proc.userdata["vad"] = silero.VAD.load()
    # Preload fraud database
    fraud_cases = load_fraud_cases()
    if fraud_cases:
        logger.info(f"Loaded {len(fraud_cases['fraud_cases'])} fraud cases during prewarm")
    else:
        logger.error("Failed to load fraud cases during prewarm")

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "agent": "sbi-fraud-alert"
    }
    
    logger.info("Starting State Bank of India Fraud Alert agent session...")
    
    try:
        # Initialize Fraud Alert agent
        fraud_agent = FraudAlertAgent()
        logger.info("State Bank of India Fraud Alert agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return

    # Set up voice AI pipeline with valid Murf voice
    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=google.LLM(
            model="gemini-2.0-flash",
        ),
        tts=murf.TTS(
            voice="en-US-matthew",  # Using valid Murf voice
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
            agent=fraud_agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        logger.info("State Bank of India fraud agent session started successfully")
        
        # Join the room and connect to the user
        await ctx.connect()
        logger.info("Connected to room successfully")
        
    except Exception as e:
        logger.error(f"Error during fraud agent session: {e}")
        raise

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
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

logger = logging.getLogger("food-ordering-agent")
load_dotenv(".env.local")

# Create necessary directories
os.makedirs("orders", exist_ok=True)

def load_catalog():
    """Load food catalog from existing catalog.json file"""
    catalog_file = "catalog.json"
    
    try:
        with open(catalog_file, 'r') as f:
            data = json.load(f)
        logger.info("Loaded food catalog successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading catalog: {e}")
        # Return empty catalog if file doesn't exist
        return {"categories": [], "recipes": {}}

def save_order(order_data):
    """Save order to JSON file in orders folder"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"orders/order_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(order_data, f, indent=2)
        
        logger.info(f"Order saved to: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving order: {e}")
        return None

def get_cart_total(cart):
    """Calculate total amount from cart"""
    return sum(item["price"] * item["quantity"] for item in cart)

class FoodOrderingAgent(Agent):
    def __init__(self):
        self.catalog = load_catalog()
        self.cart = []
        self.conversation_state = "greeting"
        
        instructions = """You are MIMI, a friendly and enthusiastic food ordering assistant for foodieWay. You help customers order groceries and food items with a warm, personalized touch.

PERSONALITY:
- Warm, friendly, and enthusiastic like a helpful store assistant
- Use Indian English with occasional Hindi words like "Achha", "Theek hai", "Shukriya"
- Always address customers respectfully
- Show excitement when helping with recipes or finding items

CONVERSATION FLOW:
1. Start with energetic greeting: "Namaste! Welcome to foodieWay! I'm MIMI, your personal shopping assistant. I'm so excited to help you with your grocery shopping today. What would you like to start with?"

2. ACTIVE LISTENING & SUGGESTIONS:
   - Listen carefully to what customer wants
   - Suggest related items: "Would you like some cheese with that bread?"
   - Offer alternatives if items are unavailable
   - Help with meal planning: "That sounds delicious! Do you need anything else for your meal?"

3. RECIPE ASSISTANCE:
   - For recipe requests, enthusiastically explain what you're adding
   - "Wonderful choice! For a perfect sandwich, I'll add fresh bread, eggs, and tomatoes"
   - Suggest additional items that might complement the recipe

4. CART MANAGEMENT:
   - Always confirm additions with price and quantity
   - Read back cart updates clearly
   - Show excitement when cart has good variety

5. ORDER COMPLETION:
   - Celebrate successful orders: "Yay! Your order is ready!"
   - Provide clear order summary with total
   - Thank warmly and invite back

SPECIAL FEATURES:
- Remember customer preferences during conversation
- Suggest popular combinations
- Help stay within budget if mentioned
- Offer seasonal suggestions

IMPORTANT:
- Always be positive and encouraging
- Use conversational, natural language
- Make the shopping experience joyful
- Confirm important details
- Prices are in Indian Rupees (₹)
"""

        super().__init__(instructions=instructions)

    def find_item(self, item_name):
        """Find item in catalog by name with fuzzy matching"""
        item_name_lower = item_name.lower()
        
        # Exact match first
        for category in self.catalog["categories"]:
            for item in category["items"]:
                if item_name_lower == item["name"].lower():
                    return item
        
        # Partial match
        for category in self.catalog["categories"]:
            for item in category["items"]:
                if (item_name_lower in item["name"].lower() or 
                    any(item_name_lower in tag for tag in item.get("tags", []))):
                    return item
        return None

    def get_recipe_items(self, recipe_name):
        """Get items for a recipe with enthusiastic descriptions"""
        recipe_name_lower = recipe_name.lower()
        recipe_descriptions = {
            "sandwich": "a delicious sandwich",
            "pasta": "a tasty pasta meal", 
            "salad": "a fresh salad",
            "breakfast": "a complete breakfast"
        }
        
        for recipe, items in self.catalog["recipes"].items():
            if recipe_name_lower in recipe:
                return items, recipe_descriptions.get(recipe, "this recipe")
        return [], ""

    @function_tool
    async def add_item_to_cart(self, context: RunContext, item_name: str, quantity: int = 1) -> str:
        """Add an item to the shopping cart with enthusiastic confirmation"""
        item = self.find_item(item_name)
        if not item:
            return f"Oh dear! I couldn't find '{item_name}' in our store. Maybe try a different name? Or I can help you search for similar items!"
        
        # Check if item already in cart
        for cart_item in self.cart:
            if cart_item["id"] == item["id"]:
                cart_item["quantity"] += quantity
                total_price = cart_item["quantity"] * item["price"]
                return f"Achha! Updated your {item['name']} to {cart_item['quantity']} {item['unit']}(s). Perfect! Total for this item: ₹{total_price}"

        # Add new item to cart
        cart_item = {
            "id": item["id"],
            "name": item["name"],
            "price": item["price"],
            "quantity": quantity,
            "unit": item["unit"],
            "brand": item.get("brand", ""),
            "total": item["price"] * quantity
        }
        self.cart.append(cart_item)
        
        # Enthusiastic confirmation with suggestions
        suggestions = {
            "bread": "Would you like some butter or jam with your bread?",
            "eggs": "How about some vegetables to make an omelette?",
            "milk": "Some cookies or cereals would go great with milk!",
            "rice": "Would you like some dal or vegetables to go with your rice?"
        }
        
        suggestion = suggestions.get(item['name'].lower(), "")
        return f"Wonderful! Added {quantity} {item['unit']} of {item['name']} to your cart. ₹{item['price']} each. {suggestion}"

    @function_tool
    async def add_recipe_to_cart(self, context: RunContext, recipe_name: str) -> str:
        """Add all ingredients for a recipe to the cart with excited explanation"""
        recipe_items, recipe_desc = self.get_recipe_items(recipe_name)
        if not recipe_items:
            return f"Oh! I don't have a specific recipe for '{recipe_name}' yet. But I can help you add items individually! Available recipes: sandwich, pasta, salad, breakfast."
        
        added_items = []
        for item_name in recipe_items:
            item = self.find_item(item_name)
            if item:
                # Check if already in cart
                existing_item = next((ci for ci in self.cart if ci["id"] == item["id"]), None)
                if existing_item:
                    existing_item["quantity"] += 1
                    existing_item["total"] = existing_item["price"] * existing_item["quantity"]
                else:
                    cart_item = {
                        "id": item["id"],
                        "name": item["name"],
                        "price": item["price"],
                        "quantity": 1,
                        "unit": item["unit"],
                        "brand": item.get("brand", ""),
                        "total": item["price"]
                    }
                    self.cart.append(cart_item)
                added_items.append(item["name"])
        
        if added_items:
            return f"Yay! I've added everything you need for {recipe_desc}: {', '.join(added_items)}. Your cart is looking great with {len(self.cart)} items now! 🎉"
        else:
            return "Hmm, I couldn't find the ingredients for that recipe. Let me help you add them one by one!"

    @function_tool
    async def view_cart(self, context: RunContext) -> str:
        """Show current cart contents with enthusiastic summary"""
        if not self.cart:
            return "Your cart is looking a bit empty! What delicious items would you like to add today? I'm here to help! 😊"
        
        cart_summary = "Let me show you your amazing cart! 🛒\n\n"
        total_amount = get_cart_total(self.cart)
        
        for i, item in enumerate(self.cart, 1):
            item_total = item["price"] * item["quantity"]
            cart_summary += f"{i}. {item['quantity']} {item['unit']} {item['name']} - ₹{item_total}\n"
        
        cart_summary += f"\n🎊 Total amount: ₹{total_amount}\n"
        
        # Add encouraging message based on cart size
        if len(self.cart) >= 5:
            cart_summary += "Wow! You've got a wonderful selection there! 🥳"
        elif len(self.cart) >= 3:
            cart_summary += "Great choices! Your cart is looking good! 👍"
        else:
            cart_summary += "Nice start! What else would you like to add? 😊"
            
        return cart_summary

    @function_tool
    async def remove_item_from_cart(self, context: RunContext, item_name: str) -> str:
        """Remove an item from the cart with friendly confirmation"""
        item_name_lower = item_name.lower()
        for i, cart_item in enumerate(self.cart):
            if item_name_lower in cart_item["name"].lower():
                removed_item = self.cart.pop(i)
                remaining_items = len(self.cart)
                
                if remaining_items > 0:
                    return f"No problem! I've removed {removed_item['name']} from your cart. You still have {remaining_items} wonderful items left! 😊"
                else:
                    return f"Removed {removed_item['name']}. Your cart is empty now. What would you like to add? I have so many delicious options!"
        
        return f"I looked everywhere but couldn't find '{item_name}' in your cart. Want to try again or see what's in your cart?"

    @function_tool
    async def update_item_quantity(self, context: RunContext, item_name: str, new_quantity: int) -> str:
        """Update quantity of an item in the cart with positive confirmation"""
        item_name_lower = item_name.lower()
        for cart_item in self.cart:
            if item_name_lower in cart_item["name"].lower():
                if new_quantity <= 0:
                    return await self.remove_item_from_cart(context, item_name)
                
                old_quantity = cart_item["quantity"]
                cart_item["quantity"] = new_quantity
                cart_item["total"] = cart_item["price"] * new_quantity
                
                if new_quantity > old_quantity:
                    return f"Excellent! Updated {cart_item['name']} from {old_quantity} to {new_quantity}. Smart shopping! 🛍️"
                else:
                    return f"Sure thing! Updated {cart_item['name']} quantity to {new_quantity}. Perfect for your needs! 👍"
        
        return f"I couldn't find '{item_name}' in your cart. Would you like to add it?"

    @function_tool
    async def search_items(self, context: RunContext, query: str) -> str:
        """Search for items in the catalog with helpful suggestions"""
        query_lower = query.lower()
        found_items = []
        
        for category in self.catalog["categories"]:
            for item in category["items"]:
                if (query_lower in item["name"].lower() or 
                    query_lower in category["name"].lower() or
                    any(query_lower in tag for tag in item.get("tags", []))):
                    found_items.append({
                        "name": item["name"],
                        "price": item["price"],
                        "unit": item["unit"],
                        "category": category["name"]
                    })
        
        if found_items:
            response = f"I found these wonderful items matching '{query}':\n\n"
            for item in found_items[:6]:  # Limit to 6 results
                response += f"• {item['name']} - ₹{item['price']} per {item['unit']} ({item['category']})\n"
            
            if len(found_items) > 6:
                response += f"\n...and {len(found_items) - 6} more! Would you like me to be more specific?"
            else:
                response += "\nWhich of these would you like to add to your cart? 😊"
                
            return response
        else:
            return f"I searched high and low but couldn't find '{query}'. Try searching by category like 'groceries', 'fruits', or 'snacks'. Or I can show you all categories!"

    @function_tool
    async def show_categories(self, context: RunContext) -> str:
        """Show all available categories with enticing descriptions"""
        if not self.catalog["categories"]:
            return "Our store is getting ready! Categories will be available soon. 🛒"
        
        response = "Here are all our wonderful categories:\n\n"
        category_descriptions = {
            "Groceries": "Daily essentials like bread, milk, eggs and more! 🥚🥛",
            "Fruits & Vegetables": "Fresh and crunchy fruits & veggies! 🍎🥦", 
            "Snacks & Beverages": "Yummy snacks and refreshing drinks! 🍫🥤",
            "Prepared Food": "Ready-to-eat delicious meals! 🍕🍛"
        }
        
        for category in self.catalog["categories"]:
            desc = category_descriptions.get(category["name"], "Amazing products!")
            item_count = len(category["items"])
            response += f"• {category['name']} - {desc} ({item_count} items)\n"
        
        response += "\nWhich category interests you? I can show you items from any category! 😊"
        return response

    @function_tool
    async def place_order(self, context: RunContext, customer_name: str = "Valued Customer") -> str:
        """Place the final order with celebration and save to file"""
        if not self.cart:
            return "Your cart is empty! Let's fill it with some delicious items first. What would you like to add? 🛒"

        # Calculate total
        total_amount = get_cart_total(self.cart)
        item_count = sum(item["quantity"] for item in self.cart)
        
        # Create comprehensive order object
        order_data = {
            "order_id": f"QB{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "customer_name": customer_name,
            "timestamp": datetime.now().isoformat(),
            "items": self.cart.copy(),
            "item_count": item_count,
            "total_amount": total_amount,
            "status": "confirmed",
            "delivery_estimate": "30-45 minutes",
            "store": "foodieWay "
        }
        
        # Save order to file in orders folder
        filename = save_order(order_data)
        
        if filename:
            # Celebration message based on order size
            if item_count >= 8:
                celebration = "WOW! What a fantastic order! 🎉"
            elif item_count >= 5:
                celebration = "Excellent choices! Your order looks amazing! 🌟"
            else:
                celebration = "Lovely selection! Your order is perfect! 👍"
            
            # Clear cart after successful order
            order_summary = self.cart.copy()
            self.cart.clear()
            
            return f"""{celebration}

🎊 ORDER PLACED SUCCESSFULLY! 🎊

Order ID: {order_data['order_id']}
Items: {item_count} products
Total: ₹{total_amount}
Delivery: {order_data['delivery_estimate']}

Thank you for shopping with foodieWay! Your order has been saved and will be delivered soon. Shukriya! 💝"""
        else:
            return "Oh no! There was a small issue saving your order. Please try again in a moment. I'm really sorry for the inconvenience! 🙏"

    @function_tool
    async def clear_cart(self, context: RunContext) -> str:
        """Clear all items from the cart with understanding response"""
        if not self.cart:
            return "Your cart is already empty and ready for new adventures! What would you like to add? 😊"
        
        item_count = len(self.cart)
        self.cart.clear()
        return f"Cleared your cart of {item_count} items. No problem at all! Fresh start - what delicious items would you like to add now? 🛒"

def prewarm(proc: JobProcess):
    """Preload models and food catalog"""
    logger.info("Prewarming foodieWay food ordering agent...")
    proc.userdata["vad"] = silero.VAD.load()
    # Preload food catalog
    catalog = load_catalog()
    if catalog and catalog["categories"]:
        logger.info(f"Loaded catalog with {len(catalog['categories'])} categories during prewarm")
    else:
        logger.warning("Catalog is empty or couldn't be loaded during prewarm")

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "agent": "foodieWay-ordering"
    }
    
    logger.info("Starting foodieWay Food Ordering agent session...")
    
    try:
        # Initialize Food Ordering agent
        food_agent = FoodOrderingAgent()
        logger.info("QuickBasket Food Ordering agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return

    # Set up voice AI pipeline with friendly Indian-accented voice
    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=google.LLM(
            model="gemini-2.0-flash",
        ),
        tts=murf.TTS(
            voice="en-US-alicia",  # Friendly female voice
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
        logger.info(f"Customer said: {transcript}")

    @session.on("agent_speech") 
    def on_agent_speech(transcript: str):
        logger.info(f"MIMI (Assistant) responding: {transcript}")

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
            agent=food_agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        logger.info("foodieWay ordering agent session started successfully")
        
        # Join the room and connect to the user
        await ctx.connect()
        logger.info("Connected to room successfully")
        
    except Exception as e:
        logger.error(f"Error during foodieWay ordering session: {e}")
        raise

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
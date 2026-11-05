import time

import torch
import whisper
t0 = time.time()

import json
from unittest import result
from ollama import chat
from pydantic import BaseModel
from typing import Literal, List, Dict

#Text To Speech and Speech To Text imports
import numpy as np
import sounddevice as sd
from kokoro import KPipeline
import warnings
warnings.filterwarnings("ignore")

MODEL = 'qwen2.5:7b'

print(f"imports: {time.time() - t0:.1f}s")

#----------- Text to Speech and Speech to Text Setup ----------

SAMPLE_RATE = 16000
DURATION = 5  # seconds

stt = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")

pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M") #getting the pipeline object for the Kokoro model
print(f"model load: {time.time() - t0:.1f}s")

def warming_upspeak(text, voice="af_heart"):
    for _, _, audio in pipeline(text, voice=voice): 
        pass 

def speak(text, voice="af_heart"):
    for _, _, audio in pipeline(text, voice=voice):
        a = audio.numpy() if hasattr(audio, "numpy") else audio
        sd.play(a, 24000)
        sd.wait()




def listen():
    chunks = []
    def callback(indata, frames, time, status):
        chunks.append(indata.copy())

    input("[Press Enter to start speaking]")
    with sd.InputStream(samplerate=16000, channels=1, dtype='float32', callback=callback):
        input("[Recording... press Enter to stop]")

    if not chunks:
        return ""
    audio = np.concatenate(chunks).flatten().astype(np.float32)
    result = stt.transcribe(audio, language="en", fp16=True)   # fp16=False on CPU
    return result["text"].strip()

def warm_up_stt():
    silence = np.zeros(16000, dtype=np.float32)   # 1 second of silence at 16 kHz
    stt.transcribe(silence, language="en", fp16=True)

t1 = time.time()

warming_upspeak("Hello, this is a test. This is to initialise the model and load it into memory. The first sentence is always slower than subsequent sentences.") #only a warmup to ensure faster response later.
print(f"warming up speak: {time.time() - t1:.1f}s")
print("Model ready to use. Please Continue To Next Session.")

print(sd.query_devices())
print(sd.default.device)


#-----------MAIN PROGRAM----------------

menu = {
  "Classic Barn Burger": 7.99,
  "Double Barn Burger": 10.49,
  "Bacon Cheddar Burger": 9.49,
  "Crispy Chicken Sandwich": 8.99,
  "Veggie Burger": 8.49,
  "French Fries": 3.99,
  "Onion Rings": 4.99,
  "Chocolate Milkshake": 5.49,
  "Fountain Soda": 2.49,
  "Bottled Water": 1.99
}

ALIASES = {
    "Classic Barn Burger": ["classic", "barn burger", "regular burger", "burger"],
    "Double Barn Burger": ["double"],
    "Bacon Cheddar Burger": ["bacon", "cheddar"],
    "Crispy Chicken Sandwich": ["chicken"],
    "Veggie Burger": ["veggie", "vegetarian"],
    "French Fries": ["fries", "french"],
    "Onion Rings": ["onion ring", "rings"],
    "Chocolate Milkshake": ["shake", "chocolate"],
    "Fountain Soda": ["soda", "coke", "cola", "drink"],
    "Bottled Water": ["water"],
}

TAX_RATE = 0.10
menu_cents = {name: round(price * 100) for name, price in menu.items()}

order = {}

def warm_up():
    print("Warming up the model...")
    chat(MODEL, messages=[], keep_alive='30m')
    print("Ready!")
    warm_up_stt()
    print("Ready!")


class OrderItem(BaseModel):
    action: Literal["add", "remove", "set", "note"]
    name: Literal[
        "Classic Barn Burger", "Double Barn Burger", "Bacon Cheddar Burger",
        "Crispy Chicken Sandwich", "Veggie Burger", "French Fries",
        "Onion Rings", "Chocolate Milkshake", "Fountain Soda", "Bottled Water",
    ]
    quantity: int
    notes: str = ""

class ItemList(BaseModel):
    items: List[OrderItem]

alias_rules = "\n".join(f'- {", ".join(words)} -> {name}' for name, words in ALIASES.items())

EXTRACT_PROMPT = """You extract order changes from a customer's message at Burger Barn.
For each item, set action to one of:
- add: the customer wants more of an item ("add a fries", "another burger")
- remove: the customer cancels some of an item ("remove the fries")
- set: the customer states the total they want ("just one", "make it two", "only one burger")
- note: a special request for an item that is already ordered ("no onions on one burger", "extra sauce"). A note never changes the quantity.
Ingredients like onions, pickles, or cheese are NOT menu items. "No onions" is a note, never a remove.
Never add or remove an item because of a note.

Example:
Customer: "2 double barn burgers, one without onions"
Output: {"items": [{"action": "add", "name": "Double Barn Burger", "quantity": 2, "notes": "No Onions on one burger"}]}, 

For a note, write the complete current request for that item based on the customer's latest message,
and replace any earlier note about the same thing. If the customer says "all", write "all".
Example: earlier note "no cheese on one", customer says "make it all of them" -> note "no cheese on all".

If the item is not in the current order, "just one" means add 1.
A swap like "fries instead of onion rings" is one remove (the fries) and one add (the onion rings).
Only use items from the menu. Use the exact menu names, and convert number words to digits.
Put special requests (like "no pickles") in notes.
Respond with JSON in the form {"items": [{"action": "add", "name": "...", "quantity": 1, "notes": ""}]}.
Menu: """ + ", ".join(menu.keys()) + "\n" + alias_rules

# ---------- Bot 1: router ----------

class Intent(BaseModel):
    intent: Literal["question", "change_order", "confirm_order","other"]

ROUTER_PROMPT = """You classify what a customer wants at a burger restaurant.
Pick exactly one intent:
- greeting: says hello or a similar greeting ("hi", "hello", "good evening")
- question: asks about the menu, prices, ingredients, hours, or location
- change_order: wants to order, add, remove, cancel, or swap food or drinks
- confirm_order: says they are done ordering and want to finalize ("that's all", "that's it", "I'm done")
- other: anything unrelated to the restaurant

If the message names any food or drink items, choose change_order, even if it includes a greeting or "please".
confirm_order is only for finishing, with no new items ("that's all", "I'd like to pay").

Respond with JSON in the form {"intent": "<one of the intents above>"}."""


def route(user_text):
    print("Routing user text:", user_text)
    response = chat(
        MODEL,
        messages=[
            {'role': 'system', 'content': ROUTER_PROMPT},
            {'role': 'user', 'content': user_text},
        ],
        format=Intent.model_json_schema(),
        options={'temperature': 0},
    )
    return Intent.model_validate_json(response['message']['content']).intent
    print("Routing Completed...")

# ---------- Bot 2a: question answerer ----------

QUESTION_PROMPT = f"""You are the phone assistant for Burger Barn, a burger restaurant at
5312 Leary Ave NW, Ballard, Seattle, WA 98107.
Answer the customer's question in 1-2 short sentences. Your reply will be read aloud,
so do not use lists, markdown, or emojis. 
Only mention items and prices from the menu below. If you don't know something,
say a staff member can help.

MENU (item: price in USD):
{json.dumps(menu, indent=2)}

Current Order: {json.dumps(order, indent=2)}

Vegan options: The Veggie Burger is vegan. French Fries, Onion Rings, and Bottled Water are also vegan.

Hours: 8 AM to 10 PM, Monday to Friday. 10 AM to 8 PM, Saturday and Sunday.

No delivery but takeout and uber eats is available.

Only use facts written in this prompt. If the answer isn't here (allergens, etc.), say: "I'm not sure about that, a staff member can help you."

"""

def answer_question(user_text):
    response = chat(
        MODEL,
        messages=[
            {'role': 'system', 'content': QUESTION_PROMPT},
            {'role': 'user', 'content': user_text},
        ],
        options={'temperature': 0.3, 'num_predict': 100},
    )
    return response['message']['content']

def extract_items(user_text, current_order):
    system = EXTRACT_PROMPT + f"\nCurrent order: {json.dumps(current_order)}"
    response = chat(
        MODEL,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user_text},
        ],
        format=ItemList.model_json_schema(),
        options={'temperature': 0},
    )
    return ItemList.model_validate_json(response['message']['content']).items

def mentioned(name, text):          # <- here
    keys = ALIASES[name] + [name.lower()]
    return any(k in text for k in keys)

def apply_changes(items):
    for item in items:
        entry = order.setdefault(item.name, {"qty": 0, "notes": []})

        if item.action == "add":
            entry["qty"] += item.quantity
        elif item.action == "remove":
            entry["qty"] -= item.quantity
        elif item.action == "set":
            entry["qty"] = item.quantity
        if item.action == "note":
            if item.name in order:
                order[item.name]["notes"].append(item.notes)
            continue

        if item.notes and item.action != "remove":
            entry["notes"].append(item.notes)

        if entry["qty"] <= 0:
            del order[item.name]

def confirmation_text(items):
    parts = []
    for item in items:
        if item.action == "note":
            parts.append(f"noting {item.notes} for the {item.name}")
            continue

        verb = {"add": "adding", "remove": "removing", "set": "changing to"}[item.action]
        text = f"{verb} {item.quantity} {item.name}"
        if item.notes:
            text += f" ({item.notes})"
        parts.append(text)

    listed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"Got it, {listed}. Anything else?"

def fmt(cents):
    return f"${cents / 100:.2f}"

def process_order():
    if not order:
        return "Your order is empty. What would you like to order?"

    subtotal = sum(menu_cents[name] * e["qty"] for name, e in order.items())
    tax = round(subtotal * TAX_RATE)
    total = subtotal + tax

    lines = []
    for name, e in order.items():
        line = f"{name} x{e['qty']}: {fmt(menu_cents[name] * e['qty'])}"
        if e["notes"]:
            line += f"\n    Note: {'; '.join(e['notes'])}"
        lines.append(line)

    return ("Receipt:\n" + "\n".join(lines) + "\n"
            f"Subtotal: {fmt(subtotal)}\n"
            f"Tax: {fmt(tax)}\n"
            f"Total: {fmt(total)}\n"
            "Thank you!")

def handle_turn(user_text):
    print("Routing")
    intent = route(user_text)
    if intent in ("confirm_order", "greeting", "other") and any(mentioned(n, user_text.lower()) for n in menu):
        intent = "change_order"
    print("[debug] intent:", intent)
    print("Routing Completed...")

    if intent == "question":
        return answer_question(user_text), False

    elif intent == "change_order":
        items = extract_items(user_text, order)
        text = user_text.lower()


        # only add/set items need to be mentioned in the message
        good = [i for i in items if i.action not in ("add", "set") or mentioned(i.name, text)]
        bad = [i for i in items if i not in good]

        if good:
            apply_changes(good)
            reply = confirmation_text(good)
        else:
            reply = "Sorry, I didn't catch that."
        if bad:
            reply += " I wasn't sure about one item. Could you repeat it?"
        return reply, False

    elif intent == "confirm_order":
        if not order:
            return "Your order is empty. What would you like to order?"
        return process_order(), True

"""
def check_order():
    if not order:
        return "Your order is empty. What would you like to order?"
    lines = []
    for name, e in order.items():
        line = f"{name} x{e['qty']}"
        if e["notes"]:
            line += f" ({'; '.join(e['notes'])})"
        lines.append(line)
    return "Your order:\n" + "\n".join(lines)
"""
# ---------- Test ----------


def run_call(mode):
    global strikes
    order.clear()
    strikes = 0
    greeting = "Welcome to Burger Barn! How can I assist you today?"
    send_reply(mode, greeting)

    while True:
        user_input = get_user_input(mode)
        if not user_input:
            continue
        response, call_over = handle_turn(user_input)
        send_reply(mode, response)
        if call_over:
            print("[System] Session ended.")
            break

def wait_for_call():
    start = input("[Waiting for a call. Press Enter to simulate one, or q to shut down] ").strip().lower()
    return start != "q"

def wait_for_contact():
    """Returns 'call', 'message', or None to shut down."""
    while True:
        choice = input("\n[Waiting] c = incoming call, m = incoming message, q = quit: ").strip().lower()
        if choice == "c":
            return "call"
        if choice == "m":
            return "message"
        if choice == "q":
            return None
        else:
            print("Please choose the appropriate choice")


def get_user_input(mode):
    if mode == "call":
        text = listen()
        print("You (heard):", text)
        return text
    return input("You: ").strip()


def send_reply(mode, response):
    print("Bot:", response)
    if mode == "call":
        speak(spoken_version(response))   # short version for voice


def spoken_version(response):
    # speak only the total instead of the full receipt
    if "Total:" in response:
        total_line = [l for l in response.split("\n") if l.startswith("Total:")][0]
        return f"Your {total_line.lower()}. Thank you!"
    return response







warm_up()
print("Bot: Welcome to Burger Barn! How can I assist you today?")



while True:
    mode = wait_for_contact()
    if mode is None:
        break
    run_call(mode)


#Next: Make a dictionary for the receipt and print it out at the end of the order.  Make sure total is calculated correctly and autonomously using the python code instead of the llm. Add a bye message.

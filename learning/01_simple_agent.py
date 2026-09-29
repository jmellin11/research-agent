"""
simple_agent.py — a minimal agentic AI, built to learn the mechanics.

The whole idea of an "agent" is the LOOP at the bottom of this file:
    1. Send the conversation to the model.
    2. If the model asks to use a tool, run it and feed the result back.
    3. Repeat until the model stops asking for tools and just answers.

Everything above the loop is just plumbing: defining tools and a helper.
"""

import os
import json
import datetime
from openai import OpenAI   # the OpenAI SDK also talks to Groq — just change the URL

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
# Get a FREE key at https://console.groq.com  ->  save it as the GROQ_API_KEY environment variable (see README).
# (Later, to use OpenAI/Anthropic instead, swap the key, base_url, and model.)
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "openai/gpt-oss-20b"   # a free model on Groq

# ---------------------------------------------------------------------------
# TOOLS — the "hands" of the agent. Each is just a normal Python function.
# ---------------------------------------------------------------------------
def calculator(expression: str) -> str:
    """Evaluate a math expression like '2 * (3 + 4)'."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

def get_current_time(_: str = "") -> str:
    """Return the current date and time."""
    return datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

# Map tool names -> the actual functions, so the loop can call them.
TOOLS = {"calculator": calculator, "get_current_time": get_current_time}

# We also describe each tool to the model so it knows what it can call.
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Do arithmetic. Use for any math.",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "parameters": {"type": "object", "properties": {}}}},
]

# ---------------------------------------------------------------------------
# THE AGENT LOOP — this is the actual "agent". Read this part slowly.
# ---------------------------------------------------------------------------
def run_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    while True:  # keep looping until the model gives a final answer
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SPECS,
        )
        msg = response.choices[0].message
        messages.append(msg)  # remember what the model said

        # If the model did NOT ask for a tool, we're done — print and stop.
        if not msg.tool_calls:
            print("\nAGENT:", msg.content)
            return

        # Otherwise, run each tool the model asked for and feed results back.
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"  [agent decided to use tool: {name}({args})]")
            result = TOOLS[name](**args)
            print(f"  [tool returned: {result}]")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })
        # loop continues: model now sees the tool result and decides next step

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Simple agent. Try: 'What is 47 * 89, and what day is it today?'")
    while True:
        q = input("\nYou: ")
        if q.lower() in ("quit", "exit"):
            break
        run_agent(q)

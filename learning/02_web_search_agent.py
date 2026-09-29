"""
simple_agent_v2.py — the same agent, now with REAL web search.

What changed from v1 (look for the ">>> NEW" comments):
  - a third tool, web_search(), that actually searches the internet
  - it's registered in TOOLS and described in TOOL_SPECS, just like the others

The loop at the bottom did NOT change at all. That's the point:
adding a capability = write a function + register it. The agent gets smarter
without you touching the core logic.
"""

import os
import json
import datetime
from openai import OpenAI

# >>> NEW: the web-search library. DuckDuckGo, free, no API key needed.
# Install it first with:  pip3 install ddgs
try:
    from ddgs import DDGS            # newer package name
except ImportError:
    from duckduckgo_search import DDGS   # older name, just in case

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "openai/gpt-oss-20b"   # try "openai/gpt-oss-120b" to feel the difference

# ---------------------------------------------------------------------------
# TOOLS
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

# >>> NEW TOOL: search the web and return the top results as text.
def web_search(query: str) -> str:
    """Search the internet for current information."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"{r.get('title','')}\n{r.get('body','')}\n{r.get('href','')}"
            for r in results
        )
    except Exception as e:
        return f"Search error: {e}"

# Register every tool so the loop can call it by name.
TOOLS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "web_search": web_search,          # >>> NEW
}

# Describe each tool to the model.
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
    # >>> NEW SPEC: this is how the model learns web_search exists.
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for current events, facts, prices, "
                       "or anything you don't already know. Use this for "
                       "anything recent or that changes over time.",
        "parameters": {"type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]}}},
]

# ---------------------------------------------------------------------------
# THE AGENT LOOP — unchanged from v1.
# ---------------------------------------------------------------------------
def run_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SPECS,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print("\nAGENT:", msg.content)
            return

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"  [agent decided to use tool: {name}({args})]")
            result = TOOLS[name](**args)
            print(f"  [tool returned: {result[:200]}...]")   # trim long results
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Agent with web search. Try: 'Who won the last F1 race?'")
    while True:
        q = input("\nYou: ")
        if q.lower() in ("quit", "exit"):
            break
        run_agent(q)

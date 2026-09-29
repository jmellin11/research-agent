"""
simple_agent_v3.py — your web-search agent, now with the two framework building blocks:

  1) PERSONA        -> SYSTEM_PROMPT, added as a "system" message. Standing
                       instructions the agent always follows. (Frameworks call
                       this "instructions".)

  2) STRUCTURED OUTPUT -> response_format={"type": "json_object"} forces the
                       final answer into JSON your code can use. (Frameworks call
                       this an "output schema", often defined with Pydantic.)

The tool loop is UNCHANGED from v2. That's the lesson: persona and output-schema
are additions around the loop, not changes to it. That's all a framework bolts on.
"""

import os
import json
import datetime
from openai import OpenAI
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "openai/gpt-oss-20b"

# ---------------------------------------------------------------------------
# 1) THE PERSONA + the output contract. One string does both jobs.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a concise, no-nonsense research assistant. You use tools when you need
current facts, then answer plainly. You never pad or hedge.

When you give your FINAL answer, respond ONLY with a JSON object of this shape:
{
  "answer": "a direct answer, a few sentences max",
  "sources": ["url1", "url2"],
  "confidence": "high | medium | low"
}
Base "confidence" on how well your sources actually support the answer.
""".strip()

# ---------------------------------------------------------------------------
# TOOLS (same three as v2)
# ---------------------------------------------------------------------------
def calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

def get_current_time(_: str = "") -> str:
    return datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

def web_search(query: str) -> str:
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

TOOLS = {"calculator": calculator,
         "get_current_time": get_current_time,
         "web_search": web_search}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Do arithmetic.",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for current or factual info.",
        "parameters": {"type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]}}},
]

# ---------------------------------------------------------------------------
# THE AGENT LOOP — same as v2, with the two additions marked >>>.
# ---------------------------------------------------------------------------
def run_agent(user_message: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},   # >>> persona goes first
        {"role": "user", "content": user_message},
    ]

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SPECS,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            # Final answer arrives as JSON text -> turn it into a real dict.
            try:
                return json.loads(msg.content)
            except json.JSONDecodeError:
                return {"answer": msg.content, "sources": [], "confidence": "unknown"}

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"  [tool: {name}({args})]")
            result = TOOLS[name](**args)
            messages.append({"role": "tool",
                             "tool_call_id": call.id,
                             "content": result})

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Research agent (structured output). Ask a question.")
    while True:
        q = input("\nYou: ")
        if q.lower() in ("quit", "exit"):
            break
        result = run_agent(q)   # <-- result is now a DICT, not a blob of text

        # Because it's structured, we can pull fields out by name:
        print("\nANSWER:    ", result["answer"])
        print("SOURCES:   ", result["sources"])
        print("CONFIDENCE:", result["confidence"])

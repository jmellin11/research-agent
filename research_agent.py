"""
research_agent.py — Researcher (live search) -> Analyst, with MEMORY.

v2 fix: the model reached for a tool called web_open that didn't exist.
Now web_open and web_find exist (sharing the same 2-call limit), and if the
model ever invents another tool, the run falls back to writing an answer
from what it already found instead of crashing.

Two kinds of memory:
  1) Conversation memory: the Researcher sees your recent exchanges, so
     follow-ups like "which of them is youngest?" work.
  2) Long-term memory: the conversation is saved to agent_memory.json next to
     this script, so it still remembers after you close Terminal.

Commands at the Topic: prompt
  memory  -> show what it remembers
  forget  -> wipe its memory
  quit    -> exit

Reads GROQ_API_KEY and TAVILY_API_KEY from your environment. Nothing to edit.
Run with:  python3 research_agent.py
"""

import os
import sys
import json
import time
import datetime
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

groq_key = os.environ.get("GROQ_API_KEY")
tavily_key = os.environ.get("TAVILY_API_KEY")
if not groq_key or not tavily_key:
    sys.exit("Missing GROQ_API_KEY or TAVILY_API_KEY. Open a new Terminal window and try again.")

import openai
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool
from agents.exceptions import MaxTurnsExceeded
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from tavily import TavilyClient

groq_client = AsyncOpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
model = OpenAIChatCompletionsModel(model="openai/gpt-oss-120b", openai_client=groq_client)
tavily = TavilyClient(api_key=tavily_key)

TODAY = datetime.date.today().strftime("%B %d, %Y")
MAX_TOOL_CALLS = 2

# ---------------------------------------------------------------------------
# MEMORY — the new part.
# history is a list of messages: {"role": "user"/"assistant", "content": "..."}
# It's saved to a JSON file so it survives closing Terminal.
# ---------------------------------------------------------------------------
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_memory.json")
RECENT_EXCHANGES = 4     # how many past Q&As the agent sees each time (keeps us under Groq's limit)
SAVED_EXCHANGES = 20     # how many are kept in the file


def load_memory() -> list:
    try:
        with open(MEMORY_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_memory(history: list):
    with open(MEMORY_FILE, "w") as f:
        json.dump(history[-SAVED_EXCHANGES * 2:], f, indent=2)


def remember(history: list, question: str, answer: str):
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer[:1200]})   # trimmed to save tokens
    save_memory(history)


def recent(history: list) -> list:
    return history[-RECENT_EXCHANGES * 2:]


# ---------------------------------------------------------------------------
# Search tool (same as v4: accepts the model's habits, hard limit in code)
# ---------------------------------------------------------------------------
last_urls, tool_calls_used, gathered = [], 0, []
last_page = ""      # text of the most recently opened page, for web_find


def over_budget() -> str | None:
    """The hard limit, shared by every tool so none can be used to dodge it."""
    global tool_calls_used
    if tool_calls_used >= MAX_TOOL_CALLS:
        print("  [limit reached: telling the agent to write its answer]")
        return ("SEARCH LIMIT REACHED. You may not use any more tools. Write your "
                "final answer now using only the results you already have.")
    tool_calls_used += 1
    return None


def open_page(url: str) -> str:
    print(f"  [opening: {url[:70]}]")
    try:
        data = tavily.extract(urls=[url])
        pages = data.get("results", [])
        if not pages:
            return f"Couldn't read {url}. Use what you already found."
        global last_page
        last_page = pages[0].get("raw_content", "")
        return last_page[:1500]
    except Exception as e:
        return f"Couldn't open {url} ({e}). Use what you already found."


def open_by_id(id) -> str:
    if isinstance(id, str) and id.startswith("http"):
        return open_page(id)
    if isinstance(id, int) and 0 <= id < len(last_urls):
        return open_page(last_urls[id])
    return "Pass a URL from the search results as 'id' to open it."


@function_tool(strict_mode=False)
def web_search(query: str = "", id: str | int | None = None, cursor: int | None = None) -> str:
    """Search the web, or open a page from earlier results.

    Args:
        query: What to search for.
        id: A URL (or result number) from earlier results to open and read.
        cursor: Ignored.
    """
    global last_urls
    blocked = over_budget()
    if blocked:
        return blocked

    if id is not None and not query:
        result = open_by_id(id)
    elif query:
        print(f"  [searching: {query}]")
        try:
            response = tavily.search(query, max_results=3)
            results = response.get("results", [])
            last_urls = [r.get("url", "") for r in results]
            result = "\n\n".join(
                f"[{i}] {r.get('title','')}\n{(r.get('content') or r.get('snippet',''))[:300]}\n{r.get('url','')}"
                for i, r in enumerate(results)
            ) or "No results found. Answer from what you already know."
        except Exception as e:
            result = f"Search failed ({e}). Answer from what you know."
    else:
        result = "Call web_search with a 'query' to search, or an 'id' URL to open a page."

    gathered.append(result)
    return result


@function_tool(strict_mode=False)
def web_open(id: str | int | None = None, cursor: int | None = None, url: str | None = None) -> str:
    """Open a page from the search results and read it.

    Args:
        id: The page URL, or its result number from the last search.
        cursor: Ignored.
        url: Same as id.
    """
    blocked = over_budget()
    if blocked:
        return blocked
    result = open_by_id(url or id)
    gathered.append(result)
    return result


@function_tool(strict_mode=False)
def web_find(pattern: str = "", cursor: int | None = None) -> str:
    """Find lines containing a word or phrase in the page you last opened.

    Args:
        pattern: The word or phrase to look for.
        cursor: Ignored.
    """
    blocked = over_budget()
    if blocked:
        return blocked
    print(f"  [finding: {pattern}]")
    if not last_page:
        return "No page is open yet. Use web_open first."
    lines = [ln for ln in last_page.splitlines() if pattern.lower() in ln.lower()]
    result = "\n".join(lines[:15])[:1500] or f"'{pattern}' not found on the page."
    gathered.append(result)
    return result


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
researcher = Agent(
    name="Researcher",
    instructions=(
        f"Today's date is {TODAY}. Results from {TODAY[-4:]} are current, not errors. "
        "You are a concise research assistant in an ongoing conversation. Use the "
        "earlier messages to understand follow-ups (words like 'them', 'that', 'he'). "
        "If the answer is already in the conversation, answer directly without "
        "searching. Otherwise do one web_search (don't add old years to the query) "
        "and, if needed, open one page with web_open. Report the key "
        "findings in a short paragraph or two, with source URLs at the end."
    ),
    model=model,
    tools=[web_search, web_open, web_find],
)

writer = Agent(
    name="Writer",
    instructions=(
        f"Today's date is {TODAY}. Using ONLY the search results provided, write the "
        "key findings in a short paragraph or two, with source URLs at the end."
    ),
    model=model,
)

analyst = Agent(
    name="Analyst",
    instructions=(
        "You are a sharp analyst. Given the research you're handed, write a short "
        "brief with exactly three parts: the key takeaway, why it matters, and one "
        "open question worth exploring. Be concise."
    ),
    model=model,
)


def run_chain(topic: str, history: list):
    global last_urls, tool_calls_used, gathered, last_page
    last_urls, tool_calls_used, gathered, last_page = [], 0, [], ""

    # The memory handoff: recent conversation + the new question.
    conversation = recent(history) + [{"role": "user", "content": topic}]

    print("\n=== Agent 1 (Researcher) ===")
    try:
        research_text = Runner.run_sync(researcher, conversation, max_turns=6).final_output
    except (MaxTurnsExceeded, openai.BadRequestError) as e:
        why = "ran out of turns" if isinstance(e, MaxTurnsExceeded) else "made a bad tool call"
        print(f"  [researcher {why}: writing from what it found]")
        material = "\n\n".join(gathered) or "No results were gathered."
        research_text = Runner.run_sync(
            writer, f"Question: {topic}\n\nSearch results:\n{material}"
        ).final_output
    print(research_text)

    print("\n=== Agent 2 (Analyst) ===")
    print(Runner.run_sync(analyst, research_text).final_output)

    remember(history, topic, research_text)


if __name__ == "__main__":
    history = load_memory()
    n = len(history) // 2
    print(f"Research agent with memory. Today is {TODAY}.")
    print(f"Remembering {n} earlier exchange{'s' if n != 1 else ''}. "
          "Commands: 'memory', 'forget', 'quit'.")

    while True:
        topic = input("\nTopic: ").strip()
        cmd = topic.lower()
        if cmd in ("quit", "exit"):
            break
        if not topic:
            continue
        if cmd == "forget":
            history = []
            save_memory(history)
            print("[Memory wiped.]")
            continue
        if cmd == "memory":
            asked = [m["content"] for m in history if m["role"] == "user"]
            print("[I remember you asking about:]" if asked else "[Nothing yet.]")
            for q in asked:
                print(f"  - {q}")
            continue

        for attempt in (1, 2):
            try:
                run_chain(topic, history)
                break
            except openai.RateLimitError:
                if attempt == 1:
                    print("\n[Hit Groq's free-tier speed limit. Waiting 60 seconds, then retrying...]")
                    time.sleep(60)
                else:
                    print("\n[Still rate-limited. Wait a minute and try again.]")
            except openai.APIStatusError as e:
                print(f"\n[Groq error {e.status_code}: {e.message}]")
                break
            except Exception as e:
                print(f"\n[Error: {type(e).__name__}: {e}]")
                break

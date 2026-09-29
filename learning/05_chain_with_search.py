"""
chain_search4.py — Researcher (live Tavily search) -> Analyst, looping.

v4 lesson: instructions are suggestions, code is a rule.
  - Tells the model today's date (it thought it was 2024 and kept re-searching).
  - The tool itself enforces a hard limit of 2 searches/page-opens per topic.
  - If the Researcher still runs out of turns, a fallback writes the answer from
    whatever was already found, so you always get a result.

Reads GROQ_API_KEY and TAVILY_API_KEY from your environment. Nothing to edit.
Run with:  python3 chain_search4.py      (type 'quit' to stop)
"""

import os
import sys
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

# Per-topic state (reset at the start of every topic)
last_urls = []
tool_calls_used = 0
gathered = []          # everything the tool returned, for the fallback


def open_page(url: str) -> str:
    print(f"  [opening: {url[:70]}]")
    try:
        data = tavily.extract(urls=[url])
        pages = data.get("results", [])
        if not pages:
            return f"Couldn't read {url}. Use what you already found."
        return pages[0].get("raw_content", "")[:1500]
    except Exception as e:
        return f"Couldn't open {url} ({e}). Use what you already found."


@function_tool(strict_mode=False)
def web_search(query: str = "", id: str | int | None = None, cursor: int | None = None) -> str:
    """Search the web, or open a page from earlier results.

    Args:
        query: What to search for.
        id: A URL (or result number) from earlier results to open and read.
        cursor: Ignored.
    """
    global last_urls, tool_calls_used

    # THE HARD LIMIT: enforced in code, so the model can't talk its way past it.
    if tool_calls_used >= MAX_TOOL_CALLS:
        print("  [limit reached: telling the agent to write its answer]")
        return ("SEARCH LIMIT REACHED. You may not search again. Write your final "
                "answer now using only the results you already have.")
    tool_calls_used += 1

    if isinstance(id, str) and id.startswith("http"):
        result = open_page(id)
    elif isinstance(id, int) and 0 <= id < len(last_urls):
        result = open_page(last_urls[id])
    elif query:
        print(f"  [searching: {query}]")
        try:
            response = tavily.search(query, max_results=3)
            results = response.get("results", [])
            last_urls = [r.get("url", "") for r in results]
            if not results:
                result = "No results found. Answer from what you already know."
            else:
                result = "\n\n".join(
                    f"[{i}] {r.get('title','')}\n{(r.get('content') or r.get('snippet',''))[:300]}\n{r.get('url','')}"
                    for i, r in enumerate(results)
                )
        except Exception as e:
            result = f"Search failed ({e}). Answer from what you know."
    else:
        result = "Call web_search with a 'query' to search, or an 'id' URL to open a page."

    gathered.append(result)
    return result


researcher = Agent(
    name="Researcher",
    instructions=(
        f"Today's date is {TODAY}. Results from {TODAY[-4:]} are current, not errors. "
        "You are a concise research assistant. Do one web_search with a good query "
        "(don't add old years to it). If the snippets aren't enough, open one page "
        "by calling web_search with its URL as 'id'. Then report the key findings in "
        "a short paragraph or two, with source URLs listed at the end. No fluff."
    ),
    model=model,
    tools=[web_search],
)

# Fallback writer: no tools, just turns gathered results into an answer.
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


def run_chain(topic: str):
    global last_urls, tool_calls_used, gathered
    last_urls, tool_calls_used, gathered = [], 0, []     # fresh budget per topic

    print("\n=== Agent 1 (Researcher, searching the web) ===")
    try:
        research_text = Runner.run_sync(researcher, topic, max_turns=6).final_output
    except MaxTurnsExceeded:
        print("  [researcher ran out of turns: writing from what it found]")
        material = "\n\n".join(gathered) or "No results were gathered."
        research_text = Runner.run_sync(
            writer, f"Topic: {topic}\n\nSearch results:\n{material}"
        ).final_output
    print(research_text)

    print("\n=== Agent 2 (Analyst) ===")
    brief = Runner.run_sync(analyst, research_text)
    print(brief.final_output)


if __name__ == "__main__":
    print(f"Researcher (with live search) -> Analyst. Today is {TODAY}. Type 'quit' to stop.")
    while True:
        topic = input("\nTopic: ").strip()
        if topic.lower() in ("quit", "exit"):
            break
        if not topic:
            continue

        for attempt in (1, 2):
            try:
                run_chain(topic)
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

"""
chain.py — two-agent chain (Researcher -> Analyst) in the OpenAI Agents SDK.

NEW: no key in this file. It reads GROQ_API_KEY from your environment,
so you never paste your key into code again.

Run with:  python3 chain.py
"""

import os
import sys
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"   # we're on Groq, not OpenAI

# Read the key from the environment variable instead of the file.
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    sys.exit("GROQ_API_KEY isn't set. Open a new Terminal window and try again.")

from openai import AsyncOpenAI
from agents import Agent, Runner
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

groq_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)
model = OpenAIChatCompletionsModel(model="openai/gpt-oss-120b", openai_client=groq_client)

# AGENT 1: the Researcher. Explains the topic from what it knows.
researcher = Agent(
    name="Researcher",
    instructions=(
        "You are a concise research assistant. Explain the given topic clearly "
        "from your own knowledge: the key facts and context, in a short paragraph "
        "or two. No fluff."
    ),
    model=model,
)

# AGENT 2: the Analyst. Receives Agent 1's output and turns it into a brief.
analyst = Agent(
    name="Analyst",
    instructions=(
        "You are a sharp analyst. Given the research you're handed, write a short "
        "brief with exactly three parts: the key takeaway, why it matters, and one "
        "open question worth exploring. Be concise."
    ),
    model=model,
)

# THE CHAIN: output of agent 1 flows into agent 2.
if __name__ == "__main__":
    topic = input("Topic: ")

    print("\n=== Agent 1 (Researcher) ===")
    research = Runner.run_sync(researcher, topic)
    print(research.final_output)

    print("\n=== Agent 2 (Analyst), working on the Researcher's output ===")
    brief = Runner.run_sync(analyst, research.final_output)
    print(brief.final_output)

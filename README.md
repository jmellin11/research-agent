# Agents

A research agent built from scratch while learning agentic AI, plus the
step-by-step versions that got there.

## research_agent.py

A two-agent system:

- **Researcher** searches the live web (Tavily), can open pages, and reports
  findings with sources.
- **Analyst** turns those findings into a brief: key takeaway, why it matters,
  and an open question.

It also has:

- **Memory.** Follow-up questions work ("which of them is youngest?"), and the
  conversation is saved to `agent_memory.json` so it remembers across sessions.
  Type `memory` to see it, `forget` to clear it.
- **Guardrails in code.** It knows today's date, has a hard limit on searches,
  and falls back to writing an answer instead of crashing.

Runs on Groq (`openai/gpt-oss-120b`) through the OpenAI Agents SDK.

## Setup

1. Get free API keys from [Groq](https://console.groq.com) and [Tavily](https://tavily.com).
2. Add them to `~/.zshrc`:
   ```
   export GROQ_API_KEY=your_groq_key
   export TAVILY_API_KEY=your_tavily_key
   ```
   Then open a new Terminal window.
3. Install the libraries:
   ```
   pip3 install -r requirements.txt
   ```
4. Run it:
   ```
   python3 research_agent.py
   ```

## learning/

Each file adds one idea on top of the last:

| File | What it adds |
|---|---|
| `01_simple_agent.py` | The core agent loop, hand-built, with a calculator and clock tool |
| `02_web_search_agent.py` | A web search tool (DuckDuckGo) |
| `03_persona_json_agent.py` | A persona (system prompt) and structured JSON output |
| `04_two_agent_chain.py` | The OpenAI Agents SDK, and two agents handing off work |
| `05_chain_with_search.py` | Live Tavily search plus guardrails enforced in code |

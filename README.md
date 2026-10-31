# Local Agent Project

This project is a lightweight local agent setup that can:

- try a real CrewAI + Ollama agent flow when the local model service is available
- automatically fall back to a fast local report when Ollama is not running
- print a readable result in the terminal
- save a markdown summary to `ai_news_report.md`

## Files

- `agent.py` — main script that runs the logic
- `ai_news_report.md` — generated markdown report
- `run_agent.bat` — Windows batch launcher
- `multi_agent.py` — alternative multi-agent example file
- `token_usage.jsonl` — local per-run token history (ignored by Git)

## How the flow works

1. The script checks whether Ollama is running on `http://localhost:11434`.
2. If Ollama is available, it tries the full CrewAI workflow with:
   - a research agent
   - a writer agent
   - a web search tool
3. If Ollama is unavailable or the agent fails, the script falls back to a local visual summary.
4. The final result is printed in the terminal and saved to `ai_news_report.md`.

## Why the fallback exists

A fully connected AI agent setup depends on:

- Python packages in the virtual environment
- Ollama running locally
- compatible tool integrations

If any of those are missing, the script still gives a working result instead of crashing.

## Setup

From the project folder, create or activate the virtual environment:

```powershell
.\.venv\Scripts\activate
```

Install dependencies if needed:

```powershell
.\.venv\Scripts\python -m pip install crewai langchain-community langchain-ollama ddgs
```

For the QA workflow in `app_qa_agent.py`, install its additional dependencies:

```powershell
.\.venv\Scripts\python -m pip install langgraph pydantic ollama pytest
```

## Run it

### Option 1: direct Python

```powershell
.\.venv\Scripts\python agent.py
```

### Option 2: Windows batch file

```powershell
run_agent.bat
```

To run the QA workflow, use the project's virtual environment:

```powershell
.\.venv\Scripts\python app_qa_agent.py
```

## Expected behavior

When Ollama is running, the script tries the real agent flow and may show a longer AI-generated response.

When Ollama is not running, it prints a clean, readable fallback report like:

```text
====================================================================
                          AI AGENT REPORT
====================================================================
Status: success
Message: Fast local agent result is ready.
Timestamp: 2026-10-10 20:35:16

Highlights:
  • Local-first execution keeps the result fast and reliable.
  • The app falls back automatically when Ollama is unavailable.
  • The output is easy to read and suitable for quick review.
====================================================================
```

After each agent run, the script prints the run's token count and totals
recorded today and this month. Counts are appended to `token_usage.jsonl`;
only runs after tracking is enabled are included. To view the totals without
starting the agents, run either:

```powershell
.\.venv\Scripts\python agent.py --usage-today
.\.venv\Scripts\python agent.py --usage-month
```

If the model/provider doesn't report token metrics, that run cannot be included.

## Notes

- `multi_agent.py` is an example of a more advanced multi-agent setup.
- The README is meant to help you understand the execution path without needing to read the whole script.
- If you want the full AI-powered version, ensure Ollama is installed and running before starting the program.

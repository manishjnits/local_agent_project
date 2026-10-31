import argparse
import json
from urllib.error import URLError
from urllib.request import urlopen

from crewai import Agent, Task, Crew, Process, CrewOutput, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from token_usage import (
    current_month_token_usage,
    record_token_usage,
    today_token_usage,
)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"
local_llm = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_BASE_URL,
)


def ensure_ollama_model_available() -> None:
    try:
        with urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as response:
            models = json.load(response).get("models", [])
    except (OSError, URLError) as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_BASE_URL}. Start Ollama and retry."
        ) from exc

    available_models = {model["name"] for model in models}
    if OLLAMA_MODEL not in available_models:
        raise RuntimeError(
            f"Ollama model {OLLAMA_MODEL!r} is not installed. "
            f"Install it with `ollama pull {OLLAMA_MODEL}` and retry."
        )


# 1. Setup the free web search tool wrapper
search_run = DuckDuckGoSearchRun()

@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """Search the internet for raw facts, data, and recent technology updates."""
    return search_run.run(query)

# 2. Build the Agent with string-based LLM integration
research_agent = Agent(
    role='Senior Research Specialist',
    goal='Find and analyze the latest technology breakthroughs on the internet',
    backstory='You are an elite autonomous AI research assistant. You solve problems step-by-step using tools.',
    tools=[web_search_tool],        # Fix 1: Passed the compatible wrapped tool
    llm=local_llm,
    verbose=True
)

# 3. Define the assignment
research_task = Task(
    description='Search the web and summarize the latest updates regarding local AI models released this week.',
    expected_output='A clean 3-bullet-point summary report.',
    agent=research_agent
)


# # Change your task definition to a simple local formatting assignment
# research_task = Task(
#     description='Count the total number of vowels in the word "AUTONOMOUS" and explain the calculation step-by-step.',
#     expected_output='A clear count total followed by a list showing each vowel found.',
#     agent=research_agent
# )


local_crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    process=Process.sequential
)

# Add this second agent below your first one
writer_agent = Agent(
    role='Lead Technical Writer',
    goal='Format raw research into a publication-ready markdown report.',
    backstory='You parse raw web data and convert it into beautiful, executive summaries.',
    tools=[], # The writer doesn't search, it just processes text
    llm=local_llm,
    verbose=True
)

# Create a second task that depends on the first one
write_task = Task(
    description='Take the raw findings from the research specialist and turn them into a clean markdown report.',
    expected_output='A markdown file with clear headings and 3 main bullet points.',
    agent=writer_agent
)

# Update your crew configuration at the bottom
local_crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[research_task, write_task]
)
write_task = Task(
    description='...',
    expected_output='...',
    agent=writer_agent,
    output_file='ai_news_report.md' # This creates a file in your project directory
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    usage_group = parser.add_mutually_exclusive_group()
    usage_group.add_argument(
        "--usage-today",
        action="store_true",
        help="Show token usage recorded today without running the agents.",
    )
    usage_group.add_argument(
        "--usage-month",
        action="store_true",
        help="Show token usage recorded this month without running the agents.",
    )
    args = parser.parse_args()

    if args.usage_today:
        usage = today_token_usage()
        print(
            f"Tokens recorded today: {usage.total_tokens} total "
            f"({usage.prompt_tokens} prompt, "
            f"{usage.completion_tokens} completion)"
        )
        raise SystemExit(0)

    if args.usage_month:
        usage = current_month_token_usage()
        print(
            f"Tokens recorded this month: {usage.total_tokens} total "
            f"({usage.prompt_tokens} prompt, "
            f"{usage.completion_tokens} completion)"
        )
        raise SystemExit(0)

    ensure_ollama_model_available()
    print("🤖 Agent is initializing thoughts and executing tasks...")
    result = local_crew.kickoff()
    usage = result.token_usage if isinstance(result, CrewOutput) else None
    if usage is not None:
        record_token_usage(usage)
        daily_usage = today_token_usage()
        print(
            f"\nTokens this run: {usage.total_tokens} total "
            f"({usage.prompt_tokens} prompt, "
            f"{usage.completion_tokens} completion)"
        )
        print(
            f"Tokens recorded today: {daily_usage.total_tokens} total "
            f"({daily_usage.prompt_tokens} prompt, "
            f"{daily_usage.completion_tokens} completion)"
        )
        monthly_usage = current_month_token_usage()
        print(
            f"Tokens recorded this month: {monthly_usage.total_tokens} total "
            f"({monthly_usage.prompt_tokens} prompt, "
            f"{monthly_usage.completion_tokens} completion)"
        )
    else:
        print("Token usage isn't available for this crew output.")

    print("\n=== FINAL AGENT OUTPUT ===")
    print(result)

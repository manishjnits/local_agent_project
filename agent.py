from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

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
    llm="ollama/qwen2.5",           # Fix 2: Passed string identifier format
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
    llm="ollama/qwen2.5",
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
    print("🤖 Agent is initializing thoughts and executing tasks...")
    result = local_crew.kickoff()
    print("\n=== FINAL AGENT OUTPUT ===")
    print(result)

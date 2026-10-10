from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Setup the free web search tool wrapper
search_run = DuckDuckGoSearchRun()

@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """Search the internet for current facts, product updates, and AI breakthroughs."""
    return search_run.run(query)

# 2. Create Agent 1: The Researcher
researcher = Agent(
    role='Senior Research Specialist',
    goal='Uncover groundbreaking facts and data about the requested topic',
    backstory='You are an elite, analytical researcher. You excel at filtering internet noise to find hard facts.',
    tools=[web_search_tool],
    llm='ollama/qwen2.5',
    verbose=True
)

# 3. Create Agent 2: The Writer
writer = Agent(
    role='Lead Content Editor',
    goal='Transform complex raw data into a clear, highly structured report',
    backstory='You are a master communicator. You take raw, messy facts and structure them beautifully.',
    tools=[],
    llm='ollama/qwen2.5',
    verbose=True
)

# 4. Define Task 1: Researching
task_research = Task(
    description='Search the web and gather key updates about the latest open-source AI advancements released recently.',
    expected_output='A detailed list of raw notes, bullet points, and sources.',
    agent=researcher
)

# 5. Define Task 2: Writing (Takes input from Task 1 automatically)
task_write = Task(
    description='Review the raw research notes and write a compelling, beginner-friendly 3-paragraph summary report.',
    expected_output='A clean markdown report with a catchy title and structured paragraphs.',
    agent=writer,
    output_file='multi_agent_report.md'
)

# 6. Assemble the multi-agent crew to run sequentially
local_crew = Crew(
    agents=[researcher, writer],
    tasks=[task_research, task_write],
    process=Process.sequential
)

# 7. Fire up the crew!
print("🚀 Launching local multi-agent system...")
final_output = local_crew.kickoff()

print("\n=============================================")
print("🎯 FINAL AGENT TEAM REPORT:")
print("=============================================")
print(final_output)
print("=============================================")
print("\nSaved output to multi_agent_report.md")
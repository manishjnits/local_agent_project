from datetime import datetime
from time import perf_counter

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from ddgs import DDGS


def fast_search(query: str, limit: int = 5) -> str:
    """Use the lightweight ddgs package directly for faster live search."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=limit, backend='duckduckgo')
            items = []
            for result in results:
                title = result.get('title', '').strip()
                body = result.get('body', '').strip()
                link = result.get('href', '').strip()
                if title:
                    items.append(f"- {title}: {body} ({link})" if body else f"- {title} ({link})")
            return "\n".join(items) if items else f"No live results found for: {query}"
    except Exception as exc:
        return f"Live search failed: {exc}"


@tool("Movie Search Tool")
def movie_search_tool(query: str) -> str:
    """Return compact live search results for Indian movies released in 2026."""
    return fast_search(f'{query} Indian movies released in 2026', limit=5)


researcher = Agent(
    role='Movie Research Analyst',
    goal='Find Indian movies released in 2026 and verify their release year from live web sources',
    backstory='You are a careful movie researcher who reports only source-supported Indian film releases from 2026.',
    tools=[movie_search_tool],
    llm='ollama/qwen2.5:3b',
    max_iter=2,
    verbose=True
)

writer = Agent(
    role='Movie Ranking Editor',
    goal='Turn verified 2026 Indian movie research into a clear, readable list',
    backstory='You are a reliable editor who excludes unverified titles and releases from other years.',
    tools=[],
    llm='ollama/qwen2.5:3b',
    max_iter=1,
    verbose=True
)

timing_state = {'last_completion': None, 'agent_durations': {}}


def record_agent_completion(agent_name: str) -> None:
    completed_at = perf_counter()
    started_at = timing_state['last_completion']
    if started_at is not None:
        timing_state['agent_durations'][agent_name] = completed_at - started_at
    timing_state['last_completion'] = completed_at


def record_researcher_completion(*_args, **_kwargs) -> None:
    record_agent_completion('Researcher')


def record_writer_completion(*_args, **_kwargs) -> None:
    record_agent_completion('Writer')


research_task = Task(
    description=(
        'Search the live web specifically for Indian movies released in 2026. '
        'Include only movies whose 2026 release is verified by a source; exclude '
        'movies from every other year and do not invent titles or release dates.'
    ),
    expected_output='A source-grounded list of Indian movies released in 2026, with titles, release dates/years, and brief notes.',
    agent=researcher,
    callback=record_researcher_completion
)

write_task = Task(
    description=(
        'Create a clean markdown list using only the verified 2026 Indian movie '
        'releases from the research. Exclude all other years and do not pad the '
        'list with unverified titles. Include source links where available.'
    ),
    expected_output='A readable markdown list of verified Indian movies released in 2026, with release years and source links.',
    agent=writer,
    output_file='movie_top20_india.md',
    callback=record_writer_completion
)

movie_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential
)


def main():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('🚀 Searching live for Indian movies released in 2026...')
    crew_started_at = perf_counter()
    timing_state['last_completion'] = crew_started_at
    timing_state['agent_durations'] = {}
    result = movie_crew.kickoff()
    total_runtime = perf_counter() - crew_started_at
    agent_durations = timing_state['agent_durations']
    runtime_report = (
        '## Agent Runtime\n\n'
        f"- Researcher: {agent_durations.get('Researcher', 0.0):.2f} seconds\n"
        f"- Writer: {agent_durations.get('Writer', 0.0):.2f} seconds\n"
        f'- Total crew runtime: {total_runtime:.2f} seconds\n'
    )

    report = (
        '# Indian Movies Released in 2026\n\n'
        f'Generated on: {now}\n\n'
        '## 2026 Movie List\n\n'
        f'{result}\n'
        f'\n{runtime_report}'
    )

    with open('movie_top20_india.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print('\n=============================================')
    print('🎬 INDIAN MOVIES RELEASED IN 2026')
    print('=============================================')
    print(result)
    print(runtime_report)
    print('=============================================')
    print('\nSaved output to movie_top20_india.md')


if __name__ == '__main__':
    main()

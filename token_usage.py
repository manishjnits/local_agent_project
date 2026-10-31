import json
from datetime import datetime
from pathlib import Path

from crewai.types.usage_metrics import UsageMetrics


USAGE_LOG = Path(__file__).with_name("token_usage.jsonl")


def record_token_usage(
    usage: UsageMetrics, log_path: Path = USAGE_LOG
) -> None:
    now = datetime.now().astimezone()
    entry = {
        "date": now.date().isoformat(),
        "timestamp": now.isoformat(timespec="seconds"),
        "usage": usage.model_dump(),
    }
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry) + "\n")


def today_token_usage(log_path: Path = USAGE_LOG) -> UsageMetrics:
    today = datetime.now().astimezone().date().isoformat()
    return _token_usage_since(today, log_path)


def current_month_token_usage(log_path: Path = USAGE_LOG) -> UsageMetrics:
    month = datetime.now().astimezone().strftime("%Y-%m")
    return _token_usage_since(month, log_path)


def _token_usage_since(date_prefix: str, log_path: Path) -> UsageMetrics:
    totals = UsageMetrics()

    if not log_path.exists():
        return totals

    with log_path.open(encoding="utf-8") as log_file:
        for line in log_file:
            entry = json.loads(line)
            if entry["date"].startswith(date_prefix):
                usage = UsageMetrics.model_validate(entry["usage"])
                totals.add_usage_metrics(usage)

    return totals

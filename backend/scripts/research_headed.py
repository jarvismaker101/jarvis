"""CLI wrapper around the Jarvis research service (headed Chrome + real profile).

Keeps the standalone experiment runnable: `python backend/scripts/research_headed.py`.
The production path is backend.services.research_service (used by the intent router).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from backend.services.research_service import run_research

QUERY = sys.argv[1] if len(sys.argv) > 1 else "upcoming AAA games with realistic graphics"


def main():
    def progress(message):
        print(f"[BROWSER] {message}")

    print(f'\n>>> RESEARCHING: "{QUERY}"\n')
    result = run_research(QUERY, on_progress=progress)
    print("\n===== DETAILED REPORT =====")
    print(result["detailed_markdown"])
    print("===== SPOKEN SUMMARY =====")
    print(result["spoken_summary"])
    print(f"\n[REPORT] saved to {result['report_path']} | "
          f"visited: {result['visited_count']}, failed: {result['failed_count']}")


if __name__ == "__main__":
    main()

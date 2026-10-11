"""CLI entry point for DeepResearch Agent."""

import argparse
import sys
import json

from deepresearch.config import Config
from deepresearch.agent import run_research


def main():
    parser = argparse.ArgumentParser(
        description="DeepResearch Agent — autonomous multi-step research powered by SerpApi"
    )
    parser.add_argument("question", help="The research question to investigate")
    parser.add_argument(
        "--max-iterations", type=int, default=3,
        help="Maximum research iterations (default: 3)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save report to file (default: print to stdout)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output full state as JSON instead of just the report"
    )
    args = parser.parse_args()

    config = Config()
    errors = config.validate()
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    def progress_callback(node: str, state: dict):
        logs = state.get("progress_log", [])
        if logs:
            print(logs[-1], file=sys.stderr)

    print(f"\n🔬 DeepResearch Agent v1.0.0", file=sys.stderr)
    print(f"   Question: {args.question}", file=sys.stderr)
    print(f"   LLM: {config.llm_provider}/{config.llm_model}", file=sys.stderr)
    print(f"   Max iterations: {args.max_iterations}\n", file=sys.stderr)

    result = run_research(
        question=args.question,
        max_iterations=args.max_iterations,
        callback=progress_callback,
    )

    if args.json:
        output = json.dumps(result, indent=2, default=str)
    else:
        output = result.get("report", "No report generated.")

        # Append metadata summary
        meta = result.get("report_metadata", {})
        if meta:
            output += "\n\n---\n"
            output += f"*Research completed in {meta.get('iterations', '?')} iterations | "
            output += f"{meta.get('total_sources', '?')} sources found | "
            output += f"{meta.get('cited_sources', '?')} cited | "
            output += f"Engines: {', '.join(meta.get('engines_used', []))} | "
            output += f"Quality: {meta.get('quality_score', 0):.0%}*"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n📄 Report saved to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()

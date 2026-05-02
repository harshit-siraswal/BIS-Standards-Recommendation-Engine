"""Run the BIS pipeline on the public test set and evaluate the results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run and evaluate BIS recommendations.")
    parser.add_argument("--input", default="public_test_set.json", help="Path to evaluator input JSON.")
    parser.add_argument("--results", default="results.json", help="Path to results JSON.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild catalog and indexes before running.")
    args = parser.parse_args(argv)

    command = [
        sys.executable,
        "run.py",
        "--input",
        args.input,
        "--output",
        args.results,
    ]
    if args.rebuild:
        command.append("--rebuild")
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, "scripts/check_format.py", args.results], check=True)
    subprocess.run([sys.executable, "eval_script.py", "--results", args.results], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

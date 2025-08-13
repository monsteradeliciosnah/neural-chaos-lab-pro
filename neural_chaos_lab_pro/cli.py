from __future__ import annotations

import argparse
import json
from typing import Any, Dict


def run_command(args: argparse.Namespace) -> Dict[str, Any]:
    return {"ok": True, "cmd": "run", "params": {"steps": args.steps}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nclp", description="Neural Chaos Lab Pro CLI"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a short simulation")
    p_run.add_argument("--steps", type=int, default=100, help="Number of steps")
    p_run.set_defaults(func=run_command)

    ns = parser.parse_args(argv)
    out = ns.func(ns)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

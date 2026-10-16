"""Command line entry point:  python -m battery_pipeline run | schedule"""
from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")   # no telemetry
    parser = argparse.ArgumentParser(prog="battery_pipeline", description="EV battery data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the whole pipeline once")
    run.add_argument("--root", default=None, help="project root (default: this repository)")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--vehicles", type=int, default=24)
    run.add_argument("--days", type=int, default=180)
    run.add_argument("--no-regenerate", action="store_true", help="reuse existing data/raw files")

    sched = sub.add_parser("schedule", help="run the pipeline on a cron schedule (keeps running)")
    sched.add_argument("--cron", default="0 6 * * *", help='cron expression, default "0 6 * * *"')

    args = parser.parse_args(argv)
    from .flows import battery_pipeline, serve_on_schedule   # import after the env var is set

    if args.command == "run":
        battery_pipeline(root=args.root, regenerate=not args.no_regenerate, seed=args.seed,
                         n_vehicles=args.vehicles, days=args.days)
    else:
        serve_on_schedule(args.cron)


if __name__ == "__main__":
    main()

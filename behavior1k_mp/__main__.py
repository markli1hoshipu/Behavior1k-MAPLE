"""maple — unified CLI for BEHAVIOR-1K MAPLE.

Two subcommands:
  collect  Data-collection driver — runs the hybrid MP + X-VLA pipeline and
           writes parquet + hdf5 + annotations + phase_segments per episode.
  eval     Leaderboard rollout driver — runs metric-only rollouts and packages
           the challenge submission archive.

Each subcommand shells out to `behavior1k_mp.<subcmd>.cli:main`.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="maple", description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    subparsers.add_parser("collect", help="data-collection driver", add_help=False)
    subparsers.add_parser("eval",    help="leaderboard rollout driver", add_help=False)

    # Split argv so the subcommand parses its own args.
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        parser.print_help(sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]

    if cmd in ("-h", "--help"):
        parser.print_help()
        return 0

    if cmd == "collect":
        from .collect.cli import main as _main
        return _main(rest)
    if cmd == "eval":
        from .evaluation.cli import main as _main
        return _main(rest)

    parser.error(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())

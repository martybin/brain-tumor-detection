#!/usr/bin/env python3
"""Open the local MRI review dashboard."""

from __future__ import annotations

import argparse

import bootstrap  # noqa: F401
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Research MRI dashboard. Not a diagnostic device.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run("dashboard.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import uvicorn
from src.config.settings import get_settings


def parse_args() -> argparse.Namespace:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Run the local FastAPI server for diabetes readmission prediction."
    )
    parser.add_argument(
        "--host",
        default=settings.api_host,
        help="Host interface to bind the API server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.api_port,
        help="Port to bind the API server.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=settings.api_reload,
        help="Enable auto-reload for local development.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    docs_url = f"http://{args.host}:{args.port}/docs"

    print("Starting Diabetes Readmission Local API")
    print(f"- host: {args.host}")
    print(f"- port: {args.port}")
    print(f"- reload: {args.reload}")
    print(f"- docs: {docs_url}")

    uvicorn.run(
        "src.serving.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

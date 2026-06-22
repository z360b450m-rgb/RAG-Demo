#!/usr/bin/env python
"""
RAG Agent — CLI entry point.

Usage:
    python main.py                          # interactive chat
    python main.py --query "你的问题"        # single query
    python main.py --ingest document.pdf    # index a document
    python main.py --list                   # list indexed sources
    python main.py --mode direct            # use direct RAG mode (no agent)
"""

import argparse
import io
import sys
from pathlib import Path

# Fix encoding for Windows GBK consoles
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

from app_config import validate_config
from core.agent import RAGAgent
from core.pipeline import DirectRAGPipeline


def cmd_ingest(file_path: str):
    pipeline = DirectRAGPipeline()
    count = pipeline.ingest_document(Path(file_path))
    print(f"Ingested {count} chunks from '{file_path}'")


def cmd_list():
    pipeline = DirectRAGPipeline()
    sources = pipeline.list_ingested_sources()
    if sources:
        print("Indexed documents:")
        for s in sources:
            print(f"  - {s}")
    else:
        print("No documents indexed.")


def cmd_query(question: str, mode: str = "agent"):
    if mode == "agent":
        agent = RAGAgent()
        print(f"\n> {question}\n")
        result = agent.run(question)
        print(result)
    else:
        pipeline = DirectRAGPipeline()
        stream = pipeline.query_stream(question, [])
        for chunk in stream:
            print(chunk, end="", flush=True)
        print()


def cmd_interactive(mode: str = "agent"):
    agent = RAGAgent() if mode == "agent" else None
    pipeline = DirectRAGPipeline() if mode == "direct" else None

    print("RAG Agent — interactive mode")
    print(f"Mode: {mode}")
    print("Type '/exit' to quit, '/clear' to reset memory\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit"):
            print("Goodbye.")
            break
        if user_input.lower() == "/clear":
            if agent:
                agent.clear_memory()
            print("Memory cleared.")
            continue

        if mode == "agent" and agent:
            result = agent.run(user_input)
            print(f"\n{result}\n")
        elif pipeline:
            stream = pipeline.query_stream(user_input, [])
            for chunk in stream:
                print(chunk, end="", flush=True)
            print("\n")


def main():
    validate_config()

    parser = argparse.ArgumentParser(description="RAG Agent CLI")
    parser.add_argument(
        "--query", "-q", type=str, help="Single query (non-interactive)"
    )
    parser.add_argument("--ingest", "-i", type=str, help="Ingest a document")
    parser.add_argument("--list", "-l", action="store_true", help="List sources")
    parser.add_argument(
        "--mode", "-m", choices=["agent", "direct"], default="agent",
        help="Query mode (default: agent)",
    )
    args = parser.parse_args()

    if args.ingest:
        cmd_ingest(args.ingest)
    elif args.list:
        cmd_list()
    elif args.query:
        cmd_query(args.query, mode=args.mode)
    else:
        cmd_interactive(mode=args.mode)


if __name__ == "__main__":
    main()

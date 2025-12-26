#!/usr/bin/env python3

"""
CLI script to build RAG index from knowledge base documents

Usage:
    python -m services.rag.build_index
    python -m services.rag.build_index --path /custom/knowledge/base
    python -m services.rag.build_index --name custom_index
    python -m services.rag.build_index --help
"""

from services.rag.rag_service import build_rag_index, KNOWLEDGE_BASE_DIR, INDEX_DIR
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build RAG index from knowledge base documents"
    )

    parser.add_argument(
        "--path",
        type=str,
        default=KNOWLEDGE_BASE_DIR,
        help=f"Path to knowledge base directory (default: {KNOWLEDGE_BASE_DIR})"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="default",
        help="Name for the index (default: default)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index exists"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RAG INDEX BUILDER")
    print("=" * 70)
    print(f"Knowledge Base: {args.path}")
    print(f"Index Name: {args.name}")
    print(f"Output Directory: {INDEX_DIR}")
    print("=" * 70)
    print()

    # Build index
    try:
        rag_index = build_rag_index(
            knowledge_base_path=args.path,
            index_name=args.name
        )

        print()
        print("=" * 70)
        print("SUCCESS")
        print("=" * 70)
        print(f"Index: {rag_index.index_path}")
        print(f"Metadata: {rag_index.metadata_path}")
        print(f"Chunks: {len(rag_index.chunks)}")
        print(f"Dimension: {rag_index.faiss_index.d}")
        print(f"Created: {rag_index.created_at}")
        print("=" * 70)

        return 0

    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(f"Failed to build index: {e}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

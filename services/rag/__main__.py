"""Entry point for RAG service CLI commands."""

from services.rag.build_index import main
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


if __name__ == "__main__":
    sys.exit(main())

# components/rag_service.py

"""
Enhanced RAG Service
Document ingestion, embedding, indexing, and retrieval with caching
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np
import faiss
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
SITE_URL = os.getenv("SITE_URL", "http://localhost:8501")
APP_NAME = os.getenv("APP_NAME", "DesignAnalysisPoc")

# Index storage paths
INDEX_DIR = os.path.join(STORAGE_DIR, "rag_indices")
CACHE_DIR = os.path.join(STORAGE_DIR, "rag_cache")

# Create directories if missing
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Constants
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks
EMBEDDING_DIMENSION = 1536  # OpenAI text-embedding-3-small
DEFAULT_TOP_K = 5


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TextChunk:
    """Single chunk of text with metadata."""
    doc_name: str
    chunk_id: int
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding numpy embedding)."""
        return {
            "doc_name": self.doc_name,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page": self.page,
            "section": self.section,
        }


@dataclass
class RAGResult:
    """Result of RAG retrieval."""
    snippet: str
    doc_name: str
    chunk_id: int
    relevance_score: float
    page: Optional[int] = None
    section: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "snippet": self.snippet,
            "doc_name": self.doc_name,
            "chunk_id": self.chunk_id,
            "relevance_score": float(self.relevance_score),
            "page": self.page,
            "section": self.section,
            "metadata": self.metadata,
        }


@dataclass
class RAGIndex:
    """Wrapper for FAISS index with metadata."""
    faiss_index: faiss.Index
    chunks: List[TextChunk]
    index_path: str
    metadata_path: str
    created_at: str = ""

    def save(self):
        """Save index and metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.faiss_index, self.index_path)

        # Save metadata (chunks without embeddings)
        metadata = {
            "chunks": [c.to_dict() for c in self.chunks],
            "created_at": self.created_at,
            "dimension": self.faiss_index.d,
            "num_items": self.faiss_index.ntotal,
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load(index_path: str, metadata_path: str) -> 'RAGIndex':
        """Load index and metadata from disk."""
        faiss_index = faiss.read_index(index_path)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        chunks = []
        for chunk_dict in metadata["chunks"]:
            chunks.append(TextChunk(**chunk_dict))

        return RAGIndex(
            faiss_index=faiss_index,
            chunks=chunks,
            index_path=index_path,
            metadata_path=metadata_path,
            created_at=metadata.get("created_at", ""),
        )


# ============================================================================
# EMBEDDING FUNCTIONS
# ============================================================================

def create_text_embedding(text: str) -> np.ndarray:
    """
    Generate text embedding using OpenRouter API

    Args:
        text: Text to embed

    Returns:
        np.ndarray: 1536-dimensional embedding vector
    """
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": SITE_URL,
            "X-Title": APP_NAME,
            "Content-Type": "application/json"
        }

        data = {
            "model": EMBEDDING_MODEL,
            "input": text
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            embedding = np.array(
                result['data'][0]['embedding'], dtype='float32')
            return embedding
        else:
            print(f"[WARN] Embedding API error: {response.status_code}")
            return np.zeros(EMBEDDING_DIMENSION, dtype='float32')

    except Exception as e:
        print(f"[WARN] Error creating embedding: {e}")
        return np.zeros(EMBEDDING_DIMENSION, dtype='float32')


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def load_document(file_path: str) -> Tuple[str, str]:
    """
    Load document from file

    Args:
        file_path: Path to document file

    Returns:
        Tuple[str, str]: (content, doc_name)
    """
    try:
        path = Path(file_path)
        doc_name = path.stem

        if path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert JSON to readable text
                content = json.dumps(data, indent=2)
        else:
            # Try reading as text
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

        return content, doc_name

    except Exception as e:
        print(f"[WARN] Error loading document {file_path}: {e}")
        return "", ""


def chunk_text(text: str, doc_name: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[TextChunk]:
    """
    Split text into overlapping chunks

    Args:
        text: Text to chunk
        doc_name: Document name for reference
        chunk_size: Size of each chunk
        overlap: Overlap between chunks

    Returns:
        List[TextChunk]: List of text chunks
    """
    chunks = []
    chunk_id = 0

    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size].strip()

        if len(chunk_text) > 50:  # Minimum chunk size
            chunks.append(TextChunk(
                doc_name=doc_name,
                chunk_id=chunk_id,
                text=chunk_text
            ))
            chunk_id += 1

    return chunks


def ingest_documents(knowledge_base_path: str = None) -> List[TextChunk]:
    """
    Ingest all documents from knowledge base directory

    Args:
        knowledge_base_path: Path to knowledge base directory

    Returns:
        List[TextChunk]: All chunks from all documents
    """
    if knowledge_base_path is None:
        knowledge_base_path = KNOWLEDGE_BASE_DIR

    os.makedirs(knowledge_base_path, exist_ok=True)

    all_chunks = []

    # Find all documents
    doc_files = list(Path(knowledge_base_path).glob("*.txt")) + \
        list(Path(knowledge_base_path).glob("*.json"))

    if not doc_files:
        print(f"[INFO] No documents found in {knowledge_base_path}")
        return []

    print(f"[INFO] Ingesting {len(doc_files)} documents...")

    for file_path in doc_files:
        print(f"  Processing: {file_path.name}")

        content, doc_name = load_document(str(file_path))
        if content:
            chunks = chunk_text(content, doc_name)
            all_chunks.extend(chunks)
            print(f"    -> {len(chunks)} chunks created")

    print(f"[INFO] Total chunks created: {len(all_chunks)}")
    return all_chunks


# ============================================================================
# INDEXING
# ============================================================================

def build_rag_index(knowledge_base_path: str = None,
                    index_name: str = "default") -> RAGIndex:
    """
    Build FAISS index from knowledge base documents

    Args:
        knowledge_base_path: Path to knowledge base directory
        index_name: Name for the index

    Returns:
        RAGIndex: Built RAG index with metadata
    """
    from datetime import datetime

    # Ingest documents
    chunks = ingest_documents(knowledge_base_path)

    if not chunks:
        print("[WARN] No chunks to index, creating empty index")
        faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        return RAGIndex(
            faiss_index=faiss_index,
            chunks=[],
            index_path=os.path.join(INDEX_DIR, f"{index_name}.index"),
            metadata_path=os.path.join(
                INDEX_DIR, f"{index_name}_metadata.json"),
            created_at=datetime.utcnow().isoformat(),
        )

    # Create embeddings
    print("[INFO] Creating embeddings...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(chunks)} chunks embedded")

        embedding = create_text_embedding(chunk.text)
        chunk.embedding = embedding
        embeddings.append(embedding)

    # Build FAISS index
    print("[INFO] Building FAISS index...")
    embeddings_array = np.array(embeddings, dtype='float32')
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    faiss_index.add(embeddings_array)

    # Create RAG index
    rag_index = RAGIndex(
        faiss_index=faiss_index,
        chunks=chunks,
        index_path=os.path.join(INDEX_DIR, f"{index_name}.index"),
        metadata_path=os.path.join(INDEX_DIR, f"{index_name}_metadata.json"),
        created_at=datetime.utcnow().isoformat(),
    )

    # Save index
    rag_index.save()
    print(f"[INFO] Index saved: {rag_index.index_path}")

    return rag_index


# ============================================================================
# RETRIEVAL
# ============================================================================

class RAGRetriever:
    """Retriever with caching for fast lookups."""

    def __init__(self, index_name: str = "default"):
        """Initialize retriever with cached index."""
        self.index_name = index_name
        self.index_path = os.path.join(INDEX_DIR, f"{index_name}.index")
        self.metadata_path = os.path.join(
            INDEX_DIR, f"{index_name}_metadata.json")
        self.cache_path = os.path.join(CACHE_DIR, f"{index_name}_cache.pkl")

        self.rag_index: Optional[RAGIndex] = None
        self._load_index()

    def _load_index(self):
        """Load index from disk or cache."""
        try:
            if os.path.exists(self.index_path):
                self.rag_index = RAGIndex.load(
                    self.index_path, self.metadata_path)
                print(
                    f"[INFO] Loaded RAG index: {len(self.rag_index.chunks)} chunks")
            else:
                print(f"[WARN] Index not found: {self.index_path}")
                self.rag_index = None
        except Exception as e:
            print(f"[WARN] Error loading index: {e}")
            self.rag_index = None

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[RAGResult]:
        """
        Retrieve relevant chunks for query

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List[RAGResult]: Ranked results with relevance scores
        """
        if not self.rag_index or not self.rag_index.chunks:
            print("[WARN] RAG index not initialized")
            return []

        try:
            # Create query embedding
            query_embedding = create_text_embedding(query)
            query_vector = np.array([query_embedding], dtype='float32')

            # Search FAISS
            distances, indices = self.rag_index.faiss_index.search(
                query_vector,
                min(top_k, len(self.rag_index.chunks))
            )

            # Convert distances to relevance scores (L2 distance)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.rag_index.chunks):
                    chunk = self.rag_index.chunks[idx]

                    # Convert L2 distance to similarity (0-1, higher is better)
                    # Normalize using 1 / (1 + distance)
                    relevance = 1.0 / (1.0 + float(dist))

                    result = RAGResult(
                        snippet=chunk.text[:200] +
                        "..." if len(chunk.text) > 200 else chunk.text,
                        doc_name=chunk.doc_name,
                        chunk_id=chunk.chunk_id,
                        relevance_score=relevance,
                        page=chunk.page,
                        section=chunk.section,
                        metadata={
                            "full_text": chunk.text,
                            "chunk_size": len(chunk.text),
                        }
                    )
                    results.append(result)

            return results[:top_k]

        except Exception as e:
            print(f"[WARN] Error retrieving: {e}")
            return []

    def retrieve_with_query_expansion(self, query: str,
                                      top_k: int = DEFAULT_TOP_K) -> List[RAGResult]:
        """
        Retrieve using query expansion for better results

        Args:
            query: Original query
            top_k: Number of results

        Returns:
            List[RAGResult]: Combined and deduplicated results
        """
        # Simple query expansion: search for query and key terms
        results_by_id = {}

        # Main query
        results = self.retrieve(query, top_k)
        for result in results:
            key = (result.doc_name, result.chunk_id)
            results_by_id[key] = result

        # If few results, try searching for first few key terms
        terms = query.split()[:3]
        for term in terms:
            if len(term) > 3:  # Only search meaningful terms
                term_results = self.retrieve(term, top_k // 2)
                for result in term_results:
                    key = (result.doc_name, result.chunk_id)
                    if key not in results_by_id:
                        # Reduce score for expanded queries
                        result.relevance_score *= 0.8
                        results_by_id[key] = result

        # Sort by relevance
        all_results = sorted(
            results_by_id.values(),
            key=lambda x: x.relevance_score,
            reverse=True
        )

        return all_results[:top_k]


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_retriever_instance: Optional[RAGRetriever] = None


def get_rag_retriever(index_name: str = "default") -> RAGRetriever:
    """Get or create RAG retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever(index_name)
    return _retriever_instance


def clear_rag_cache():
    """Clear RAG retriever cache."""
    global _retriever_instance
    _retriever_instance = None

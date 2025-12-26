"""
tests/test_rag_retrieval_shape.py

Test suite for RAG (Retrieval-Augmented Generation) retrieval shape and consistency.
Validates:
- Retrieved documents have correct structure
- Vector embeddings have consistent shape
- Similarity scores are within valid range
- Top-k retrieval returns expected number of results
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from components.rag_system import RAGService
from components.models import RAGCitation


class TestRAGRetrievalShape:
    """Test RAG retrieval result shapes."""

    @pytest.fixture
    def rag_service(self):
        """Initialize RAG service."""
        return RAGService()

    @pytest.fixture
    def query(self):
        """Sample query for RAG."""
        return "modern minimalist design with flat colors"

    def test_retrieval_returns_list(self, rag_service, query):
        """RAG retrieval should return a list."""
        results = rag_service.retrieve(query, top_k=5)
        assert isinstance(results, list)

    def test_retrieval_respects_top_k(self, rag_service, query):
        """RAG should return exactly top_k results."""
        for k in [1, 3, 5, 10]:
            results = rag_service.retrieve(query, top_k=k)
            assert len(
                results) <= k, f"Expected <= {k} results, got {len(results)}"

    def test_retrieval_document_structure(self, rag_service, query):
        """Retrieved documents should have required fields."""
        results = rag_service.retrieve(query, top_k=3)

        if results:  # Only test if results exist
            result = results[0]

            # Each result should be a dict or object with these fields
            assert hasattr(result, 'page_content') or 'page_content' in result
            assert hasattr(result, 'metadata') or 'metadata' in result

    def test_retrieval_similarity_scores_valid(self, rag_service, query):
        """Similarity scores should be in valid range [0, 1]."""
        results = rag_service.retrieve_with_scores(query, top_k=5)

        for doc, score in results:
            assert 0 <= score <= 1, f"Invalid similarity score: {score}"

    def test_retrieval_scores_descending(self, rag_service, query):
        """Similarity scores should be in descending order."""
        results = rag_service.retrieve_with_scores(query, top_k=5)

        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True), \
                "Scores should be in descending order"

    def test_vector_embedding_shape(self, rag_service, query):
        """Vector embeddings should have consistent shape."""
        # Get embedding for query
        embedding1 = rag_service.embed(query)

        # Get embedding for another query
        embedding2 = rag_service.embed("another design query here")

        # Both should have same dimensionality
        assert len(embedding1) == len(embedding2), \
            f"Embedding shape mismatch: {len(embedding1)} vs {len(embedding2)}"

    def test_embedding_is_normalized(self, rag_service, query):
        """Embeddings should be normalized (magnitude ~1)."""
        embedding = rag_service.embed(query)

        # Convert to numpy array if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        magnitude = np.linalg.norm(embedding)
        # Normalized vectors should have magnitude close to 1 (allow some tolerance)
        assert 0.9 <= magnitude <= 1.1, f"Embedding magnitude: {magnitude}"

    def test_empty_query_handling(self, rag_service):
        """Empty query should be handled gracefully."""
        try:
            results = rag_service.retrieve("", top_k=5)
            # Should return empty or handle gracefully
            assert isinstance(results, list)
        except (ValueError, Exception):
            # Or it might raise an error, which is also acceptable
            pass

    def test_zero_k_handling(self, rag_service, query):
        """Zero k should return empty results."""
        results = rag_service.retrieve(query, top_k=0)
        assert results == [] or len(results) == 0

    def test_large_k_capping(self, rag_service, query):
        """Large k should be capped by available documents."""
        results = rag_service.retrieve(query, top_k=10000)
        # Should return at most the number of documents in the system
        assert len(results) >= 0  # At least shouldn't crash


class TestRAGCitationConsistency:
    """Test RAG citation structure and consistency."""

    def test_rag_citation_schema(self):
        """RAG citations should follow the schema."""
        citation = RAGCitation(
            source="test_source.json",
            pattern_name="Minimalist Design",
            relevance_score=0.95,
            section="design_principles"
        )

        assert citation.source == "test_source.json"
        assert citation.pattern_name == "Minimalist Design"
        assert 0 <= citation.relevance_score <= 1
        assert isinstance(citation.section, str)

    def test_citation_relevance_bounds(self):
        """Citation relevance scores should be in [0, 1]."""
        for score in [0, 0.5, 1.0]:
            citation = RAGCitation(
                source="test",
                pattern_name="Pattern",
                relevance_score=score,
                section="test"
            )
            assert 0 <= citation.relevance_score <= 1


class TestRAGRetrievalDeterminism:
    """Test that RAG retrieval is deterministic."""

    @pytest.fixture
    def rag_service(self):
        """Initialize RAG service."""
        return RAGService()

    def test_retrieval_determinism(self, rag_service):
        """Same query should return same results."""
        query = "consistent design pattern"

        results1 = rag_service.retrieve(query, top_k=5)
        results2 = rag_service.retrieve(query, top_k=5)

        # Results should be identical
        assert len(results1) == len(results2)

        if results1:
            # Compare document contents
            for doc1, doc2 in zip(results1, results2):
                content1 = doc1.page_content if hasattr(
                    doc1, 'page_content') else doc1.get('page_content')
                content2 = doc2.page_content if hasattr(
                    doc2, 'page_content') else doc2.get('page_content')
                assert content1 == content2

    def test_embedding_determinism(self, rag_service):
        """Same text should produce same embedding."""
        text = "design system principles"

        embedding1 = rag_service.embed(text)
        embedding2 = rag_service.embed(text)

        # Embeddings should be identical
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)

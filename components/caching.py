"""
Caching layer for embeddings, RAG results, and agent outputs.
Provides intelligent caching with hash-based keys to avoid redundant computation.
"""

import hashlib
import json
import pickle
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Multi-layer cache manager for:
    - Image embeddings (per image hash)
    - RAG results (per query/settings)
    - Agent outputs (per image_hash + platform + creative_type + agent_version)
    """

    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24):
        """
        Initialize cache manager.

        Args:
            cache_dir: Root cache directory
            ttl_hours: Cache time-to-live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = timedelta(hours=ttl_hours)

        # Create subdirectories
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        self.rag_cache_dir = self.cache_dir / "rag"
        self.agent_cache_dir = self.cache_dir / "agents"
        self.metadata_cache_dir = self.cache_dir / "metadata"

        for cache_subdir in [
            self.embedding_cache_dir,
            self.rag_cache_dir,
            self.agent_cache_dir,
            self.metadata_cache_dir,
        ]:
            cache_subdir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Cache manager initialized: {self.cache_dir}")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    @staticmethod
    def _hash_key(*parts: str) -> str:
        """
        Create deterministic hash key from parts.

        Args:
            *parts: String parts to hash

        Returns:
            Hex hash string (first 12 chars)
        """
        combined = "|".join(str(p) for p in parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:12]

    def _is_cache_valid(self, filepath: Path) -> bool:
        """Check if cache file exists and is not expired."""
        if not filepath.exists():
            return False

        age = datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)
        if age > self.ttl:
            logger.debug(f"Cache expired: {filepath}")
            return False

        return True

    # ========================================================================
    # Embedding Cache (per image hash)
    # ========================================================================

    def get_embedding(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached embedding for image.

        Args:
            image_hash: SHA256 hash of image bytes

        Returns:
            Embedding dict or None if not cached
        """
        cache_key = self._hash_key(image_hash, "embedding")
        cache_file = self.embedding_cache_dir / f"{cache_key}.pkl"

        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                logger.debug(f"Embedding cache HIT: {image_hash[:8]}...")
                return data
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                return None

        return None

    def cache_embedding(
        self,
        image_hash: str,
        embedding: Dict[str, Any],
    ) -> None:
        """
        Cache embedding for image.

        Args:
            image_hash: SHA256 hash of image bytes
            embedding: Embedding data to cache
        """
        cache_key = self._hash_key(image_hash, "embedding")
        cache_file = self.embedding_cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(embedding, f)
            logger.debug(f"Embedding cached: {image_hash[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    # ========================================================================
    # RAG Cache (per query + settings)
    # ========================================================================

    def get_rag_results(
        self,
        query: str,
        platform: str,
        creative_type: str,
        top_k: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached RAG results.

        Args:
            query: Search query
            platform: Platform (Instagram, Facebook, etc.)
            creative_type: Creative type (Marketing Creative, etc.)
            top_k: Number of results

        Returns:
            RAG results or None if not cached
        """
        cache_key = self._hash_key(query, platform, creative_type, str(top_k))
        cache_file = self.rag_cache_dir / f"{cache_key}.pkl"

        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                logger.debug(f"RAG cache HIT: {query[:20]}... ({platform})")
                return data
            except Exception as e:
                logger.warning(f"Failed to load RAG cache: {e}")
                return None

        return None

    def cache_rag_results(
        self,
        query: str,
        platform: str,
        creative_type: str,
        results: Dict[str, Any],
        top_k: int = 5,
    ) -> None:
        """
        Cache RAG results.

        Args:
            query: Search query
            platform: Platform (Instagram, Facebook, etc.)
            creative_type: Creative type (Marketing Creative, etc.)
            results: Results to cache
            top_k: Number of results
        """
        cache_key = self._hash_key(query, platform, creative_type, str(top_k))
        cache_file = self.rag_cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)
            logger.debug(f"RAG results cached: {query[:20]}... ({platform})")
        except Exception as e:
            logger.warning(f"Failed to cache RAG results: {e}")

    # ========================================================================
    # Agent Cache (per image_hash + platform + creative_type + agent_version)
    # ========================================================================

    def get_agent_result(
        self,
        image_hash: str,
        agent_name: str,
        platform: str,
        creative_type: str,
        agent_version: str = "1.0",
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached agent result.

        Args:
            image_hash: SHA256 hash of image bytes
            agent_name: Name of agent (ux_critique, market_research, etc.)
            platform: Platform (Instagram, Facebook, etc.)
            creative_type: Creative type (Marketing Creative, etc.)
            agent_version: Agent version

        Returns:
            Agent result or None if not cached
        """
        cache_key = self._hash_key(
            image_hash, agent_name, platform, creative_type, agent_version
        )
        cache_file = self.agent_cache_dir / f"{cache_key}.pkl"

        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                logger.debug(
                    f"Agent cache HIT: {agent_name} ({image_hash[:8]}... on {platform})"
                )
                return data
            except Exception as e:
                logger.warning(f"Failed to load agent cache: {e}")
                return None

        return None

    def cache_agent_result(
        self,
        image_hash: str,
        agent_name: str,
        platform: str,
        creative_type: str,
        result: Dict[str, Any],
        agent_version: str = "1.0",
    ) -> None:
        """
        Cache agent result.

        Args:
            image_hash: SHA256 hash of image bytes
            agent_name: Name of agent (ux_critique, market_research, etc.)
            platform: Platform (Instagram, Facebook, etc.)
            creative_type: Creative type (Marketing Creative, etc.)
            result: Result to cache
            agent_version: Agent version
        """
        cache_key = self._hash_key(
            image_hash, agent_name, platform, creative_type, agent_version
        )
        cache_file = self.agent_cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            logger.debug(
                f"Agent result cached: {agent_name} ({image_hash[:8]}... on {platform})"
            )
        except Exception as e:
            logger.warning(f"Failed to cache agent result: {e}")

    # ========================================================================
    # Cache Statistics
    # ========================================================================

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache counts and sizes
        """
        def count_files(directory: Path) -> Tuple[int, float]:
            """Count files and total size in directory."""
            if not directory.exists():
                return 0, 0.0

            count = len(list(directory.glob("*.pkl")))
            size_mb = sum(f.stat().st_size for f in directory.glob("*.pkl")) / (
                1024 * 1024
            )
            return count, size_mb

        embeddings_count, embeddings_size = count_files(
            self.embedding_cache_dir)
        rag_count, rag_size = count_files(self.rag_cache_dir)
        agents_count, agents_size = count_files(self.agent_cache_dir)

        total_count = embeddings_count + rag_count + agents_count
        total_size = embeddings_size + rag_size + agents_size

        return {
            "embeddings": {"count": embeddings_count, "size_mb": round(embeddings_size, 2)},
            "rag": {"count": rag_count, "size_mb": round(rag_size, 2)},
            "agents": {"count": agents_count, "size_mb": round(agents_size, 2)},
            "total": {"count": total_count, "size_mb": round(total_size, 2)},
        }

    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear cache files.

        Args:
            cache_type: Type to clear ('embeddings', 'rag', 'agents') or None for all
        """
        dirs_to_clear = []

        if cache_type is None or cache_type == "embeddings":
            dirs_to_clear.append(self.embedding_cache_dir)
        if cache_type is None or cache_type == "rag":
            dirs_to_clear.append(self.rag_cache_dir)
        if cache_type is None or cache_type == "agents":
            dirs_to_clear.append(self.agent_cache_dir)

        for directory in dirs_to_clear:
            if directory.exists():
                for file in directory.glob("*.pkl"):
                    try:
                        file.unlink()
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete cache file {file}: {e}")

        logger.info(f"Cache cleared: {cache_type or 'all'}")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(cache_dir: str = ".cache", ttl_hours: int = 24) -> CacheManager:
    """
    Get or create global cache manager instance.

    Args:
        cache_dir: Root cache directory
        ttl_hours: Cache time-to-live in hours

    Returns:
        CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(cache_dir=cache_dir, ttl_hours=ttl_hours)
    return _cache_manager

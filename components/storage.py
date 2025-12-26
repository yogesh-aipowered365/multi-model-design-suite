"""
Storage service for analysis results.
Supports JSON file storage (dev), SQLite, and PostgreSQL backends.
"""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from components.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisRecord:
    """Analysis record for history tracking."""
    session_id: str
    timestamp: str
    platform: str
    creative_type: str
    analysis_mode: str
    enabled_agents: List[str]
    overall_score: float
    agent_scores: Dict[str, float]
    num_findings: int
    num_recommendations: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class StorageBackend:
    """Base storage backend interface."""

    def save_report(self, session_id: str, report_data: Dict) -> bool:
        """Save full report."""
        raise NotImplementedError

    def get_report(self, session_id: str) -> Optional[Dict]:
        """Get full report by session."""
        raise NotImplementedError

    def save_record(self, record: AnalysisRecord) -> bool:
        """Save analysis record for history."""
        raise NotImplementedError

    def get_history(self, limit: int = 20) -> List[AnalysisRecord]:
        """Get recent analysis history."""
        raise NotImplementedError


class JSONStorageBackend(StorageBackend):
    """JSON file storage backend (for development)."""

    def __init__(self, base_path: str = "data/reports"):
        """Initialize JSON storage.

        Args:
            base_path: Base directory for storing JSON reports
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.history_file = self.base_path / "history.jsonl"
        logger.info(f"JSON storage initialized at {self.base_path}")

    def save_report(self, session_id: str, report_data: Dict) -> bool:
        """Save full report as JSON."""
        try:
            session_dir = self.base_path / session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = session_dir / f"report_{timestamp}.json"

            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            logger.info(f"Report saved: {report_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return False

    def get_report(self, session_id: str) -> Optional[Dict]:
        """Get most recent report for session."""
        try:
            session_dir = self.base_path / session_id
            if not session_dir.exists():
                return None

            # Get most recent report
            reports = sorted(session_dir.glob("report_*.json"), reverse=True)
            if reports:
                with open(reports[0], 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to get report: {e}")

        return None

    def save_record(self, record: AnalysisRecord) -> bool:
        """Save analysis record to history."""
        try:
            with open(self.history_file, 'a') as f:
                f.write(json.dumps(record.to_dict()) + '\n')
            logger.info(f"Record saved for session {record.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save record: {e}")
            return False

    def get_history(self, limit: int = 20) -> List[AnalysisRecord]:
        """Get recent analysis history."""
        try:
            if not self.history_file.exists():
                return []

            records = []
            with open(self.history_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        records.append(AnalysisRecord(**data))

            # Return most recent records (reverse sort)
            return sorted(records, key=lambda r: r.timestamp, reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []


class SQLiteStorageBackend(StorageBackend):
    """SQLite storage backend."""

    def __init__(self, db_path: str = "data/reports.db"):
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"SQLite storage initialized at {self.db_path}")

    def _init_db(self):
        """Initialize database schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    session_id TEXT PRIMARY KEY,
                    report_json TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    platform TEXT,
                    creative_type TEXT
                )
            """)

            # History table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    platform TEXT,
                    creative_type TEXT,
                    analysis_mode TEXT,
                    enabled_agents TEXT,
                    overall_score REAL,
                    agent_scores TEXT,
                    num_findings INTEGER,
                    num_recommendations INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def save_report(self, session_id: str, report_data: Dict) -> bool:
        """Save full report to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            platform = report_data.get('platform', '')
            creative_type = report_data.get('creative_type', '')
            report_json = json.dumps(report_data, default=str)

            cursor.execute("""
                INSERT OR REPLACE INTO reports 
                (session_id, report_json, platform, creative_type)
                VALUES (?, ?, ?, ?)
            """, (session_id, report_json, platform, creative_type))

            conn.commit()
            conn.close()
            logger.info(f"Report saved to SQLite: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save report to SQLite: {e}")
            return False

    def get_report(self, session_id: str) -> Optional[Dict]:
        """Get report from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT report_json FROM reports WHERE session_id = ?", (session_id,))
            result = cursor.fetchone()
            conn.close()

            if result:
                return json.loads(result[0])
        except Exception as e:
            logger.error(f"Failed to get report from SQLite: {e}")

        return None

    def save_record(self, record: AnalysisRecord) -> bool:
        """Save analysis record to history."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO history
                (session_id, timestamp, platform, creative_type, analysis_mode,
                 enabled_agents, overall_score, agent_scores, num_findings, num_recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.session_id,
                record.timestamp,
                record.platform,
                record.creative_type,
                record.analysis_mode,
                json.dumps(record.enabled_agents),
                record.overall_score,
                json.dumps(record.agent_scores),
                record.num_findings,
                record.num_recommendations
            ))

            conn.commit()
            conn.close()
            logger.info(f"Record saved to SQLite: {record.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save record to SQLite: {e}")
            return False

    def get_history(self, limit: int = 20) -> List[AnalysisRecord]:
        """Get recent analysis history."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT session_id, timestamp, platform, creative_type, analysis_mode,
                       enabled_agents, overall_score, agent_scores, num_findings, num_recommendations
                FROM history
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            records = []
            for row in cursor.fetchall():
                record = AnalysisRecord(
                    session_id=row[0],
                    timestamp=row[1],
                    platform=row[2],
                    creative_type=row[3],
                    analysis_mode=row[4],
                    enabled_agents=json.loads(row[5]),
                    overall_score=row[6],
                    agent_scores=json.loads(row[7]),
                    num_findings=row[8],
                    num_recommendations=row[9]
                )
                records.append(record)

            conn.close()
            return records
        except Exception as e:
            logger.error(f"Failed to get history from SQLite: {e}")
            return []


class PostgresStorageBackend(StorageBackend):
    """PostgreSQL storage backend."""

    def __init__(self, connection_string: str):
        """Initialize PostgreSQL storage.

        Args:
            connection_string: PostgreSQL connection string
                Format: postgresql://user:password@host:port/database
        """
        self.connection_string = connection_string
        try:
            import psycopg2
            self.psycopg2 = psycopg2
            self._init_db()
            logger.info(f"PostgreSQL storage initialized")
        except ImportError:
            logger.error(
                "psycopg2 not installed. Install with: pip install psycopg2-binary")
            raise

    def _init_db(self):
        """Initialize database schema."""
        try:
            conn = self.psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            # Reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    session_id TEXT PRIMARY KEY,
                    report_json JSONB NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    platform TEXT,
                    creative_type TEXT
                )
            """)

            # History table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    platform TEXT,
                    creative_type TEXT,
                    analysis_mode TEXT,
                    enabled_agents JSONB,
                    overall_score FLOAT,
                    agent_scores JSONB,
                    num_findings INTEGER,
                    num_recommendations INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def save_report(self, session_id: str, report_data: Dict) -> bool:
        """Save full report to database."""
        try:
            conn = self.psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            platform = report_data.get('platform', '')
            creative_type = report_data.get('creative_type', '')
            report_json = json.dumps(report_data, default=str)

            cursor.execute("""
                INSERT INTO reports (session_id, report_json, platform, creative_type)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE
                SET report_json = EXCLUDED.report_json
            """, (session_id, report_json, platform, creative_type))

            conn.commit()
            conn.close()
            logger.info(f"Report saved to PostgreSQL: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save report to PostgreSQL: {e}")
            return False

    def get_report(self, session_id: str) -> Optional[Dict]:
        """Get report from database."""
        try:
            conn = self.psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT report_json FROM reports WHERE session_id = %s", (session_id,))
            result = cursor.fetchone()
            conn.close()

            if result:
                return json.loads(result[0])
        except Exception as e:
            logger.error(f"Failed to get report from PostgreSQL: {e}")

        return None

    def save_record(self, record: AnalysisRecord) -> bool:
        """Save analysis record to history."""
        try:
            conn = self.psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO history
                (session_id, timestamp, platform, creative_type, analysis_mode,
                 enabled_agents, overall_score, agent_scores, num_findings, num_recommendations)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                record.session_id,
                record.timestamp,
                record.platform,
                record.creative_type,
                record.analysis_mode,
                json.dumps(record.enabled_agents),
                record.overall_score,
                json.dumps(record.agent_scores),
                record.num_findings,
                record.num_recommendations
            ))

            conn.commit()
            conn.close()
            logger.info(f"Record saved to PostgreSQL: {record.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save record to PostgreSQL: {e}")
            return False

    def get_history(self, limit: int = 20) -> List[AnalysisRecord]:
        """Get recent analysis history."""
        try:
            conn = self.psycopg2.connect(self.connection_string)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT session_id, timestamp, platform, creative_type, analysis_mode,
                       enabled_agents, overall_score, agent_scores, num_findings, num_recommendations
                FROM history
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

            records = []
            for row in cursor.fetchall():
                record = AnalysisRecord(
                    session_id=row[0],
                    timestamp=row[1],
                    platform=row[2],
                    creative_type=row[3],
                    analysis_mode=row[4],
                    enabled_agents=json.loads(row[5]),
                    overall_score=row[6],
                    agent_scores=json.loads(row[7]),
                    num_findings=row[8],
                    num_recommendations=row[9]
                )
                records.append(record)

            conn.close()
            return records
        except Exception as e:
            logger.error(f"Failed to get history from PostgreSQL: {e}")
            return []


class StorageService:
    """Main storage service manager."""

    def __init__(self):
        """Initialize storage service based on environment configuration."""
        # Determine backend from environment
        backend_type = os.getenv("STORAGE_BACKEND", "json").lower()

        if backend_type == "sqlite":
            db_path = os.getenv("SQLITE_DB_PATH", "data/reports.db")
            self.backend = SQLiteStorageBackend(db_path)
        elif backend_type == "postgres":
            conn_string = os.getenv("DATABASE_URL")
            if not conn_string:
                logger.warning(
                    "DATABASE_URL not set. Falling back to JSON storage.")
                self.backend = JSONStorageBackend()
            else:
                self.backend = PostgresStorageBackend(conn_string)
        else:
            # Default to JSON
            base_path = os.getenv("STORAGE_PATH", "data/reports")
            self.backend = JSONStorageBackend(base_path)

        logger.info(
            f"Storage service initialized with {self.backend.__class__.__name__}")

    def save_analysis(self, session_id: str, sidebar_config: Dict, results: Dict) -> bool:
        """Save complete analysis results.

        Args:
            session_id: Session ID
            sidebar_config: Sidebar configuration
            results: Analysis results

        Returns:
            True if successful
        """
        # Save full report
        report_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "sidebar_config": sidebar_config,
            "results": results
        }

        if not self.backend.save_report(session_id, report_data):
            return False

        # Save history record
        agent_scores = results.get("agent_scores", {})
        findings = results.get("findings_summary", {})
        recommendations = results.get("top_recommendations", [])

        record = AnalysisRecord(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            platform=sidebar_config.get("platform", ""),
            creative_type=sidebar_config.get("creative_type", ""),
            analysis_mode=sidebar_config.get("analysis_mode", ""),
            enabled_agents=sidebar_config.get("enabled_agents", []),
            overall_score=results.get("overall_score", 0),
            agent_scores=agent_scores,
            num_findings=findings.get("total", 0),
            num_recommendations=len(recommendations)
        )

        return self.backend.save_record(record)

    def get_analysis(self, session_id: str) -> Optional[Dict]:
        """Get saved analysis.

        Args:
            session_id: Session ID

        Returns:
            Analysis data or None
        """
        return self.backend.get_report(session_id)

    def get_history(self, limit: int = 20) -> List[AnalysisRecord]:
        """Get analysis history.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of analysis records
        """
        return self.backend.get_history(limit)

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path


def _default_cache_path() -> Path:
    override = os.getenv("EMBX_CACHE_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "embx" / "cache.db"


class EmbeddingCache:
    def __init__(self, enabled: bool, path: Path | None = None) -> None:
        self.enabled = enabled
        self.path = path or _default_cache_path()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    vector_json TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def build_key(
        provider: str,
        model: str,
        dimensions: int | None,
        text: str,
    ) -> str:
        raw = f"{provider}|{model}|{dimensions or 0}|{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(
        self,
        provider: str,
        model: str,
        dimensions: int | None,
        text: str,
    ) -> list[float] | None:
        if not self.enabled:
            return None
        cache_key = self.build_key(provider, model, dimensions, text)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT vector_json FROM embeddings WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def set(
        self,
        provider: str,
        model: str,
        dimensions: int | None,
        text: str,
        vector: list[float],
    ) -> None:
        if not self.enabled:
            return
        cache_key = self.build_key(provider, model, dimensions, text)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings(cache_key, vector_json) VALUES (?, ?)",
                (cache_key, json.dumps(vector)),
            )

"""
Per-agent ChromaDB vector store for long-term memory.

Each agent gets its own named collection. Research packets, past decisions,
and learned preferences are stored as documents with metadata so agents can
do semantic search over their own history.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Optional

import chromadb
from chromadb.config import Settings

from hedge_fund.config import CHROMA_PERSIST_DIR


_client: Optional[chromadb.ClientAPI] = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _doc_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class AgentMemory:
    """Vector store façade for a single agent."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        client = _get_client()
        # collection name must be 3-63 chars, alphanum + hyphens
        safe_name = agent_id.replace("_", "-")[:63]
        self.collection = client.get_or_create_collection(
            name=safe_name,
            metadata={"hnsw:space": "cosine"},
        )

    def store(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Embed and store a text document. Returns the doc_id."""
        if doc_id is None:
            doc_id = _doc_id(text + datetime.utcnow().isoformat())

        meta = {"agent_id": self.agent_id, "stored_at": datetime.utcnow().isoformat()}
        if metadata:
            # ChromaDB metadata values must be str/int/float/bool
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = json.dumps(v)

        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
        return doc_id

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """Semantic search. Returns list of {text, metadata, distance}."""
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, self.collection.count() or 1),
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        output = []
        for i, doc in enumerate(results["documents"][0]):
            output.append(
                {
                    "text": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return output

    def store_research(self, packet_json: str, subject: str, ticker: Optional[str] = None) -> str:
        meta: dict[str, Any] = {"type": "research", "subject": subject}
        if ticker:
            meta["ticker"] = ticker
        return self.store(packet_json, metadata=meta)

    def store_decision(self, decision_text: str, outcome: Optional[str] = None) -> str:
        meta: dict[str, Any] = {"type": "decision"}
        if outcome:
            meta["outcome"] = outcome
        return self.store(decision_text, metadata=meta)

    def recall_recent_research(self, query: str, n: int = 3) -> list[dict]:
        return self.search(query, n_results=n, where={"type": "research"})

    def recall_past_decisions(self, query: str, n: int = 3) -> list[dict]:
        return self.search(query, n_results=n, where={"type": "decision"})

    def count(self) -> int:
        return self.collection.count()

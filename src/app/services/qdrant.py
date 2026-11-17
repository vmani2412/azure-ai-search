"""Qdrant vector database service for indexing local `docs/` files and vector search."""

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings

log = logging.getLogger(__name__)

VECTOR_DIM = 256
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "can",
    "detail",
    "details",
    "do",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "show",
    "tell",
    "the",
    "to",
    "what",
    "where",
    "which",
    "with",
    "you",
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _embed_text(text: str, dim: int = VECTOR_DIM) -> list[float]:
    """Compute a unit-normalized dense embedding vector from word tokens and char n-grams."""
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        tokens = re.findall(r"[a-z0-9]+", text.lower()) or ["empty"]

    features: list[tuple[str, float]] = []
    for tok in tokens:
        features.append((f"w:{tok}", 2.0))
        if tok.endswith("s") and len(tok) > 3:
            features.append((f"w:{tok[:-1]}", 1.5))
        padded = f"^{tok}$"
        for i in range(max(1, len(padded) - 2)):
            features.append((f"g:{padded[i : i + 3]}", 0.5))

    for feat, weight in features:
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "big") % dim
        sign = 1.0 if (digest[2] & 1) == 0 else -1.0
        vec[idx] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [round(v / norm, 6) for v in vec]
    return vec


def _format_record(source: str, item: Any) -> str:
    if isinstance(item, dict):
        lines = [f"{k}: {v}" for k, v in item.items() if v is not None and str(v).strip()]
        body = "\n".join(lines)
    else:
        body = str(item).strip()
    return f"[Source: {source}]\n{body}" if body else ""


def load_docs_from_folder(docs_dir: str | Path) -> list[dict[str, str]]:
    """Read all supported files (.json, .txt, .md, .csv) from the docs directory."""
    root = Path(docs_dir)
    if not root.exists() or not root.is_dir():
        return []

    records: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        rel_name = str(path.relative_to(root))
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            continue

        if path.suffix.lower() == ".json":
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw
            if isinstance(parsed, list):
                items: list[Any] = parsed
            elif isinstance(parsed, dict):
                list_vals = [v for v in parsed.values() if isinstance(v, list)]
                items = list_vals[0] if len(list_vals) == 1 else [parsed]
            else:
                items = [parsed]
            for item in items:
                formatted = _format_record(rel_name, item)
                if formatted:
                    records.append({"source": rel_name, "text": formatted})
        else:
            formatted = _format_record(rel_name, raw)
            if formatted:
                records.append({"source": rel_name, "text": formatted})

    return records


class QdrantService:
    """Manages document ingestion into Qdrant and vector search retrieval."""

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._s = settings
        self._http = http_client or httpx.AsyncClient(timeout=settings.llm_timeout_s)
        self._ingested = False

    @property
    def collection_name(self) -> str:
        return self._s.qdrant_collection

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._s.qdrant_api_key and self._s.qdrant_api_key.get_secret_value().strip():
            headers["api-key"] = self._s.qdrant_api_key.get_secret_value()
        return headers

    async def ingest_docs(self) -> int:
        """Read all files in `docs_dir` and upsert their vectors + payloads into Qdrant."""
        records = load_docs_from_folder(self._s.docs_dir)
        base_url = self._s.qdrant_url.rstrip("/")
        collection = self._s.qdrant_collection

        # Recreate collection cleanly so re-ingesting never fails with 409 Conflict
        await self._http.delete(
            f"{base_url}/collections/{collection}",
            headers=self._headers(),
        )
        create_resp = await self._http.put(
            f"{base_url}/collections/{collection}",
            headers=self._headers(),
            json={"vectors": {"size": VECTOR_DIM, "distance": "Cosine"}},
        )
        if create_resp.status_code != 409:
            create_resp.raise_for_status()

        if not records:
            self._ingested = True
            return 0

        points: list[dict[str, Any]] = []
        for idx, rec in enumerate(records, start=1):
            points.append(
                {
                    "id": idx,
                    "vector": _embed_text(rec["text"]),
                    "payload": {
                        "source": rec["source"],
                        "text": rec["text"],
                    },
                }
            )

        upsert_resp = await self._http.put(
            f"{base_url}/collections/{collection}/points",
            params={"wait": "true"},
            headers=self._headers(),
            json={"points": points},
        )
        upsert_resp.raise_for_status()
        self._ingested = True
        log.info(
            "qdrant_ingest_completed",
            extra={"ctx": {"collection": collection, "points": len(points)}},
        )
        return len(points)

    async def search(self, query: str) -> str:
        """Search Qdrant vector database for relevant documents from `docs/`."""
        if not self._ingested:
            await self.ingest_docs()

        base_url = self._s.qdrant_url.rstrip("/")
        collection = self._s.qdrant_collection
        query_vec = _embed_text(query)

        resp = await self._http.post(
            f"{base_url}/collections/{collection}/points/search",
            headers=self._headers(),
            json={
                "vector": query_vec,
                "limit": self._s.qdrant_top_k,
                "with_payload": True,
                "score_threshold": 0.15,
            },
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        hits = data.get("result")
        if not isinstance(hits, list):
            return ""

        query_tokens = set(_tokenize(query))
        chunks: list[str] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            payload = hit.get("payload")
            if not isinstance(payload, dict):
                continue
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            doc_tokens = set(_tokenize(text))
            if query_tokens and not (query_tokens & doc_tokens):
                continue
            chunks.append(text.strip())

        log.info(
            "qdrant_search_completed",
            extra={"ctx": {"collection": collection, "hits": len(chunks)}},
        )
        return "\n\n".join(chunks)

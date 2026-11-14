"""Azure AI Search client for retrieving grounded context indexed from Azure Blob Storage."""

import logging
from typing import Any

import httpx

from app.config import Settings

log = logging.getLogger(__name__)

_BODY_FIELDS = ("content", "chunk", "merged_content", "text")
_SOURCE_FIELDS = ("metadata_storage_name", "title", "filepath", "id")
_IGNORED_KEYS = {"AzureSearch_DocumentKey", *_SOURCE_FIELDS}


def _extract_doc_text(doc: dict[str, Any]) -> str:
    """Extract readable text and structured fields from an Azure AI Search document."""
    parts: list[str] = []
    for k, v in doc.items():
        if k.startswith("@search") or k.startswith("metadata_") or k in _IGNORED_KEYS:
            continue
        if k in _BODY_FIELDS and isinstance(v, str) and v.strip():
            parts.append(v.strip())
        elif isinstance(v, str | int | float | bool) and str(v).strip():
            parts.append(f"{k}: {v}")

    if not parts:
        return ""

    source = next(
        (str(doc[f]) for f in _SOURCE_FIELDS if isinstance(doc.get(f), str) and doc.get(f)),
        "document",
    )
    body = "\n".join(parts)
    return f"[Source: {source}]\n{body}"


class SearchClient:
    """Queries Azure AI Search (`learnaisearch24`) for documents matching the user's message."""

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._s = settings
        self._http = http_client or httpx.AsyncClient(timeout=settings.llm_timeout_s)

    @property
    def is_configured(self) -> bool:
        return bool(
            self._s.azure_search_endpoint
            and self._s.azure_search_api_key
            and self._s.azure_search_api_key.get_secret_value().strip()
        )

    async def search(self, query: str) -> str:
        """Return concatenated text chunks from Azure AI Search, or '' if nothing matches."""
        if (
            not self.is_configured
            or not self._s.azure_search_endpoint
            or not self._s.azure_search_api_key
        ):
            return ""

        endpoint = self._s.azure_search_endpoint.rstrip("/")
        url = f"{endpoint}/indexes/{self._s.azure_search_index}/docs/search"

        resp = await self._http.post(
            url,
            params={"api-version": "2024-07-01"},
            headers={
                "Content-Type": "application/json",
                "api-key": self._s.azure_search_api_key.get_secret_value(),
            },
            json={
                "search": query,
                "top": self._s.azure_search_top_k,
            },
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        raw_docs = data.get("value")
        if not isinstance(raw_docs, list):
            return ""

        chunks = [
            text for doc in raw_docs if isinstance(doc, dict) and (text := _extract_doc_text(doc))
        ]
        log.info(
            "azure_search_completed",
            extra={"ctx": {"index": self._s.azure_search_index, "hits": len(chunks)}},
        )
        return "\n\n".join(chunks)

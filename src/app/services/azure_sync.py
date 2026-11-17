"""Sync local `docs/` files to Azure Blob Storage, index into Azure AI Search, and verify."""

import base64
import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings

log = logging.getLogger(__name__)


class AzureSyncError(Exception):
    """Raised when Azure Storage or Azure AI Search sync fails."""


def _parse_conn_str(conn_str: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for segment in conn_str.split(";"):
        if "=" in segment:
            k, v = segment.split("=", 1)
            parts[k.strip()] = v.strip()
    return parts


def _sign_blob_request(
    method: str,
    account: str,
    account_key: str,
    resource_path: str,
    content_length: int,
    content_type: str,
    ms_date: str,
    extra_headers: dict[str, str] | None = None,
    query_params: dict[str, str] | None = None,
) -> str:
    """Create Azure Storage SharedKey authorization header (version 2021-08-06)."""
    headers = {"x-ms-date": ms_date, "x-ms-version": "2021-08-06"}
    if extra_headers:
        headers.update({k.lower(): v for k, v in extra_headers.items()})

    canon_headers = "".join(f"{k}:{headers[k]}\n" for k in sorted(headers))
    canon_resource = f"/{account}{resource_path}"
    if query_params:
        for k in sorted(query_params):
            canon_resource += f"\n{k.lower()}:{query_params[k]}"

    len_str = str(content_length) if content_length > 0 else ""
    string_to_sign = (
        f"{method}\n\n\n{len_str}\n\n{content_type}\n\n\n\n\n\n\n{canon_headers}{canon_resource}"
    )
    key_bytes = base64.b64decode(account_key)
    sig = base64.b64encode(
        hmac.new(key_bytes, string_to_sign.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")
    return f"SharedKey {account}:{sig}"


def _extract_items_for_azure_index(docs_dir: str | Path) -> list[dict[str, Any]]:
    """Convert all files in `docs/` into Azure AI Search index documents."""
    root = Path(docs_dir)
    if not root.exists() or not root.is_dir():
        return []

    search_docs: list[dict[str, Any]] = []
    idx = 0
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
        else:
            items = [raw]

        for item in items:
            idx += 1
            raw_key = f"{rel_name}:{idx}"
            doc_id = base64.urlsafe_b64encode(raw_key.encode("utf-8")).decode("utf-8").rstrip("=")

            if isinstance(item, dict):
                prod_id = str(item.get("product_id") or item.get("id") or idx)
                name = str(item.get("name") or item.get("title") or rel_name)
                category = str(item.get("category") or "general")
                raw_price = item.get("price", 0)
                try:
                    price_int = int(round(float(raw_price)))
                except (TypeError, ValueError):
                    price_int = 0

                extra_lines: list[str] = []
                if isinstance(item.get("description"), str):
                    extra_lines.append(item["description"])
                for k, v in item.items():
                    if k in (
                        "id",
                        "product_id",
                        "name",
                        "title",
                        "category",
                        "description",
                        "images",
                        "thumbnail",
                        "meta",
                    ):
                        continue
                    if isinstance(v, str | int | float | bool) and str(v).strip():
                        extra_lines.append(f"{k}: {v}")
                description = " | ".join(extra_lines)
            else:
                prod_id = str(idx)
                name = rel_name
                category = "document"
                price_int = 0
                description = str(item)

            search_docs.append(
                {
                    "@search.action": "mergeOrUpload",
                    "id": doc_id,
                    "title": rel_name,
                    "product_id": prod_id,
                    "name": name,
                    "category": category,
                    "price": price_int,
                    "description": description,
                }
            )
    return search_docs


def _read_raw_files(docs_dir: str | Path) -> list[tuple[str, bytes, str]]:
    """Read raw files from `docs/` as (relative_name, content_bytes, content_type)."""
    root = Path(docs_dir)
    if not root.exists() or not root.is_dir():
        return []
    files: list[tuple[str, bytes, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        rel_name = str(path.relative_to(root)).replace("\\", "/")
        ctype = (
            "application/json" if path.suffix.lower() == ".json" else "text/plain; charset=utf-8"
        )
        files.append((rel_name, path.read_bytes(), ctype))
    return files


class AzureSyncService:
    """Uploads `docs/` to Azure Blob Storage, indexes into Azure AI Search, and verifies."""

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._s = settings
        self._http = http_client or httpx.AsyncClient(timeout=settings.llm_timeout_s)

    async def upload_docs_to_blob(self) -> list[str]:
        """Upload all files in `docs/` to Azure Blob Storage if connection string is configured."""
        conn_secret = self._s.azure_storage_connection_string
        if not conn_secret or not conn_secret.get_secret_value().strip():
            return []

        conn = _parse_conn_str(conn_secret.get_secret_value())
        account = conn.get("AccountName", "")
        key = conn.get("AccountKey", "")
        suffix = conn.get("EndpointSuffix", "core.windows.net")
        if not account or not key:
            raise AzureSyncError(
                "Invalid AZURE_STORAGE_CONNECTION_STRING (missing AccountName/AccountKey)."
            )

        container = self._s.azure_storage_container
        raw_files = _read_raw_files(self._s.docs_dir)
        if not raw_files:
            return []

        # 1. Ensure container exists
        ms_date = format_datetime(datetime.now(UTC), usegmt=True)
        container_path = f"/{container}"
        auth = _sign_blob_request(
            "PUT",
            account,
            key,
            container_path,
            0,
            "",
            ms_date,
            query_params={"restype": "container"},
        )
        c_resp = await self._http.put(
            f"https://{account}.blob.{suffix}{container_path}",
            params={"restype": "container"},
            headers={
                "x-ms-date": ms_date,
                "x-ms-version": "2021-08-06",
                "Authorization": auth,
            },
        )
        if c_resp.status_code not in (201, 409):
            raise AzureSyncError(f"Blob container error ({c_resp.status_code}): {c_resp.text}")

        # 2. Upload each file in docs/
        uploaded: list[str] = []
        for rel_name, content, ctype in raw_files:
            ms_date = format_datetime(datetime.now(UTC), usegmt=True)
            blob_path = f"/{container}/{rel_name}"
            extra = {"x-ms-blob-type": "BlockBlob"}
            auth = _sign_blob_request(
                "PUT",
                account,
                key,
                blob_path,
                len(content),
                ctype,
                ms_date,
                extra_headers=extra,
            )
            b_resp = await self._http.put(
                f"https://{account}.blob.{suffix}{blob_path}",
                content=content,
                headers={
                    "x-ms-date": ms_date,
                    "x-ms-version": "2021-08-06",
                    "x-ms-blob-type": "BlockBlob",
                    "Content-Type": ctype,
                    "Content-Length": str(len(content)),
                    "Authorization": auth,
                },
            )
            if b_resp.status_code not in (201, 200):
                raise AzureSyncError(
                    f"Blob upload error for {rel_name} ({b_resp.status_code}): {b_resp.text}"
                )
            uploaded.append(rel_name)

        return uploaded

    async def sync_and_verify(self) -> dict[str, Any]:
        """Sync `docs/` to Blob Storage, index into Azure AI Search, and verify."""
        if not self._s.azure_search_endpoint or not self._s.azure_search_api_key:
            raise AzureSyncError(
                "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY must be configured."
            )

        blobs_uploaded = await self.upload_docs_to_blob()

        endpoint = self._s.azure_search_endpoint.rstrip("/")
        index_name = self._s.azure_search_index
        api_key = self._s.azure_search_api_key.get_secret_value()
        headers = {"Content-Type": "application/json", "api-key": api_key}

        # 1. Ensure the index exists (create if 404)
        idx_check = await self._http.get(
            f"{endpoint}/indexes/{index_name}",
            params={"api-version": "2024-07-01"},
            headers=headers,
        )
        if idx_check.status_code in (401, 403):
            raise AzureSyncError(
                "AZURE_SEARCH_API_KEY is currently a Query key. "
                "Set AZURE_SEARCH_API_KEY to your Primary Admin Key "
                "(Azure Portal -> learnaisearch24 -> Settings -> Keys -> Primary admin key)."
            )
        if idx_check.status_code == 404:
            schema = {
                "name": index_name,
                "fields": [
                    {"name": "id", "type": "Edm.String", "key": True, "searchable": False},
                    {
                        "name": "title",
                        "type": "Edm.String",
                        "searchable": True,
                        "retrievable": True,
                    },
                    {
                        "name": "product_id",
                        "type": "Edm.String",
                        "searchable": True,
                        "retrievable": True,
                    },
                    {"name": "name", "type": "Edm.String", "searchable": True, "retrievable": True},
                    {
                        "name": "category",
                        "type": "Edm.String",
                        "searchable": True,
                        "retrievable": True,
                    },
                    {"name": "price", "type": "Edm.Int64", "filterable": True, "retrievable": True},
                    {
                        "name": "description",
                        "type": "Edm.String",
                        "searchable": True,
                        "retrievable": True,
                    },
                ],
            }
            create_idx = await self._http.put(
                f"{endpoint}/indexes/{index_name}",
                params={"api-version": "2024-07-01"},
                headers=headers,
                json=schema,
            )
            if create_idx.status_code not in (200, 201, 204):
                raise AzureSyncError(
                    f"Failed to create index ({create_idx.status_code}): {create_idx.text}"
                )

        # 2. Push all documents from docs/ into the Azure AI Search index
        docs_to_index = _extract_items_for_azure_index(self._s.docs_dir)
        if docs_to_index:
            push_resp = await self._http.post(
                f"{endpoint}/indexes/{index_name}/docs/index",
                params={"api-version": "2024-07-01"},
                headers=headers,
                json={"value": docs_to_index},
            )
            if push_resp.status_code == 403:
                raise AzureSyncError(
                    "AZURE_SEARCH_API_KEY is a Query key and cannot write to the index. "
                    "Replace AZURE_SEARCH_API_KEY in .env with your Primary Admin Key."
                )
            if push_resp.status_code not in (200, 207):
                raise AzureSyncError(f"Indexing failed ({push_resp.status_code}): {push_resp.text}")

        # 3. Verify via search query count
        verify_resp = await self._http.post(
            f"{endpoint}/indexes/{index_name}/docs/search",
            params={"api-version": "2024-07-01"},
            headers=headers,
            json={"search": "*", "count": True, "top": 5},
        )
        verify_resp.raise_for_status()
        verify_data: dict[str, Any] = verify_resp.json()
        total_hits = verify_data.get("@search.count")
        if not isinstance(total_hits, int):
            vals = verify_data.get("value")
            total_hits = len(vals) if isinstance(vals, list) else 0

        log.info(
            "azure_sync_completed",
            extra={
                "ctx": {
                    "blobs": len(blobs_uploaded),
                    "indexed": len(docs_to_index),
                    "verified": total_hits,
                }
            },
        )
        return {
            "status": "ok",
            "storage_container": self._s.azure_storage_container,
            "blobs_uploaded": blobs_uploaded,
            "search_index": index_name,
            "documents_indexed": len(docs_to_index),
            "verified_search_hits": total_hits,
        }

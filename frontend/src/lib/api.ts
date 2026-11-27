export type ChatMode = "azure" | "qdrant";

export interface ChatResponse {
  answer: string;
  prompt_tokens: number;
  completion_tokens: number;
  retries: number;
  requestId?: string;
}

export interface IngestResponse {
  status: string;
  collection: string;
  points_indexed: number;
}

export interface AzureSyncResponse {
  status: string;
  storage_container: string;
  blobs_uploaded: string[];
  search_index: string;
  documents_indexed: number;
  verified_search_hits: number;
}

const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
).replace(/\/+$/, "");

export function getApiBaseUrl(): string {
  return API_BASE;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}

export async function sendChatMessage(
  message: string,
  mode: ChatMode,
  maxTokens: number
): Promise<ChatResponse> {
  const endpoint = mode === "qdrant" ? "/qdrant/chat" : "/chat";
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, max_tokens: maxTokens }),
  });

  const requestId = res.headers.get("X-Request-ID") ?? undefined;

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const errBody = await res.json();
      if (errBody?.detail) {
        detail = typeof errBody.detail === "string"
          ? errBody.detail
          : JSON.stringify(errBody.detail);
      }
    } catch {
      // keep fallback detail
    }
    throw new Error(detail);
  }

  const data: ChatResponse = await res.json();
  return { ...data, requestId };
}

export async function triggerQdrantIngest(): Promise<IngestResponse> {
  const res = await fetch(`${API_BASE}/qdrant/ingest`, { method: "POST" });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Qdrant ingest failed (${res.status})`);
  }
  return res.json();
}

export async function triggerAzureSync(): Promise<AzureSyncResponse> {
  const res = await fetch(`${API_BASE}/azure/sync`, { method: "POST" });
  if (!res.ok) {
    let detail = `Azure sync failed (${res.status})`;
    try {
      const errBody = await res.json();
      if (errBody?.detail) detail = String(errBody.detail);
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return res.json();
}

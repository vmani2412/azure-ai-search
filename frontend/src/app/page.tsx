"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import {
  ChatMode,
  checkHealth,
  getApiBaseUrl,
  sendChatMessage,
  triggerAzureSync,
  triggerQdrantIngest,
} from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  mode?: ChatMode;
  promptTokens?: number;
  completionTokens?: number;
  retries?: number;
  requestId?: string;
  isError?: boolean;
}

const STARTER_PROMPTS = [
  "Nike Running Shoes detail?",
  "What is the price of Gaming Laptop?",
  "Chanel Coco Noir Eau De price and warranty?",
  "What is the price of iPhone 16?",
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<ChatMode>("azure");
  const [maxTokens, setMaxTokens] = useState<number>(1000);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState<string | null>(null);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [toast, setToast] = useState<{ type: "info" | "error"; text: string } | null>(null);

  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    checkHealth().then(setHealthy);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend(promptText?: string) {
    const text = (promptText ?? input).trim();
    if (!text || loading) return;

    const userMsg: Message = {
      id: `u-${Date.now()}`,
      role: "user",
      content: text,
      mode,
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!promptText) setInput("");
    setLoading(true);

    try {
      const res = await sendChatMessage(text, mode, maxTokens);
      const botMsg: Message = {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: res.answer || "(Empty response)",
        mode,
        promptTokens: res.prompt_tokens,
        completionTokens: res.completion_tokens,
        retries: res.retries,
        requestId: res.requestId,
      };
      setMessages((prev) => [...prev, botMsg]);
      setHealthy(true);
    } catch (err) {
      const errText = err instanceof Error ? err.message : "Failed to reach API";
      setMessages((prev) => [
        ...prev,
        {
          id: `e-${Date.now()}`,
          role: "assistant",
          content: `Error: ${errText}`,
          mode,
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void handleSend();
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  }

  async function handleQdrantIngest() {
    setSyncing("qdrant");
    setToast(null);
    try {
      const res = await triggerQdrantIngest();
      setToast({
        type: "info",
        text: `Qdrant indexed ${res.points_indexed} documents into "${res.collection}".`,
      });
    } catch (err) {
      setToast({
        type: "error",
        text: err instanceof Error ? err.message : "Qdrant ingest failed",
      });
    } finally {
      setSyncing(null);
    }
  }

  async function handleAzureSync() {
    setSyncing("azure");
    setToast(null);
    try {
      const res = await triggerAzureSync();
      setToast({
        type: "info",
        text: `Azure sync ok: ${res.documents_indexed} indexed (${res.verified_search_hits} verified in "${res.search_index}").`,
      });
    } catch (err) {
      setToast({
        type: "error",
        text: err instanceof Error ? err.message : "Azure sync failed",
      });
    } finally {
      setSyncing(null);
    }
  }

  return (
    <main className="app-shell">
      <header className="header">
        <div className="brand">
          <div className="brand-title">
            <span
              className={`status-dot ${
                healthy === true ? "online" : healthy === false ? "offline" : ""
              }`}
              title={
                healthy === true
                  ? "Backend Connected"
                  : healthy === false
                    ? "Backend Offline"
                    : "Checking..."
              }
            />
            <span>Azure AI RAG Chat</span>
          </div>
          <span className="brand-sub">API: {getApiBaseUrl()}</span>
        </div>

        <div className="controls">
          <div className="mode-switch" role="group" aria-label="Search backend">
            <button
              type="button"
              className={`mode-btn ${mode === "azure" ? "active-azure" : ""}`}
              onClick={() => setMode("azure")}
            >
              Azure AI Search
            </button>
            <button
              type="button"
              className={`mode-btn ${mode === "qdrant" ? "active-qdrant" : ""}`}
              onClick={() => setMode("qdrant")}
            >
              Qdrant Vector DB
            </button>
          </div>

          <button
            type="button"
            className="action-btn"
            disabled={syncing !== null}
            onClick={handleQdrantIngest}
          >
            {syncing === "qdrant" ? "Ingesting..." : "Ingest Qdrant"}
          </button>

          <button
            type="button"
            className="action-btn"
            disabled={syncing !== null}
            onClick={handleAzureSync}
          >
            {syncing === "azure" ? "Syncing..." : "Sync Azure"}
          </button>
        </div>
      </header>

      {toast && (
        <div className={`toast ${toast.type}`}>
          <span>{toast.text}</span>
          <button
            type="button"
            className="toast-close"
            onClick={() => setToast(null)}
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      )}

      <section className="chat-window">
        {messages.length === 0 ? (
          <div className="empty-state">
            <h2>Ask your grounded RAG assistant</h2>
            <p>
              Currently querying{" "}
              <strong>
                {mode === "azure" ? "Azure AI Search (/chat)" : "Qdrant Vector DB (/qdrant/chat)"}
              </strong>
              . Questions outside your indexed documents will return{" "}
              <code>Requested data is not available.</code>
            </p>
            <div className="starter-grid">
              {STARTER_PROMPTS.map((q) => (
                <button
                  key={q}
                  type="button"
                  className="starter-chip"
                  onClick={() => void handleSend(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => (
            <div
              key={m.id}
              className={`msg-row ${m.role} ${m.isError ? "error" : ""}`}
            >
              <div className="msg-bubble">{m.content}</div>
              {m.role === "assistant" && !m.isError && (
                <div className="msg-meta">
                  <span className="meta-pill">
                    {m.mode === "qdrant" ? "Qdrant" : "Azure Search"}
                  </span>
                  <span className="meta-pill">
                    Tokens: {m.promptTokens ?? 0} in / {m.completionTokens ?? 0} out
                  </span>
                  {typeof m.retries === "number" && m.retries > 0 && (
                    <span className="meta-pill">Retries: {m.retries}</span>
                  )}
                  {m.requestId && (
                    <span className="meta-pill">Req: {m.requestId.slice(0, 8)}</span>
                  )}
                </div>
              )}
            </div>
          ))
        )}

        {loading && (
          <div className="msg-row assistant">
            <div className="msg-bubble">Thinking...</div>
          </div>
        )}
        <div ref={bottomRef} />
      </section>

      <form className="composer" onSubmit={onSubmit}>
        <div className="composer-row">
          <textarea
            className="composer-input"
            rows={1}
            placeholder={
              mode === "azure"
                ? "Ask a question using Azure AI Search..."
                : "Ask a question using Qdrant Vector DB..."
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={loading}
          />
          <button
            type="submit"
            className="send-btn"
            disabled={loading || !input.trim()}
          >
            {loading ? "..." : "Send"}
          </button>
        </div>

        <div className="composer-footer">
          <span>Enter to send · Shift+Enter for new line</span>
          <div className="token-control">
            <label htmlFor="max-tokens">max_tokens:</label>
            <select
              id="max-tokens"
              className="token-select"
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
            >
              <option value={300}>300</option>
              <option value={1000}>1000</option>
              <option value={2000}>2000</option>
              <option value={4096}>4096</option>
            </select>
          </div>
        </div>
      </form>
    </main>
  );
}

"""Find where Azure OpenAI starts returning 429s, with and without retry.

Usage (from repo root, with .env filled in):
  python scripts/loadtest_429.py --no-retry --concurrency 5 10 20 40 --requests 60
  python scripts/loadtest_429.py --retries 5 --concurrency 5 10 20 40 --requests 60

Compare the two runs: success rate, 429 count, p50/p95 latency, retries used.
Record the results in LAB_NOTES.md (deployment type, quota, region, date).
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.config import get_settings  # noqa: E402
from app.services.llm import (  # noqa: E402
    ContentFilterError,
    LLMClient,
    RateLimitExhaustedError,
)

PROMPT = "Write two sentences about the Golden Gate Bridge."


async def run_level(client: LLMClient, concurrency: int, total: int) -> dict[str, float]:
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []
    stats = {"ok": 0, "rate_limited": 0, "filtered": 0, "error": 0, "retries": 0}

    async def one() -> None:
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.chat(PROMPT, max_tokens=60)
                stats["ok"] += 1
                stats["retries"] += r.retries
            except RateLimitExhaustedError:
                stats["rate_limited"] += 1
            except ContentFilterError:
                stats["filtered"] += 1
            except Exception as exc:  # lab script: report anything unexpected
                stats["error"] += 1
                print(f"  error: {type(exc).__name__}", file=sys.stderr)
            lat.append(time.perf_counter() - t0)

    t_start = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(total)))
    wall = time.perf_counter() - t_start
    lat.sort()
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    return {
        **stats,
        "p50": statistics.median(lat),
        "p95": p95,
        "wall": wall,
        "rpm": total / wall * 60,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--requests", type=int, default=60, help="requests per level")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--no-retry", action="store_true")
    a = ap.parse_args()

    settings = get_settings()
    settings.llm_max_retries = 0 if a.no_retry else a.retries
    client = LLMClient(settings)

    mode = "NO RETRY" if a.no_retry else f"retries={a.retries}"
    print(f"Deployment: {settings.azure_openai_deployment} | mode: {mode}")
    print(
        f"{'conc':>5} {'ok':>5} {'429':>5} {'filt':>5} {'err':>4} {'retries':>8} "
        f"{'p50 s':>7} {'p95 s':>7} {'req/min':>8}"
    )
    for c in a.concurrency:
        r = await run_level(client, c, a.requests)
        print(
            f"{c:>5} {r['ok']:>5.0f} {r['rate_limited']:>5.0f} {r['filtered']:>5.0f} "
            f"{r['error']:>4.0f} {r['retries']:>8.0f} {r['p50']:>7.2f} {r['p95']:>7.2f} "
            f"{r['rpm']:>8.0f}"
        )


if __name__ == "__main__":
    asyncio.run(main())

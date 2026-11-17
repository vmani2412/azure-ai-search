"""Git / CI-CD MCP Server using FastMCP to inspect commits, diffs, and deployments."""

import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import httpx

F = TypeVar("F", bound=Callable[..., Any])


class _MCPServerProtocol(Protocol):
    def tool(self, name: str | None = None, description: str | None = None) -> Callable[[F], F]: ...
    def run(self, *args: Any, **kwargs: Any) -> None: ...


class _FallbackFastMCP:
    """Lightweight fallback when `fastmcp` package is not installed in the current environment."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: dict[str, Callable[..., Any]] = {}

    def tool(self, name: str | None = None, description: str | None = None) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            tool_name = name or fn.__name__
            self.tools[tool_name] = fn
            return fn

        return decorator

    def run(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Install `fastmcp` (`pip install fastmcp`) to run the MCP server.")


def _create_mcp_server(name: str) -> _MCPServerProtocol:
    try:
        mod_name = "fastmcp"
        fastmcp_mod = __import__(mod_name)
        cls = getattr(fastmcp_mod, "FastMCP", None)
        if cls is not None:
            return cast(_MCPServerProtocol, cls(name))
    except ImportError:
        pass
    return _FallbackFastMCP(name)


mcp: _MCPServerProtocol = _create_mcp_server("git-cicd-inspector")

DEFAULT_GITHUB_REPO = "vmani2412/azure-ai-search"


def _run_git(args: list[str], cwd: str | Path | None = None) -> str:
    """Execute a git command and return stdout, or raise RuntimeError if git fails."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return proc.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        raise RuntimeError(f"Git command failed ({' '.join(args)}): {exc}") from exc


def _detect_github_repo(cwd: str | Path | None = None) -> str:
    """Parse `owner/repo` from `git remote get-url origin` or return default."""
    try:
        remote_url = _run_git(["remote", "get-url", "origin"], cwd=cwd)
        match = re.search(r"github\.com[:/]([^/]+/[^/.]+?)(?:\.git)?$", remote_url)
        if match:
            return match.group(1)
    except RuntimeError:
        pass
    return DEFAULT_GITHUB_REPO


def get_recent_commits(
    limit: int = 10,
    branch: str = "HEAD",
    repo_path: str | None = None,
) -> list[dict[str, str]]:
    """Return the most recent commits from the repository."""
    safe_limit = max(1, min(limit, 50))
    fmt = "%H%x1f%h%x1f%an%x1f%ae%x1f%aI%x1f%s"
    try:
        raw = _run_git(
            ["log", branch, f"-n{safe_limit}", f"--pretty=format:{fmt}"],
            cwd=repo_path,
        )
    except RuntimeError:
        return []

    commits: list[dict[str, str]] = []
    for line in raw.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 6:
            commits.append(
                {
                    "sha": parts[0],
                    "short_sha": parts[1],
                    "author": parts[2],
                    "email": parts[3],
                    "timestamp": parts[4],
                    "message": parts[5],
                }
            )
    return commits


def get_commit_diff(
    target: str = "HEAD",
    stat_only: bool = False,
    repo_path: str | None = None,
) -> dict[str, Any]:
    """Return the git diff and changed files summary for a given commit or ref."""
    try:
        stat_output = _run_git(["show", "--stat", "--oneline", target], cwd=repo_path)
        if stat_only:
            return {
                "target": target,
                "stat": stat_output,
                "diff": "",
            }
        diff_output = _run_git(["show", "--format=", target], cwd=repo_path)
        return {
            "target": target,
            "stat": stat_output,
            "diff": diff_output[:15000],
        }
    except RuntimeError as exc:
        return {
            "target": target,
            "error": str(exc),
            "stat": "",
            "diff": "",
        }


async def get_deployment_timestamps(
    limit: int = 5,
    repo_path: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Return recent CI/CD pipeline runs and deployment commit timestamps."""
    safe_limit = max(1, min(limit, 20))
    repo_slug = _detect_github_repo(repo_path)

    # 1. Local Git deployment branch (`main` / `origin/main`) commit timestamps
    git_deployments = get_recent_commits(limit=safe_limit, branch="HEAD", repo_path=repo_path)

    # 2. GitHub Actions CI/CD workflow runs (if reachable)
    workflow_runs: list[dict[str, Any]] = []
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=10.0)
    try:
        resp = await client.get(
            f"https://api.github.com/repos/{repo_slug}/actions/runs",
            params={"per_page": safe_limit},
            headers={"Accept": "application/vnd.github+json"},
        )
        if resp.status_code == 200:
            data: dict[str, Any] = resp.json()
            raw_runs = data.get("workflow_runs")
            if isinstance(raw_runs, list):
                for run in raw_runs:
                    if not isinstance(run, dict):
                        continue
                    workflow_runs.append(
                        {
                            "id": run.get("id"),
                            "name": run.get("name"),
                            "head_branch": run.get("head_branch"),
                            "head_sha": str(run.get("head_sha") or "")[:7],
                            "status": run.get("status"),
                            "conclusion": run.get("conclusion"),
                            "started_at": run.get("run_started_at") or run.get("created_at"),
                            "updated_at": run.get("updated_at"),
                            "url": run.get("html_url"),
                        }
                    )
    except httpx.HTTPError:
        pass
    finally:
        if owns_client:
            await client.aclose()

    latest_deployed_at = (
        workflow_runs[0].get("updated_at")
        if workflow_runs
        else (git_deployments[0]["timestamp"] if git_deployments else None)
    )

    return {
        "repository": repo_slug,
        "latest_deployment_timestamp": latest_deployed_at,
        "ci_cd_workflow_runs": workflow_runs,
        "recent_branch_commits": git_deployments,
    }


@mcp.tool(
    name="get_recent_commits",
    description="Inspect recent Git commits with hash, author, timestamp, and message.",
)
def _mcp_get_recent_commits(limit: int = 10, branch: str = "HEAD") -> list[dict[str, str]]:
    return get_recent_commits(limit=limit, branch=branch)


@mcp.tool(
    name="get_commit_diff",
    description="Inspect git diff for a commit, commit range (e.g. HEAD~1..HEAD), or working tree.",
)
def _mcp_get_commit_diff(target: str = "HEAD", stat_only: bool = False) -> dict[str, Any]:
    return get_commit_diff(target=target, stat_only=stat_only)


@mcp.tool(
    name="get_deployment_timestamps",
    description="Inspect CI/CD deployment timestamps from GitHub Actions runs and Git refs.",
)
async def _mcp_get_deployment_timestamps(limit: int = 5) -> dict[str, Any]:
    return await get_deployment_timestamps(limit=limit)


if __name__ == "__main__":  # pragma: no cover
    import os

    transport = os.getenv("MCP_TRANSPORT", "http")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", "8001"))
        mcp.run(transport="http", host=host, port=port)

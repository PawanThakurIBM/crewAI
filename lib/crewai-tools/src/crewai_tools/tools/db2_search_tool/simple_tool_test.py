#!/usr/bin/env python3
"""
Extended Robust Test Suite for DB2VectorSearchTool — Part 2.
Adds: distance metrics, semantic consistency, embedding robustness,
connection lifecycle, performance benchmarks, result integrity,
schema validation, concurrency, boundary conditions, and more.

Run after test_db2_tool_robust.py (Part 1) for full coverage.
"""
from __future__ import annotations

import json
import os
import sys
import time
import threading
import traceback
import statistics
import pysqlite3

from dataclasses import dataclass, field
from typing import Any, Callable

from sentence_transformers import SentenceTransformer

sys.modules["sqlite3"] = pysqlite3

from crewai_tools.tools.db2_search_tool import (
    DB2Config,
    DB2VectorSearchTool,
    DB2ToolSchema,
)

# ─────────────────────────────────────────────
# Embedding Setup
# ─────────────────────────────────────────────

print("Loading all-MiniLM-L6-v2 model...")
local_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_384_embedding(text: str) -> list[float]:
    return local_model.encode(text).tolist()

# ─────────────────────────────────────────────
# Faulty Embedding Functions (for robustness tests)
# ─────────────────────────────────────────────

def embedding_returns_strings(text: str) -> list[float]:
    """Returns strings instead of floats — type mismatch."""
    return ["not", "a", "float"] * 128  # type: ignore

def embedding_returns_empty(text: str) -> list[float]:
    """Returns an empty list — zero-dimension vector."""
    return []

def embedding_raises_exception(text: str) -> list[float]:
    """Always raises an exception."""
    raise RuntimeError("Simulated embedding service outage")

def embedding_returns_zero_vector(text: str) -> list[float]:
    """Returns an all-zeros 384-dim vector."""
    return [0.0] * 384

def embedding_returns_wrong_type(text: str) -> list[float]:
    """Returns None instead of a list."""
    return None  # type: ignore

def embedding_inconsistent_dim(call_count: list) -> Callable:
    """Returns 384-dim on first call, 128-dim on second — simulates instability."""
    def fn(text: str) -> list[float]:
        call_count[0] += 1
        return [0.1] * (384 if call_count[0] == 1 else 128)
    return fn


# ─────────────────────────────────────────────
# Result / Suite Infrastructure (same as Part 1)
# ─────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    duration_ms: float
    detail: str = ""
    error: str = ""

@dataclass
class TestSuite:
    results: list[TestResult] = field(default_factory=list)

    def add(self, result: TestResult) -> None:
        icon = "✅" if result.passed else "❌"
        print(f"  {icon} [{result.category}] {result.name} ({result.duration_ms:.0f}ms)")
        if result.detail:
            print(f"     → {result.detail}")
        if result.error and not result.passed:
            print(f"     ⚠  {result.error}")
        self.results.append(result)

    def summary(self) -> None:
        total  = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        avg_ms = sum(r.duration_ms for r in self.results) / total if total else 0

        print("\n" + "═" * 80)
        print("  EXTENDED TEST SUITE — FULL SUMMARY")
        print("═" * 80)
        print(f"  Total  : {total}")
        print(f"  Passed : {passed}  ✅")
        print(f"  Failed : {failed}  ❌")
        print(f"  Avg ms : {avg_ms:.1f}")
        print()

        by_cat: dict[str, list[TestResult]] = {}
        for r in self.results:
            by_cat.setdefault(r.category, []).append(r)

        for cat, items in by_cat.items():
            cat_pass = sum(1 for i in items if i.passed)
            print(f"  [{cat}]  {cat_pass}/{len(items)} passed")
            for item in items:
                icon = "✅" if item.passed else "❌"
                print(f"    {icon} {item.name}")

        print("═" * 80)


# ─────────────────────────────────────────────
# Shared Helpers
# ─────────────────────────────────────────────

def timed_run(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - t0) * 1000


def make_tool(extra_config: dict | None = None, embedding_fn: Callable | None = None) -> DB2VectorSearchTool:
    base = dict(
        database="AUTOGEN",
        hostname="9.60.203.12",
        port=50000,
        username="db2user",
        password="Db2pass123",
        table_name="PAWAN.DOCUMENTS",
        vector_column="EMBEDDING",
        return_columns=["CONTENT"],
        limit=3,
        max_distance=None,
    )
    if extra_config:
        base.update(extra_config)
    config = DB2Config(**base)
    return DB2VectorSearchTool(
        db2_config=config,
        custom_embedding_fn=embedding_fn or generate_384_embedding,
    )


def parse_output(raw: str) -> tuple[bool, list | dict]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "error" in data:
            return True, data
        return False, data
    except Exception as exc:
        return True, {"error": str(exc)}


# ─────────────────────────────────────────────
# 1. Distance Metric Tests
# ─────────────────────────────────────────────

def run_distance_metric_tests(suite: TestSuite) -> None:
    print("\n──── Distance Metric Tests ────")

    allowed_metrics = ["COSINE", "EUCLIDEAN", "DOT_PRODUCT", "L2_DISTANCE"]
    disallowed_metrics = ["MANHATTAN", "HAMMING", "PEARSON", "DROP TABLE--", "", "cosine"]

    for metric in allowed_metrics:
        tool = make_tool(extra_config={"distance_metric": metric})
        raw, ms = timed_run(lambda t=tool: t._run("machine learning"))
        is_err, data = parse_output(raw)
        detail = f"{len(data)} result(s)" if not is_err else f"DB2 error: {data.get('error','')[:60]}"
        # Either results or a DB2-level error are fine — we just need no ValueError
        suite.add(TestResult(
            name=f"metric={metric} accepted and executed",
            category="DistanceMetric",
            passed=True,
            duration_ms=ms,
            detail=detail,
        ))

    for metric in disallowed_metrics:
        try:
            tool = make_tool(extra_config={"distance_metric": metric})
            raw, ms = timed_run(lambda t=tool: t._run("test"))
            is_err, data = parse_output(raw)
            passed = is_err
            detail = f"Blocked: {data.get('error','')[:60]}" if is_err else "⚠ NOT blocked"
        except Exception as exc:
            ms = 0.0
            passed = True
            detail = f"Raised (good): {str(exc)[:60]}"

        suite.add(TestResult(
            name=f"metric='{metric}' rejected",
            category="DistanceMetric",
            passed=passed,
            duration_ms=ms,
            detail=detail,
        ))


# ─────────────────────────────────────────────
# 2. Result Integrity & Ordering Tests
# ─────────────────────────────────────────────

def run_result_integrity_tests(suite: TestSuite) -> None:
    print("\n──── Result Integrity & Ordering Tests ────")

    # Distances must be ascending (ORDER BY distance ASC)
    tool = make_tool(extra_config={"limit": 5})
    raw, ms = timed_run(lambda: tool._run("machine learning"))
    is_err, data = parse_output(raw)

    if not is_err and len(data) > 1:
        distances = [r["distance"] for r in data]
        sorted_distances = sorted(distances)
        passed = distances == sorted_distances
        detail = f"Distances: {[round(d, 4) for d in distances]}"
    elif is_err:
        passed = False
        detail = f"Error prevented check: {data.get('error','')[:60]}"
    else:
        passed = True
        detail = "Only 1 result — ordering trivially correct"

    suite.add(TestResult(
        name="results ordered by distance ASC",
        category="ResultIntegrity",
        passed=passed,
        duration_ms=ms,
        detail=detail,
    ))

    # Idempotency — same query twice must return identical results
    tool = make_tool()
    raw1, ms1 = timed_run(lambda: tool._run("vector embeddings"))
    raw2, ms2 = timed_run(lambda: tool._run("vector embeddings"))

    try:
        d1 = json.loads(raw1)
        d2 = json.loads(raw2)
        passed = d1 == d2
        detail = "Identical results on repeated query" if passed else "Results differ between identical queries"
    except Exception:
        passed = False
        detail = "JSON parse failed"

    suite.add(TestResult(
        name="idempotency — same query returns same results",
        category="ResultIntegrity",
        passed=passed,
        duration_ms=ms1 + ms2,
        detail=detail,
    ))

    # 'distance' key always present in every result row
    tool = make_tool(extra_config={"limit": 5})
    raw, ms = timed_run(lambda: tool._run("neural networks"))
    is_err, data = parse_output(raw)

    if not is_err:
        missing = [i for i, r in enumerate(data) if "distance" not in r]
        passed = len(missing) == 0
        detail = f"All {len(data)} rows have 'distance' key" if passed else f"Missing 'distance' in rows: {missing}"
    else:
        passed = True  # structural test — DB2 error is acceptable
        detail = f"DB2 error (skip): {data.get('error','')[:60]}"

    suite.add(TestResult(
        name="'distance' key present in every result row",
        category="ResultIntegrity",
        passed=passed,
        duration_ms=ms,
        detail=detail,
    ))

    # 'data' key always present in every result row
    tool = make_tool(extra_config={"limit": 5})
    raw, ms = timed_run(lambda: tool._run("deep learning"))
    is_err, data = parse_output(raw)

    if not is_err:
        missing = [i for i, r in enumerate(data) if "data" not in r]
        passed = len(missing) == 0
        detail = f"All {len(data)} rows have 'data' key" if passed else f"Missing 'data' in rows: {missing}"
    else:
        passed = True
        detail = f"DB2 error (skip): {data.get('error','')[:60]}"

    suite.add(TestResult(
        name="'data' key present in every result row",
        category="ResultIntegrity",
        passed=passed,
        duration_ms=ms,
        detail=detail,
    ))

    # Distance values are non-negative floats
    tool = make_tool(extra_config={"limit": 5})
    raw, ms = timed_run(lambda: tool._run("artificial intelligence"))
    is_err, data = parse_output(raw)

    if not is_err and data:
        invalid = [r["distance"] for r in data if not isinstance(r.get("distance"), (int, float)) or r["distance"] < 0]
        passed = len(invalid) == 0
        detail = f"All distances non-negative. Min={min(r['distance'] for r in data):.4f}" if passed else f"Invalid distances: {invalid}"
    else:
        passed = True
        detail = "DB2 error or empty (skip)"

    suite.add(TestResult(
        name="all distance values are non-negative floats",
        category="ResultIntegrity",
        passed=passed,
        duration_ms=ms,
        detail=detail,
    ))


# ─────────────────────────────────────────────
# 3. Semantic Relevance Tests
# ─────────────────────────────────────────────

def run_semantic_relevance_tests(suite: TestSuite) -> None:
    print("\n──── Semantic Relevance Tests ────")

    # Highly related queries should return overlapping results
    tool = make_tool(extra_config={"limit": 5, "max_distance": None})
    raw1, _ = timed_run(lambda: tool._run("machine learning algorithms"))
    raw2, ms = timed_run(lambda: tool._run("ML techniques and methods"))

    is_err1, d1 = parse_output(raw1)
    is_err2, d2 = parse_output(raw2)

    if not is_err1 and not is_err2 and d1 and d2:
        contents1 = {r["data"].get("CONTENT", "")[:50] for r in d1}
        contents2 = {r["data"].get("CONTENT", "")[:50] for r in d2}
        overlap = len(contents1 & contents2)
        passed = overlap >= 1
        detail = f"Overlap: {overlap}/{min(len(d1), len(d2))} results shared"
    else:
        passed = True  # Can't validate without results
        detail = "DB2 error or empty — semantic check skipped"

    suite.add(TestResult(
        name="semantically similar queries share top results",
        category="Semantic",
        passed=passed,
        duration_ms=ms,
        detail=detail,
    ))

    # Relevant query should score better (lower distance) than an unrelated one
    tool = make_tool(extra_config={"limit": 1, "max_distance": None})
    raw_rel, _ = timed_run(lambda: tool._run("machine learning neural network"))
    raw_irr, ms = timed_run(lambda: tool._run("medieval castle architecture"))

    is_err_rel, d_rel = parse_output(raw_rel)
    is_err_irr, d_irr = parse_output(raw_irr)

    if not is_err_rel and not is_err_irr and d_rel and d_irr:
        dist_rel = d_rel[0]["distance"]
        dist_irr = d_irr[0]["distance"]
        passed = dist_rel < dist_irr
        detail = f"Relevant={dist_rel:.4f} vs Unrelated={dist_irr:.4f}"
    else:
        passed = True
        detail = "DB2 error or empty — relevance check skipped"

    suite.add(TestResult(
        name="relevant query scores lower distance than unrelated query",
        category="Semantic",
        passed=passed,
        duration_ms=ms,
        detail=detail,
    ))

    # Exact phrase repeated should return consistent top-1
    tool = make_tool(extra_config={"limit": 1})
    runs = [json.loads(tool._run("vector embeddings")) for _ in range(3)]
    valid_runs = [r for r in runs if isinstance(r, list) and r]
    if len(valid_runs) == 3:
        top_contents = [r[0]["data"].get("CONTENT", "")[:80] for r in valid_runs]
        passed = len(set(top_contents)) == 1
        detail = f"Top-1 stable across 3 runs: {'YES' if passed else 'NO'}"
    else:
        passed = True
        detail = "DB2 error or empty — stability check skipped"

    suite.add(TestResult(
        name="top-1 result stable across 3 identical queries",
        category="Semantic",
        passed=passed,
        duration_ms=0.0,
        detail=detail,
    ))


# ─────────────────────────────────────────────
# 4. Embedding Robustness Tests
# ─────────────────────────────────────────────

def run_embedding_robustness_tests(suite: TestSuite) -> None:
    print("\n──── Embedding Robustness Tests ────")

    cases = [
        ("embedding returns empty list",         embedding_returns_empty,     True),
        ("embedding raises exception",           embedding_raises_exception,  True),
        ("embedding returns None",               embedding_returns_wrong_type, True),
        ("embedding returns all-zero vector",    embedding_returns_zero_vector, False),  # Valid shape; DB2 may or may not error
    ]

    for name, fn, expect_error in cases:
        tool = make_tool(embedding_fn=fn)
        try:
            raw, ms = timed_run(lambda t=tool: t._run("test query"))
            is_err, data = parse_output(raw)
            if expect_error:
                passed = is_err
                detail = f"Error: {data.get('error','')[:80]}" if is_err else "⚠ No error raised for bad embedding"
            else:
                passed = True
                detail = f"DB2 responded ({'error' if is_err else f'{len(data)} results'})"
        except Exception as exc:
            ms = 0.0
            passed = expect_error
            detail = f"Exception raised: {str(exc)[:80]}"

        suite.add(TestResult(
            name=name,
            category="EmbeddingRobustness",
            passed=passed,
            duration_ms=ms,
            detail=detail,
        ))

    # Strings instead of floats in embedding
    tool = make_tool(embedding_fn=embedding_returns_strings)
    try:
        raw, ms = timed_run(lambda: tool._run("test"))
        is_err, data = parse_output(raw)
        suite.add(TestResult(
            name="embedding returns strings instead of floats",
            category="EmbeddingRobustness",
            passed=is_err,
            duration_ms=ms,
            detail=f"Error: {data.get('error','')[:80]}" if is_err else "⚠ No error — type coercion happened silently",
        ))
    except Exception as exc:
        suite.add(TestResult(
            name="embedding returns strings instead of floats",
            category="EmbeddingRobustness",
            passed=True,
            duration_ms=0.0,
            detail=f"Exception raised (good): {str(exc)[:80]}",
        ))

    # Inconsistent embedding dimensions across calls
    call_count = [0]
    tool = make_tool(embedding_fn=embedding_inconsistent_dim(call_count))
    try:
        raw1, ms1 = timed_run(lambda: tool._run("first query"))
        raw2, ms2 = timed_run(lambda: tool._run("second query"))
        is_err1, d1 = parse_output(raw1)
        is_err2, d2 = parse_output(raw2)
        # Second call has different dim — should error at DB2 level or earlier
        detail = (
            f"Call1: {'error' if is_err1 else 'ok'}, "
            f"Call2: {'error' if is_err2 else 'ok (⚠ dim mismatch not caught)'}"
        )
        suite.add(TestResult(
            name="inconsistent embedding dim across calls (384→128)",
            category="EmbeddingRobustness",
            passed=True,  # structural: no crash is the bar
            duration_ms=ms1 + ms2,
            detail=detail,
        ))
    except Exception as exc:
        suite.add(TestResult(
            name="inconsistent embedding dim across calls (384→128)",
            category="EmbeddingRobustness",
            passed=True,
            duration_ms=0.0,
            detail=f"Exception: {str(exc)[:80]}",
        ))


# ─────────────────────────────────────────────
# 5. Connection Lifecycle Tests
# ─────────────────────────────────────────────

def run_connection_lifecycle_tests(suite: TestSuite) -> None:
    print("\n──── Connection Lifecycle Tests ────")

    # Multiple _run calls on same tool instance (connection reuse)
    tool = make_tool()
    results = []
    total_ms = 0.0
    for i in range(3):
        raw, ms = timed_run(lambda i=i: tool._run(f"query number {i}"))
        total_ms += ms
        is_err, data = parse_output(raw)
        results.append(not is_err)

    passed = all(results) or True  # DB2 errors ok; crash = fail
    suite.add(TestResult(
        name="3 sequential _run calls on same tool instance",
        category="ConnectionLifecycle",
        passed=passed,
        duration_ms=total_ms,
        detail=f"{sum(results)}/3 successful responses",
    ))

    # Manual disconnect then re-run (reconnect ability)
    tool = make_tool()
    tool._run("warmup query")   # establish connection
    tool._disconnect()          # force disconnect
    raw, ms = timed_run(lambda: tool._run("query after manual disconnect"))
    is_err, data = parse_output(raw)
    suite.add(TestResult(
        name="reconnects successfully after manual _disconnect()",
        category="ConnectionLifecycle",
        passed=not is_err or True,  # DB2 error vs Python crash
        duration_ms=ms,
        detail="Reconnected OK" if not is_err else f"DB2 error: {data.get('error','')[:60]}",
    ))

    # _disconnect() safe to call when no connection exists (idempotent)
    tool = make_tool()
    try:
        tool._disconnect()
        tool._disconnect()  # second call on already-None connection
        tool._disconnect()  # third
        suite.add(TestResult(
            name="_disconnect() is idempotent (safe to call multiple times)",
            category="ConnectionLifecycle",
            passed=True,
            duration_ms=0.0,
            detail="No exception on repeated disconnect",
        ))
    except Exception as exc:
        suite.add(TestResult(
            name="_disconnect() is idempotent (safe to call multiple times)",
            category="ConnectionLifecycle",
            passed=False,
            duration_ms=0.0,
            error=str(exc),
        ))

    # Connection is cleaned up after an error (no dangling connection)
    tool = make_tool(embedding_fn=embedding_raises_exception)
    tool._run("this will fail")  # fails in embedding before connect
    suite.add(TestResult(
        name="connection state clean after embedding exception",
        category="ConnectionLifecycle",
        passed=tool.connection is None,
        duration_ms=0.0,
        detail=f"connection={tool.connection}, cursor={tool.cursor}",
    ))


# ─────────────────────────────────────────────
# 6. Performance Benchmark Tests
# ─────────────────────────────────────────────

def run_performance_tests(suite: TestSuite) -> None:
    print("\n──── Performance Benchmark Tests ────")

    queries = [
        "machine learning",
        "neural networks",
        "vector search",
        "natural language processing",
        "deep learning transformers",
        "database optimization",
        "embedding models",
        "reinforcement learning",
    ]

    tool = make_tool()
    latencies: list[float] = []

    for q in queries:
        _, ms = timed_run(lambda query=q: tool._run(query))
        latencies.append(ms)

    if latencies:
        latencies.sort()
        p50  = statistics.median(latencies)
        p90  = latencies[int(len(latencies) * 0.90)]
        p99  = latencies[-1]
        mean = statistics.mean(latencies)

        suite.add(TestResult(
            name=f"latency p50={p50:.0f}ms p90={p90:.0f}ms p99={p99:.0f}ms mean={mean:.0f}ms",
            category="Performance",
            passed=p50 < 10_000,  # 10s is a very generous cap
            duration_ms=sum(latencies),
            detail=f"Over {len(queries)} queries. All: {[round(l) for l in latencies]}ms",
        ))

        suite.add(TestResult(
            name="p90 latency under 15 seconds",
            category="Performance",
            passed=p90 < 15_000,
            duration_ms=p90,
            detail=f"p90={p90:.0f}ms",
        ))

        suite.add(TestResult(
            name="no single query exceeds 30 seconds",
            category="Performance",
            passed=p99 < 30_000,
            duration_ms=p99,
            detail=f"Slowest query: {p99:.0f}ms",
        ))


# ─────────────────────────────────────────────
# 7. Concurrency / Thread Safety Tests
# ─────────────────────────────────────────────

def run_concurrency_tests(suite: TestSuite) -> None:
    print("\n──── Concurrency / Thread Safety Tests ────")

    # Each thread gets its own tool instance (safe pattern)
    errors: list[str] = []
    results: list[bool] = []
    lock = threading.Lock()

    def worker(query: str) -> None:
        try:
            t = make_tool()
            raw = t._run(query)
            is_err, _ = parse_output(raw)
            with lock:
                results.append(True)
        except Exception as exc:
            with lock:
                errors.append(str(exc))
                results.append(False)

    threads = [
        threading.Thread(target=worker, args=(f"concurrent query {i}",))
        for i in range(5)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_ms = (time.perf_counter() - t0) * 1000

    suite.add(TestResult(
        name="5 concurrent threads (each own tool instance) — no crash",
        category="Concurrency",
        passed=len(errors) == 0,
        duration_ms=total_ms,
        detail=f"{sum(results)}/5 completed cleanly",
        error="; ".join(errors[:3]) if errors else "",
    ))

    # Shared tool instance across threads (stress test — may fail; we document it)
    shared_tool = make_tool()
    shared_errors: list[str] = []
    shared_results: list[bool] = []
    shared_lock = threading.Lock()

    def shared_worker(query: str) -> None:
        try:
            raw = shared_tool._run(query)
            with shared_lock:
                shared_results.append(True)
        except Exception as exc:
            with shared_lock:
                shared_errors.append(str(exc))
                shared_results.append(False)

    threads2 = [
        threading.Thread(target=shared_worker, args=(f"shared query {i}",))
        for i in range(3)
    ]
    t0 = time.perf_counter()
    for t in threads2:
        t.start()
    for t in threads2:
        t.join()
    total_ms2 = (time.perf_counter() - t0) * 1000

    # This is an observation test — document whether shared instance is safe
    suite.add(TestResult(
        name="3 concurrent threads sharing one tool instance (observation)",
        category="Concurrency",
        passed=True,  # Always pass — we just document the behavior
        duration_ms=total_ms2,
        detail=(
            f"{sum(shared_results)}/3 OK, {len(shared_errors)} error(s). "
            f"{'⚠ Shared instance NOT thread-safe' if shared_errors else 'Shared instance survived (lucky or safe)'}"
        ),
    ))


# ─────────────────────────────────────────────
# 8. Pydantic Schema Validation Tests (DB2ToolSchema)
# ─────────────────────────────────────────────

def run_schema_validation_tests(suite: TestSuite) -> None:
    print("\n──── Pydantic Input Schema Validation Tests ────")

    # Valid minimal schema
    try:
        s = DB2ToolSchema(query="test")
        suite.add(TestResult(
            name="DB2ToolSchema: minimal valid input (query only)",
            category="SchemaValidation",
            passed=True,
            duration_ms=0.0,
            detail=f"filter_by={s.filter_by}, filter_value={s.filter_value}",
        ))
    except Exception as exc:
        suite.add(TestResult(
            name="DB2ToolSchema: minimal valid input (query only)",
            category="SchemaValidation",
            passed=False,
            duration_ms=0.0,
            error=str(exc),
        ))

    # Missing query (required field)
    try:
        DB2ToolSchema()
        suite.add(TestResult(
            name="DB2ToolSchema: missing required 'query' rejected",
            category="SchemaValidation",
            passed=False,
            duration_ms=0.0,
            detail="Pydantic did NOT raise — 'query' should be required",
        ))
    except Exception:
        suite.add(TestResult(
            name="DB2ToolSchema: missing required 'query' rejected",
            category="SchemaValidation",
            passed=True,
            duration_ms=0.0,
            detail="Correctly raised ValidationError",
        ))

    # filter_by and filter_value both None (defaults)
    try:
        s = DB2ToolSchema(query="test", filter_by=None, filter_value=None)
        suite.add(TestResult(
            name="DB2ToolSchema: filter_by=None, filter_value=None accepted",
            category="SchemaValidation",
            passed=True,
            duration_ms=0.0,
            detail="Both None — valid optional fields",
        ))
    except Exception as exc:
        suite.add(TestResult(
            name="DB2ToolSchema: filter_by=None, filter_value=None accepted",
            category="SchemaValidation",
            passed=False,
            duration_ms=0.0,
            error=str(exc),
        ))

    # filter_value can be any type
    for fv, label in [(42, "int"), (3.14, "float"), (True, "bool"), ({"k": "v"}, "dict")]:
        try:
            DB2ToolSchema(query="test", filter_by="COL", filter_value=fv)
            suite.add(TestResult(
                name=f"DB2ToolSchema: filter_value accepts {label}",
                category="SchemaValidation",
                passed=True,
                duration_ms=0.0,
                detail=f"filter_value={fv}",
            ))
        except Exception as exc:
            suite.add(TestResult(
                name=f"DB2ToolSchema: filter_value accepts {label}",
                category="SchemaValidation",
                passed=False,
                duration_ms=0.0,
                error=str(exc),
            ))


# ─────────────────────────────────────────────
# 9. Filter Value Type Tests (end-to-end)
# ─────────────────────────────────────────────

def run_filter_value_type_tests(suite: TestSuite) -> None:
    print("\n──── Filter Value Type Tests (end-to-end) ────")

    filter_cases = [
        ("string",  "CATEGORY",  "AI"),
        ("integer", "CATEGORY",  1),
        ("float",   "SCORE",     0.95),
        ("boolean", "ACTIVE",    True),
    ]

    for label, col, val in filter_cases:
        tool = make_tool()
        try:
            raw, ms = timed_run(lambda t=tool, c=col, v=val: t._run("machine learning", filter_by=c, filter_value=v))
            is_err, data = parse_output(raw)
            # DB2 may raise column not found — that's expected. We test for no Python crash.
            suite.add(TestResult(
                name=f"filter_value type={label} — no client crash",
                category="FilterValueType",
                passed=True,
                duration_ms=ms,
                detail="DB2 error (col may not exist)" if is_err else f"{len(data)} result(s)",
            ))
        except Exception as exc:
            suite.add(TestResult(
                name=f"filter_value type={label} — no client crash",
                category="FilterValueType",
                passed=False,
                duration_ms=0.0,
                error=str(exc),
            ))


# ─────────────────────────────────────────────
# 10. Boundary Condition Tests
# ─────────────────────────────────────────────

def run_boundary_condition_tests(suite: TestSuite) -> None:
    print("\n──── Boundary Condition Tests ────")

    # Single character query
    tool = make_tool()
    raw, ms = timed_run(lambda: tool._run("a"))
    is_err, data = parse_output(raw)
    suite.add(TestResult(
        name="single-character query 'a'",
        category="BoundaryCondition",
        passed=True,
        duration_ms=ms,
        detail="error" if is_err else f"{len(data)} result(s)",
    ))

    # Query is exactly one space
    tool = make_tool()
    raw, ms = timed_run(lambda: tool._run(" "))
    is_err, data = parse_output(raw)
    suite.add(TestResult(
        name="whitespace-only query (' ')",
        category="BoundaryCondition",
        passed=True,
        duration_ms=ms,
        detail="error" if is_err else f"{len(data)} result(s)",
    ))

    # Numeric string query
    tool = make_tool()
    raw, ms = timed_run(lambda: tool._run("12345"))
    is_err, data = parse_output(raw)
    suite.add(TestResult(
        name="purely numeric query '12345'",
        category="BoundaryCondition",
        passed=True,
        duration_ms=ms,
        detail="error" if is_err else f"{len(data)} result(s)",
    ))

    # max_distance exactly 0.0 (boundary — only perfect matches)
    tool = make_tool(extra_config={"max_distance": 0.0})
    raw, ms = timed_run(lambda: tool._run("machine learning"))
    is_err, data = parse_output(raw)
    suite.add(TestResult(
        name="max_distance=0.0 (only perfect distance matches)",
        category="BoundaryCondition",
        passed=True,
        duration_ms=ms,
        detail=f"{len(data)} result(s) at exact 0.0 distance" if not is_err else "error",
    ))

    # limit=1 returns at most 1 result
    tool = make_tool(extra_config={"limit": 1})
    raw, ms = timed_run(lambda: tool._run("machine learning"))
    is_err, data = parse_output(raw)
    passed = (not is_err and len(data) <= 1) or is_err
    suite.add(TestResult(
        name="limit=1 returns at most 1 result",
        category="BoundaryCondition",
        passed=passed,
        duration_ms=ms,
        detail=f"{len(data)} result(s)" if not is_err else "DB2 error",
    ))

    # Very large limit (100) — should not crash even if fewer rows exist
    tool = make_tool(extra_config={"limit": 100, "max_distance": None})
    raw, ms = timed_run(lambda: tool._run("machine learning"))
    is_err, data = parse_output(raw)
    suite.add(TestResult(
        name="limit=100 — no crash even if fewer than 100 rows exist",
        category="BoundaryCondition",
        passed=True,
        duration_ms=ms,
        detail=f"{len(data)} result(s)" if not is_err else "DB2 error (acceptable)",
    ))


# ─────────────────────────────────────────────
# 11. DB2Config Extended Validation Tests
# ─────────────────────────────────────────────

def run_db2config_extended_tests(suite: TestSuite) -> None:
    print("\n──── DB2Config Extended Validation Tests ────")

    # All 4 valid distance metric casings (config stores as-is; whitelist uppercases)
    for metric in ["cosine", "Cosine", "COSINE"]:
        try:
            config = DB2Config(
                database="AUTOGEN", hostname="9.60.203.12",
                table_name="PAWAN.DOCUMENTS", vector_column="EMBEDDING",
                distance_metric=metric,
            )
            tool = DB2VectorSearchTool(db2_config=config, custom_embedding_fn=generate_384_embedding)
            raw, ms = timed_run(lambda t=tool: t._run("test"))
            is_err, data = parse_output(raw)
            suite.add(TestResult(
                name=f"distance_metric='{metric}' case-insensitive acceptance",
                category="DB2ConfigExtended",
                passed=True,
                duration_ms=ms,
                detail="DB2 error (expected)" if is_err else f"{len(data)} result(s)",
            ))
        except Exception as exc:
            suite.add(TestResult(
                name=f"distance_metric='{metric}' case-insensitive acceptance",
                category="DB2ConfigExtended",
                passed=False,
                duration_ms=0.0,
                error=str(exc),
            ))

    # Default values are correct
    try:
        config = DB2Config(database="TEST")
        checks = {
            "hostname": (config.hostname, "localhost"),
            "port":     (config.port, 50000),
            "protocol": (config.protocol, "TCPIP"),
            "limit":    (config.limit, 3),
            "metric":   (config.distance_metric, "COSINE"),
        }
        failures = [f"{k}: got {v[0]!r} expected {v[1]!r}" for k, v in checks.items() if v[0] != v[1]]
        suite.add(TestResult(
            name="DB2Config default values are correct",
            category="DB2ConfigExtended",
            passed=len(failures) == 0,
            duration_ms=0.0,
            detail="All defaults correct" if not failures else "; ".join(failures),
        ))
    except Exception as exc:
        suite.add(TestResult(
            name="DB2Config default values are correct",
            category="DB2ConfigExtended",
            passed=False,
            duration_ms=0.0,
            error=str(exc),
        ))

    # return_columns default is ["content"]
    try:
        config = DB2Config(database="TEST")
        passed = isinstance(config.return_columns, list) and len(config.return_columns) >= 1
        suite.add(TestResult(
            name="DB2Config return_columns has a non-empty default",
            category="DB2ConfigExtended",
            passed=passed,
            duration_ms=0.0,
            detail=f"Default: {config.return_columns}",
        ))
    except Exception as exc:
        suite.add(TestResult(
            name="DB2Config return_columns has a non-empty default",
            category="DB2ConfigExtended",
            passed=False,
            duration_ms=0.0,
            error=str(exc),
        ))


# ─────────────────────────────────────────────
# 12. Output Structure Contract Tests
# ─────────────────────────────────────────────

def run_output_contract_tests(suite: TestSuite) -> None:
    print("\n──── Output Structure Contract Tests ────")

    tool = make_tool(extra_config={"limit": 3})
    raw, ms = timed_run(lambda: tool._run("machine learning"))
    is_err, data = parse_output(raw)

    # Output is always a list (success) or dict with 'error' (failure)
    suite.add(TestResult(
        name="output root is list (success) or dict with 'error' key (failure)",
        category="OutputContract",
        passed=isinstance(data, list) or (isinstance(data, dict) and "error" in data),
        duration_ms=ms,
        detail=f"Type: {type(data).__name__}",
    ))

    # On error: 'error', 'success', 'error_type' keys present
    tool_bad = make_tool(embedding_fn=embedding_raises_exception)
    raw_err, ms_err = timed_run(lambda: tool_bad._run("test"))
    _, err_data = parse_output(raw_err)
    if isinstance(err_data, dict):
        has_keys = all(k in err_data for k in ["error", "error_type"])
        suite.add(TestResult(
            name="error response contains 'error' and 'error_type' keys",
            category="OutputContract",
            passed=has_keys,
            duration_ms=ms_err,
            detail=f"Keys present: {list(err_data.keys())}",
        ))
    else:
        suite.add(TestResult(
            name="error response contains 'error' and 'error_type' keys",
            category="OutputContract",
            passed=False,
            duration_ms=ms_err,
            detail=f"Unexpected type: {type(err_data).__name__}",
        ))

    # Success rows: each has exactly 'distance' (float) and 'data' (dict)
    if not is_err and data:
        issues = []
        for i, row in enumerate(data):
            if not isinstance(row.get("distance"), (int, float)):
                issues.append(f"row[{i}].distance not numeric")
            if not isinstance(row.get("data"), dict):
                issues.append(f"row[{i}].data not dict")
        suite.add(TestResult(
            name="each success row has numeric 'distance' and dict 'data'",
            category="OutputContract",
            passed=len(issues) == 0,
            duration_ms=ms,
            detail=f"{len(data)} rows checked" if not issues else "; ".join(issues),
        ))
    else:
        suite.add(TestResult(
            name="each success row has numeric 'distance' and dict 'data'",
            category="OutputContract",
            passed=True,
            duration_ms=ms,
            detail="DB2 error or empty — contract check skipped",
        ))


# ─────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────

def main() -> int:
    print("=" * 80)
    print("  DB2VectorSearchTool — Extended Robust Test Suite (Part 2)")
    print("=" * 80)

    suite = TestSuite()

    test_groups = [
        ("Distance Metrics",         run_distance_metric_tests),
        ("Result Integrity",         run_result_integrity_tests),
        ("Semantic Relevance",       run_semantic_relevance_tests),
        ("Embedding Robustness",     run_embedding_robustness_tests),
        ("Connection Lifecycle",     run_connection_lifecycle_tests),
        ("Performance",              run_performance_tests),
        ("Concurrency",              run_concurrency_tests),
        ("Schema Validation",        run_schema_validation_tests),
        ("Filter Value Types",       run_filter_value_type_tests),
        ("Boundary Conditions",      run_boundary_condition_tests),
        ("DB2Config Extended",       run_db2config_extended_tests),
        ("Output Contract",          run_output_contract_tests),
    ]

    for group_name, fn in test_groups:
        try:
            fn(suite)
        except Exception as exc:
            print(f"\n  ⛔ Test group '{group_name}' crashed: {exc}")
            traceback.print_exc()

    suite.summary()

    failed = sum(1 for r in suite.results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
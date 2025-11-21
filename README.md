# LMCache TTL Heuristics — KV Cache Experiments

This repository contains experiments and prototype code for integrating Time-To-Live (TTL) caching heuristics with an LMCache-style key-value (KV) cache, intended to be used inside a larger agentic process. The goal is to provide reusable strategies and a lightweight prototype for placing, aging, and evicting cached LM responses so agents can reduce cost and latency while staying robust.

## Project goal

- Provide a minimal, reproducible workspace for designing and testing TTL-based heuristics for LM response caching.
- Integrate heuristics with the LMCache KV cache abstraction so an agent can check the cache before calling a model, update TTLs on access, and asynchronously warm or evict entries.
- Make it easy to iterate on heuristics (static TTLs, size-aware TTLs, adaptive TTLs based on reuse probability or cost) and measure hit-rate / cost savings.

## Architecture

- Agents are ephemeral (stateless). They do orchestration, tool calls, prompt construction only. No model/LMCache in agents.
- Inference engine is stateful. It hosts the model and LMCache (GPU+Disk only) and exposes HTTP endpoints.
- LMCache tiers: GPU (hot) + Disk (warm on local NVMe). CPU and remote tiers are disabled. LMCache auto-offloads GPU→Disk (LRU).
- S3/Glacier archive is external and TTL-managed outside LMCache. A background worker compresses KV and uploads with TTL metadata. On deep miss, the engine fetches from S3 and hydrates GPU LMCache via a hydrate shim.

## Key files

- `src/caching_heuristics.py` — GPU/Disk-only heuristics; no remote.
- `src/tiered_caching.py` — logs TTL/backend decisions; does not orchestrate moves (LMCache LRU handles GPU→Disk).
- `src/engine_app.py` — inference engine API skeleton with `/generate`, `/kv/hydrate`, `/kv/lookup`, `/healthz`.
- `src/engine_lmcache_controller_client.py` — simple controller client + hydrate shim call.
- `src/engine_serialization.py` — gzip compress/decompress helpers.
- `src/s3_archive.py` — external TTL archive adapter for S3 (local fallback in dev).
- `src/workers_archive_worker.py` — background archival worker skeleton.
- `src/notebook_bootstrap.py` — helper to start/stop the LMCache controller from notebooks.

## Contract (inputs / outputs)

- Inputs: prompt identifier (string or hash), prompt metadata (optional dict: size, cost, expected reuse probability, timestamp).
- Outputs: cached KV batches, output payload (string/JSON), metadata (inserted_at, ttl_seconds, last_accessed, access_count, score).
- Error modes: missing backend credentials, malformed keys, concurrent update races.

## TTL heuristics overview

You can experiment with several families of TTL heuristics. Each heuristic implements a function that maps prompt + metadata -> TTL seconds and optionally an initial priority/score.

Suggested adaptive TTL formula (starter):

TTL = clamp(base_ttl * (1 + alpha * access_count), min_ttl, max_ttl)

Tune alpha/beta to control sensitivity to reuse and cost.

## Integrating with LMCache KV

- Agent calls inference engine only. Engine consults LMCache (GPU→Disk). On deep miss, it may fetch from S3 and hydrate GPU via `/kv/hydrate`.
- LMCache never writes to S3. All TTL stays in S3 metadata or an external index.

Integration pseudocode:

```python
key = normalize_prompt(prompt)
entry = kv.get(key)
if entry and not expired(entry):
	entry['access_count'] += 1
	kv.update(key, entry)
	return entry['response']
else:
	resp = call_model(prompt)
	ttl = compute_ttl(prompt, metadata)
	kv.put(key, {'response': resp, 'inserted_at': now(), 'ttl_seconds': ttl, 'access_count': 0})
	return resp
```

## Agentic process considerations

- Decision point: where in the agent pipeline to check the cache (before planning, after plan generation, or per-tool call).
- Consistency model: if agent composes multiple prompts, consider whether to cache intermediate tool outputs and how to invalidate them when upstream state changes.
- Async warming: agents can queue promising prompts for background warming using predicted reuse probability.
- TTL updates: when access_count increases, you can extend the TTL or refresh the deadline (sliding TTL) depending on semantics you want.

## Metrics & validation

Measure:
- Hit rate (per prompt class)
- Effective cost saved (tokens/requests avoided)
- Latency reduction
- Storage usage and eviction rate

Evaluate heuristics on representative workloads (agent logs or simulated prompts). Use notebooks in `notebooks/` to instrument runs and emit CSV/plots.

## Runtime toggles

- `LMCACHE_DISABLE_OFFLOAD` ("0" | "1"): when "1", logging only; LMCache still uses LRU but we don’t orchestrate moves.

## S3 TTL archive

- Use `src/s3_archive.py` for external TTL storage. In dev (no AWS), it falls back to a local folder.
- Background worker: `src/workers_archive_worker.py` runs periodically to upload entries.

Note: LMCache remote tier is disabled. LMCache never writes to S3; TTL lives entirely in S3 metadata or an external index.

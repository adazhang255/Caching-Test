# LMCache TTL Heuristics — KV Cache Experiments

This repository contains experiments and prototype code for integrating Time-To-Live (TTL) caching heuristics with an LMCache-style key-value (KV) cache, intended to be used inside a larger agentic process. The goal is to provide reusable strategies and a lightweight prototype for placing, aging, and evicting cached LM responses so agents can reduce cost and latency while staying robust.

## Project goal

- Provide a minimal, reproducible workspace for designing and testing TTL-based heuristics for LM response caching.
- Integrate heuristics with the LMCache KV cache abstraction so an agent can check the cache before calling a model, update TTLs on access, and asynchronously warm or evict entries.
- Make it easy to iterate on heuristics (static TTLs, size-aware TTLs, adaptive TTLs based on reuse probability or cost) and measure hit-rate / cost savings.

## Architecture & flow (high level)

1. The agent receives or generates a prompt.
2. Agent queries the LMCache KV cache for the prompt key (or normalized key).
3. If hit: agent receives the cached response and the TTL heuristic may update metadata (e.g., extend TTL or promote entry).
4. If miss: agent calls the LM, stores the KV tensors into KV cache with an initial TTL computed by the chosen heuristic.
5. Background processes may warm entries, apply decay, or evict entries when backends fail or storage limits are hit.

## Key files

- `src/caching_heuristics.py` — central heuristics used to compute TTLs / priorities (edit and iterate here).
- `Multi-Tier Caching/cache_controller.py` — orchestrates cache operations across tiers.
- `Multi-Tier Caching/lmcache_s3_backend.py` — optional S3-backed backend for persistent K/V storage.
- `archive/` — quick notebooks and a simple script to exercise caching flows and examples.

## Contract (inputs / outputs)

- Inputs: prompt identifier (string or hash), prompt metadata (optional dict: size, cost, expected reuse probability, timestamp).
- Outputs: cached KV batches, output payload (string/JSON), metadata (inserted_at, ttl_seconds, last_accessed, access_count, score).
- Error modes: missing backend credentials, malformed keys, concurrent update races.

## TTL heuristics overview

You can experiment with several families of TTL heuristics. Each heuristic implements a function that maps prompt + metadata -> TTL seconds and optionally an initial priority/score.

Suggested adaptive TTL formula (starter):

TTL = clamp(base_ttl * (1 + alpha * access_count) * (1 + beta * cost_tokens/100), min_ttl, max_ttl)

Tune alpha/beta to control sensitivity to reuse and cost.

## Integrating with LMCache KV

- Key normalization: derive a stable key from the prompt (hash or canonicalized string) so agent and cache agree.
- On write: compute TTL via the chosen heuristic and store response with metadata fields: inserted_at, ttl_seconds, access_count=0.
- On read (hit): increment access_count, update last_accessed, optionally recompute ttl and update the KV entry with new ttl_seconds.
- On miss: call model, store entry with compute_ttl(...).

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


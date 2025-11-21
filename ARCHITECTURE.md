# LMCache TTL-Heuristics Architecture

## Overview

This project integrates **LMCache** (automatic KV tensor caching for vLLM) with **TTL-driven heuristics** to intelligently migrate KV tensors between multi-tier storage backends (GPU, disk, remote) based on prompt characteristics.

---

## Components

### 1. **Heuristics Engine** (`src/caching_heuristics.py`)

Computes two key decisions after each LLM generation:

#### `compute_ttl(metadata, base_ttl=3600, min_ttl=30, max_ttl=604800, alpha=0.25)`
Calculates how long to keep a KV tensor in hot storage before migrating it.

**Formula:**
```
ttl = base_ttl × (1 + α·ln(1 + access_count)) × (1.0 + perplexity_factor) × (1 - γ·time_variance)
```

**Metadata fields:**
- `perplexity`: Prompt complexity (10-100 typical)
- `time_variance`: Temporal randomness (0-1)
- `access_count`: Number of times this prompt pattern has been seen

**Output:** TTL in seconds (clamped to [min_ttl, max_ttl])

#### `select_backend(metadata, hot_threshold=20.0, recent_seconds=3600)`
Selects which storage tier is best for the KV tensor.

**Rules:**
- **"gpu"** (hot): Low perplexity + recently accessed → keep in GPU VRAM
- **"disk"** (warm): High variance or moderate perplexity → warm cache on disk
- **"remote"** (cold): High perplexity → remote Redis/S3 backend

**Output:** Backend name ("gpu" | "disk" | "remote")

---

### 2. **Multi-Tier Caching Controller** (`src/cache_controller.py`)

#### `LMCacheController(host="localhost", port=9000, model="gemma-3-270m-it")`
HTTP client for the LMCache server (handles KV storage/retrieval).

**Key methods:**
```python
def tokenize(prompt: str) -> list[int]:
    # Tokenize prompt via vLLM's tokenizer (via LMCache API)
    return tokens

def lookup(tokens: list[int]) -> dict:
    # Query where this KV is currently stored
    # Returns: {"found": bool, "lmcache_default_instance": ["gpu" or "disk" or "remote"]}
    return layout

def move(old_position: list[str], new_position: list[str]) -> dict:
    # Migrate KV from one backend to another
    return status
```

**Note:** `LocalLMCacheController` is an alternative for local-only testing without an LMCache server.

---

### 3. **Backend Implementations** (`src/tiered_caching.py`)

Abstract layer for pluggable storage backends:

#### `BaseBackend` (abstract)
```python
def get(key: str) -> Any
def set(key: str, value: Any, ttl: int)
def delete(key: str)
```

#### `InMemoryBackend`
- Hot cache: in-memory dict with TTL-based expiry
- **Use case:** Most frequently accessed KVs

#### `LocalDiskBackend`
- Warm cache: pickle-based disk storage
- **Use case:** Medium-priority KVs (e.g., 60-300s TTL)
- **Path:** `./cache_warm/`

#### `RemoteBackend`
- Cold cache: Redis-backed (uses fakeredis for testing)
- **Use case:** Long-lived KVs (e.g., 3600s+ TTL)
- **Compatibility:** Works with real Redis or embedded FakeRedis

---

### 4. **Multi-Tier Cache Manager** (`src/tiered_caching.py`)

#### `MultiTierCache(lmcache_controller, llm=None)`
**Orchestrates automatic KV management:**

```python
def generate_and_manage(prompt, sampling_params, metadata=None) -> str:
    """
    1. Calls llm.generate() → LMCache stores KV automatically
    2. Tokenizes prompt via controller.tokenize()
    3. Queries current KV location via controller.lookup()
    4. Computes TTL via compute_ttl(metadata)
    5. Selects backend via select_backend(metadata)
    6. If backend differs, calls controller.move() to migrate KV
    7. Returns generated output
    """
```

**Key feature:** TTL heuristics are applied **automatically after every generation** with no manual intervention needed.

---

## Workflow

### Typical Agent Loop

```
1. Agent: [Prompt A] + Metadata {perplexity: 15, time_variance: 0.05}
   ↓
2. cache_manager.generate_and_manage(prompt_a, params, metadata)
   - vLLM/LMCache generate and store KV in GPU
   - Compute TTL: 3600s (base) × 1.08 (low perplexity) × 0.95 (low variance) ≈ 3700s
   - Select backend: "gpu" (low perplexity + recent)
   - Action: Keep in GPU ✓
   ↓
3. [Later, second prompt uses cached KV prefix]
   - access_count increments
   - TTL re-computed with higher weight
   - If perplexity changes → backend may shift (e.g., GPU→disk)
   ↓
4. [After TTL expires]
   - Access time > TTL → KV no longer in hot storage
   - Either: re-fetch from warm/cold or re-generate
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────┐
│   Agent / Agentic Framework         │
│   (e.g., LangChain, AutoGen)        │
└─────────────────┬───────────────────┘
                  │ prompt + metadata
                  ↓
        ┌─────────────────────┐
        │  MultiTierCache     │
        │ generate_and_manage │
        └──────────┬──────────┘
                   │
      ┌────────────┼────────────┐
      ↓            ↓            ↓
  ┌──────────┐ ┌─────────----─┐ ┌───────────┐
  │   llm    │ │ Heuristics   │ │LMCache    │
  │.generate │ │ Engine       │ │Controller │
  └────┬─────┘ │compute_ttl   │ │ tokenize()│
       │       │select_backend│ │ lookup()  │
       │       └──────────────┘ │ move()    │
       │                        └─────┬─────┘
       │                              │
       └────────────────┬─────────────┘
                        ↓
          LMCache Server (localhost:9000)
          - Tokenizer
          - KV Store Index
          - Backend Coordinatorn
                        │
       ┌────────────────┼────────────────┐
       ↓                ↓                ↓
   ┌────────┐      ┌──────────┐   ┌──────────┐
   │ GPU    │      │ Disk     │   │ Remote   │
   │VRAM    │      │ (Local)  │   │(Redis)   │
   │ Hot    │      │ Warm     │   │ Cold     │
   └────────┘      └──────────┘   └──────────┘
```

---

## Usage in Notebook

### Setup (Cell 2a)
```python
from tiered_caching import MultiTierCache
cache_manager = MultiTierCache(lmcache_controller)
```

### After LLM Init (Cell 5b)
```python
cache_manager.set_llm(llm)
print("✓ Ready for managed generation!")
```

### Generation with Automatic Heuristics (Cell 6+)
```python
prompt = "What is machine learning?"
metadata = {
    "perplexity": 12.5,
    "time_variance": 0.08
}

output = cache_manager.generate_and_manage(
    prompt, 
    sampling_params, 
    metadata
)
print(output)
# ✓ Moved KV from gpu to disk (TTL: 1800s)
# ✓ KV in remote, TTL: 7200s
```

---

## Key Design Principles

1. **LMCache Owns KV Storage**: User code only decides *where* KV tensors live, not *if* they're stored.
2. **Automatic Heuristics**: TTL computation happens transparently after each generation.
3. **Pluggable Backends**: Swap in/out storage implementations without changing core logic.
4. **Graceful Degradation**: If LMCache server is unavailable, generation still works (heuristics just fail silently).
5. **Agent-Friendly**: Designed for agentic workflows where the same prompt patterns repeat with variations in metadata.

---

## Configuration

### TTL Heuristics Tuning
In `src/caching_heuristics.py`, adjust:
- `base_ttl`: Default TTL in seconds (default: 3600s = 1 hour)
- `min_ttl`: Minimum TTL (default: 30s)
- `max_ttl`: Maximum TTL (default: 604800s = 7 days)
- `alpha`: Access count weight (default: 0.25)

### Backend Selection Thresholds
- `hot_threshold`: Perplexity cutoff for GPU (default: 20.0)
- `recent_seconds`: Time window for "recent" (default: 3600s)

### LMCache Server
- Expected at `localhost:9000` (configurable in LMCacheController)
- Requires `config.yaml` with LMCache settings

---

## Dependencies

- **vLLM** ≥ 0.6: GPU inference engine
- **LMCache**: KV tensor caching server
- **requests**: HTTP client for LMCache API
- **fakeredis**: Embedded Redis for testing (optional, can use real Redis)

---

## Next Steps

1. **Agent Integration**: Wrap `MultiTierCache.generate_and_manage()` in agent tools
2. **Unit Tests**: Add tests for heuristic formulas and backend selection
3. **Monitoring**: Log TTL/backend decisions for analysis
4. **Benchmarking**: Compare speedup vs. naive caching across different workloads
5. **Production Redis**: Replace fakeredis with real Redis for scale

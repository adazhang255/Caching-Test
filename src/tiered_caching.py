import time
import pickle
import os
from typing import Optional, Dict, Any

from caching_heuristics import compute_ttl, select_backend

# Use fakeredis for testing/dev instead of a real Redis server.
try:
    import fakeredis
except Exception:
    fakeredis = None

# -----------------------------
# Simple Backend Interfaces
# -----------------------------

class BaseBackend:
    def get(self, key):
        raise NotImplementedError
    
    def set(self, key, value, ttl=None):
        raise NotImplementedError

    def delete(self, key):
        raise NotImplementedError

class InMemoryBackend(BaseBackend):
    """Hot cache: in-memory KV store with TTL"""
    def __init__(self):
        self.store = {}  # key -> (value, expire_time)

    def get(self, key):
        if key in self.store:
            val, expire = self.store[key]
            if expire is None or expire > time.time():
                return val
            else:
                del self.store[key]
        return None

    def set(self, key, value, ttl=None):
        expire = time.time() + ttl if ttl else None
        self.store[key] = (value, expire)

    def delete(self, key):
        self.store.pop(key, None)


class LocalDiskBackend(BaseBackend):
    """Warm cache: store pickled values on disk with TTL"""
    def __init__(self, directory="./cache_warm"):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, key):
        return os.path.join(self.directory, f"{key}.pkl")

    def get(self, key):
        path = self._path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            value, expire = pickle.load(f)
        if expire is None or expire > time.time():
            return value
        else:
            os.remove(path)
            return None

    def set(self, key, value, ttl=None):
        expire = time.time() + ttl if ttl else None
        with open(self._path(key), "wb") as f:
            pickle.dump((value, expire), f)

    def delete(self, key):
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)


class RemoteBackend(BaseBackend):
    """Cold cache backed by Redis (fake/embedded redis for tests).

    This implementation uses fakeredis.FakeStrictRedis when available. It
    pickles Python objects into Redis binary values and relies on Redis TTL
    semantics for expiration (setex). For production you can pass a real
    `redis.StrictRedis`/`redis.Redis` client into the constructor instead.
    """
    def __init__(self, redis_client: Optional[object] = None, namespace: str = "remote_cache"):
        # If a client is provided, use it. Otherwise try to create a fakeredis client.
        if redis_client is not None:
            self.client = redis_client
        else:
            if fakeredis is None:
                raise RuntimeError("fakeredis not available; install fakeredis to use RemoteBackend without a Redis server")
            # create an in-memory fake redis instance
            self.client = fakeredis.FakeStrictRedis()

        self.namespace = namespace

    def _k(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get(self, key):
        raw = self.client.get(self._k(key))
        if raw is None:
            return None
        try:
            return pickle.loads(raw)
        except Exception:
            # If stored as plain bytes, return raw
            return raw

    def set(self, key, value, ttl=None):
        blob = pickle.dumps(value)
        if ttl:
            # setex expects seconds
            self.client.setex(self._k(key), int(ttl), blob)
        else:
            self.client.set(self._k(key), blob)

    def delete(self, key):
        self.client.delete(self._k(key))


# -----------------------------
# Multi-Tier TTL-Aware Cache
# -----------------------------

class MultiTierCache:
    """Manages multi-tier KV cache placement via LMCache controller.
    
    After each LLM generation, applies TTL heuristics to decide if/where
    to move KV tensors between storage backends (GPU, disk, remote).
    """
    
    def __init__(self, lmcache_controller, llm=None):
        """
        Args:
            lmcache_controller: LMCacheController instance (manages KV placement)
            llm: vLLM model instance (for generation). Can be set later.
        """
        self.controller = lmcache_controller
        self.llm = llm
        self.access_count = {}  # track prompt accesses for heuristics

    def set_llm(self, llm):
        """Set the LLM instance (useful if not available at init time)."""
        self.llm = llm

    def generate_and_manage(self, prompt, sampling_params, metadata=None):
        """Generate with vLLM and automatically apply TTL heuristics.
        
        Args:
            prompt: Input text
            sampling_params: vLLM SamplingParams
            metadata: Dict with keys like perplexity, time_variance
                     If not provided, defaults are used.
        
        Returns:
            Generated text output
        """
        if self.llm is None:
            raise RuntimeError("LLM not set. Call set_llm() or pass to __init__")
        
        # Default metadata if not provided
        if metadata is None:
            metadata = {
                "perplexity": 10.0,
                "time_variance": 0.1
            }
        
        # Track access count for this prompt
        prompt_hash = hash(prompt)
        self.access_count[prompt_hash] = self.access_count.get(prompt_hash, 0) + 1
        metadata["access_count"] = self.access_count[prompt_hash]
        
        # Generate (LMCache automatically stores KV tensors)
        outputs = self.llm.generate([prompt], sampling_params)
        output_text = outputs[0].outputs[0].text
        
        # After generation, apply TTL heuristics and move KV if needed
        self._apply_heuristics_and_move(prompt, metadata)
        
        return output_text

    def _apply_heuristics_and_move(self, prompt, metadata):
        """Query LMCache, compute TTL/backend heuristics, and move KV if needed.
        
        This is called automatically after each generation.
        """
        try:
            # Step 1: Tokenize and lookup current KV location
            tokens = self.controller.tokenize(prompt)
            current_layout = self.controller.lookup(tokens)
            
            if not current_layout.get("found"):
                # KV not yet cached (shouldn't happen after generate, but handle gracefully)
                return
            
            current_backend = current_layout.get("lmcache_default_instance", [None])[0]
            
            # Step 2: Compute TTL and preferred backend based on heuristics
            ttl = compute_ttl(metadata)
            preferred_backend = select_backend(metadata)
            
            # Step 3: Move KV to preferred backend if different
            if current_backend and preferred_backend and current_backend != preferred_backend:
                move_result = self.controller.move(
                    old_position=["lmcache_instance", current_backend],
                    new_position=["lmcache_instance", preferred_backend]
                )
                print(f"✓ Moved KV from {current_backend} to {preferred_backend} (TTL: {ttl}s)")
                return move_result
            
            # Just log the decision if no move needed
            print(f"✓ KV in {preferred_backend}, TTL: {ttl}s")
            
        except Exception as e:
            # Gracefully handle LMCache server unavailability
            print(f"⚠️  Heuristic/move failed: {e}")
            pass
    

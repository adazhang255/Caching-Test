import time
import pickle
import os
from typing import Optional

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

 # TODO: maybe get rid of class altogether?
class MultiTierCache:
    def __init__(self, tiers):
        """
        tiers: list of tuples (backend_instance, ttl_seconds)
               highest-priority first
        """
        self.tiers = tiers

   #TODO: definitely set to cache controller instead
    def get(self, key, llm_fn=None):
        """Retrieve value from tiers, optionally computing via LLM if missing"""
        for i, (backend, ttl) in enumerate(self.tiers):
            val = backend.get(key)
            if val is not None:
                # Promote to higher tiers
                for j in range(i):
                    self.tiers[j][0].set(key, val, self.tiers[j][1])
                return val

        # Cache miss: compute if llm_fn is provided
        if llm_fn is not None:
            result = llm_fn(key)
            # Store in all tiers
            for backend, ttl in self.tiers:
                backend.set(key, result, ttl)
            return result
        return None

    def set(self, key, value):
        """Set value in all tiers with respective TTLs"""
        for backend, ttl in self.tiers:
            backend.set(key, value, ttl)

    def delete(self, key):
        for backend, _ in self.tiers:
            backend.delete(key)


# -----------------------------
# Example Usage with LLM
# -----------------------------

#TODO: import GEmma 3 279m, and change things about it
# Simulated LLM function
def fake_llm(prompt):
    # In reality, this would call vLLM
    return f"LLM response to: {prompt}"

# Initialize tiers
# TODO: LMCache initialzie tiers via config, get rid of here
hot = InMemoryBackend()
warm = LocalDiskBackend()
cold = RemoteBackend()

cache = MultiTierCache([
    (hot, 10),        # 10s TTL for hot
    (warm, 60),       # 1 min TTL for warm
    (cold, 3600)      # 1 hr TTL for cold
])

# Test flow
prompt = "What is the weather today?"
print("First query (compute):", cache.get(prompt, llm_fn=fake_llm))
time.sleep(2)
print("Second query (should hit hot):", cache.get(prompt))
time.sleep(12)
print("Third query (hot expired, should hit warm):", cache.get(prompt))


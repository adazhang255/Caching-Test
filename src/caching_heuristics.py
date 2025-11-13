import time
from cache_controller import LMCacheController 

class Prompt:
    def __init__(self, key, output_size, perplexity, timestamp=None, version=1):
        self.key = key
        self.output_size = output_size
        self.perplexity = perplexity
        self.timestamp = timestamp or time.time()
        self.version = version
        self.access_history = []

    # --- Access Tracking --- 
    def accessed(self):
        """Record an access timestamp"""
        self.access_history.append(time.time())

    def access_count_last_hour(self):
        """Return how many times this prompt was accessed in the last hour"""
        cutoff = time.time() - 3600
        self.access_history = [t for t in self.access_history if t >= cutoff]
        return len(self.access_history)

# --- Heuristic Functions ---

def get_TTL(prompt, perplexity, time_variant=False):
    if (perplexity < 40):
        return {"TTL":60, "backend": "GPU"}
    elif (time_variant or perplexity < 40):
        return {"TTL": 600, "backend": "disk"}
    else:
        return {"TTL": 60 * 24, "backend": "redis"}
    
def set_TTL(prompt, TTL, new_backend):

    controller = LMCacheController()
    tokens = controller.tokenize(prompt)
    curr_backend = controller.lookup(tokens)["lmcache_default_instance"][0]
    
    controller.move(["lmcache_instance", curr_backend], ["lmcache_instance", new_backend])
    # TODO: instead call redis/backend functionality
    new_backend.expire(TTL)
    

def is_hot_candidate(prompt, active_session=True, ttl_limit_minutes=10):
    """
    Hot cache heuristic:
    - Frequent, time-variant prompts
    - Active user session
    - TTL < 10 min
    """
    return active_session and (time.time() - prompt.timestamp < ttl_limit_minutes * 60)

def is_warm_candidate(prompt, min_accesses=2, min_output_size=100, perplexity_range=(20, 40)):
    """
    Warm cache heuristic:
    - Prompt accessed ≥2x in past hour
    - Output > threshold
    - Moderate perplexity (20-40)
    """
    return (prompt.access_count_last_hour() >= min_accesses and
            prompt.output_size >= min_output_size and
            perplexity_range[0] <= prompt.perplexity <= perplexity_range[1])

def is_long_term_candidate(prompt, deterministic=True, time_invariant=True):
    """
    Long-term cache heuristic:
    - Deterministic and static
    - No expected update triggers
    """
    return deterministic and time_invariant

# --- Example Usage ---
prompt1 = Prompt("last_week_bill", output_size=50, perplexity=25)
prompt1.accessed()
prompt1.accessed()  # accessed twice in last hour

print("Hot candidate:", is_hot_candidate(prompt1, active_session=True))
print("Warm candidate:", is_warm_candidate(prompt1))
print("Long-term candidate:", is_long_term_candidate(prompt1))

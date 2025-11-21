# Add metrics to track cache behavior
import time

class CacheMetrics:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.total_time_with_cache = 0
        self.total_time_without_cache = 0
    
    def record_generation(self, duration, was_cached):
        if was_cached:
            self.hits += 1
            self.total_time_with_cache += duration
        else:
            self.misses += 1
            self.total_time_without_cache += duration
    
    def report(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        print(f"Cache Hit Rate: {hit_rate:.2%}")
        print(f"Avg time (cached): {self.total_time_with_cache/max(self.hits,1):.2f}s")
        print(f"Avg time (uncached): {self.total_time_without_cache/max(self.misses,1):.2f}s")

metrics = CacheMetrics()
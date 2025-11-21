"""
LMCache Multi-Tier Storage Configuration with TTL
Supports Hot (GPU) -> Warm (CPU/Disk) -> Cold (S3/Redis) architecture
"""

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.connector import CreateStorageConnector
import torch

# =============================================================================
# Configuration 1: S3-Compatible Storage (e.g., AWS S3, ECS) as Cold Tier
# =============================================================================

def configure_s3_backend():
    """
    Three-tier setup with S3 as cold storage
    Hot: GPU Memory -> Warm: CPU Memory -> Cold: S3
    """
    
    config = LMCacheEngineConfig(
        # Chunk configuration for KV cache
        chunk_size=256,  # Number of tokens per chunk
        
        # Local storage (Hot + Warm tiers)
        local_device="cuda",  # Hot cache on GPU
        local_size=4 * 1024 * 1024 * 1024,  # 4GB GPU cache
        
        # Enable CPU offloading for warm tier
        enable_cpu_cache=True,
        cpu_cache_size=16 * 1024 * 1024 * 1024,  # 16GB CPU cache
        
        # Remote storage (Cold tier) - S3
        remote_url="s3://your-bucket-name/lmcache-kv-store",
        remote_backend="s3",
        
        # S3-specific configuration
        s3_config={
            "region": "us-east-1",
            "access_key_id": "YOUR_ACCESS_KEY",  # Use env vars in production
            "secret_access_key": "YOUR_SECRET_KEY",
            "endpoint_url": None,  # Use None for AWS S3, custom URL for ECS/MinIO
        },
        
        # TTL Settings (in seconds)
        local_ttl=3600,      # Hot cache: 1 hour
        cpu_ttl=7200,        # Warm cache: 2 hours  
        remote_ttl=86400,    # Cold cache: 24 hours
        
        # Eviction policies
        eviction_policy="lru",  # LRU for automatic tier management
        
        # Performance tuning
        prefetch_size=2,  # Number of chunks to prefetch
        async_offload=True,  # Asynchronous offloading to remote
    )
    
    return config


def configure_s3_ecs_backend():
    """
    Configuration for Dell ECS (S3-compatible object storage)
    """
    
    config = LMCacheEngineConfig(
        chunk_size=256,
        
        # Hot tier (GPU)
        local_device="cuda",
        local_size=4 * 1024 * 1024 * 1024,
        
        # Warm tier (CPU)
        enable_cpu_cache=True,
        cpu_cache_size=16 * 1024 * 1024 * 1024,
        
        # Cold tier (ECS)
        remote_url="s3://lmcache-bucket/kv-cache",
        remote_backend="s3",
        
        # ECS-specific S3 configuration
        s3_config={
            "endpoint_url": "https://ecs.example.com:9021",  # ECS endpoint
            "region": "us-east-1",  # ECS region
            "access_key_id": "ECS_ACCESS_KEY",
            "secret_access_key": "ECS_SECRET_KEY",
            "use_ssl": True,
            "verify": True,  # SSL certificate verification
            
            # ECS-specific optimizations
            "s3": {
                "addressing_style": "path",  # or "virtual" depending on ECS setup
                "signature_version": "s3v4",
            },
            
            # Connection pooling
            "max_pool_connections": 50,
        },
        
        eviction_policy="lru",
        async_offload=True,
    )
    
    return config


# =============================================================================
# Configuration 2: Redis as Cold Tier
# =============================================================================

def configure_redis_backend():
    """
    Three-tier setup with Redis as cold storage
    Hot: GPU Memory -> Warm: CPU Memory -> Cold: Redis
    """
    
    config = LMCacheEngineConfig(
        chunk_size=256,
        
        # Hot tier (GPU)
        local_device="cuda",
        local_size=4 * 1024 * 1024 * 1024,
        
        # Warm tier (CPU)
        enable_cpu_cache=True,
        cpu_cache_size=16 * 1024 * 1024 * 1024,
        
        # Cold tier (Redis)
        remote_url="redis://localhost:6379/0",
        remote_backend="redis",
        
        # Redis-specific configuration
        redis_config={
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": "your_redis_password",  # If authentication enabled
            
            # Connection pooling
            "max_connections": 50,
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
            
            # Redis cluster support (if using Redis Cluster)
            "cluster_enabled": False,
            # "startup_nodes": [{"host": "redis1", "port": 6379}],
            
            # Persistence options
            "decode_responses": False,  # Keep binary for KV cache data
        },
        
        # TTL configuration - Redis native TTL support
        local_ttl=1800,      # 30 minutes
        cpu_ttl=3600,        # 1 hour
        remote_ttl=86400,    # 24 hours (Redis will auto-expire)
        
        eviction_policy="lru",
        async_offload=True,
    )
    
    return config


def configure_redis_cluster_backend():
    """
    Redis Cluster configuration for distributed cold storage
    """
    
    config = LMCacheEngineConfig(
        chunk_size=256,
        
        # Hot tier
        local_device="cuda",
        local_size=4 * 1024 * 1024 * 1024,
        
        # Warm tier
        enable_cpu_cache=True,
        cpu_cache_size=16 * 1024 * 1024 * 1024,
        
        # Cold tier (Redis Cluster)
        remote_url="redis://redis-cluster:6379/0",
        remote_backend="redis",
        
        redis_config={
            "cluster_enabled": True,
            "startup_nodes": [
                {"host": "redis-node-1", "port": 6379},
                {"host": "redis-node-2", "port": 6379},
                {"host": "redis-node-3", "port": 6379},
            ],
            "max_connections": 100,
            "socket_timeout": 5,
            "decode_responses": False,
            
            # Cluster-specific settings
            "skip_full_coverage_check": False,
            "max_connections_per_node": 20,
        },
        
        # TTL settings
        local_ttl=1800,
        cpu_ttl=3600,
        remote_ttl=86400,
        
        eviction_policy="lru",
        async_offload=True,
    )
    
    return config


# =============================================================================
# Usage Example: Initializing LMCache Engine
# =============================================================================

def initialize_lmcache_engine(config):
    """
    Initialize LMCache engine with the specified configuration
    """
    from lmcache.cache_engine import LMCacheEngine
    
    metadata = LMCacheEngineMetadata(
        model_name="meta-llama/Llama-2-7b-hf",
        world_size=1,
        worker_id=0,
    )
    
    engine = LMCacheEngine(config, metadata)
    return engine


# =============================================================================
# TTL Management and Manual Control
# =============================================================================

class TTLManager:
    """
    Helper class for managing TTL across different storage tiers
    """
    
    def __init__(self, engine):
        self.engine = engine
    
    def set_dynamic_ttl(self, tier, ttl_seconds):
        """
        Dynamically adjust TTL for a specific tier
        
        Args:
            tier: "local", "cpu", or "remote"
            ttl_seconds: Time-to-live in seconds
        """
        if tier == "local":
            self.engine.config.local_ttl = ttl_seconds
        elif tier == "cpu":
            self.engine.config.cpu_ttl = ttl_seconds
        elif tier == "remote":
            self.engine.config.remote_ttl = ttl_seconds
        
        # Notify storage backend of TTL change
        self.engine.storage_backend.update_ttl(tier, ttl_seconds)
    
    def get_tier_stats(self):
        """
        Get statistics about each tier
        """
        return {
            "local": {
                "size": self.engine.local_cache.size(),
                "hit_rate": self.engine.local_cache.hit_rate(),
                "ttl": self.engine.config.local_ttl,
            },
            "cpu": {
                "size": self.engine.cpu_cache.size(),
                "hit_rate": self.engine.cpu_cache.hit_rate(),
                "ttl": self.engine.config.cpu_ttl,
            },
            "remote": {
                "size": self.engine.remote_cache.size(),
                "hit_rate": self.engine.remote_cache.hit_rate(),
                "ttl": self.engine.config.remote_ttl,
            },
        }
    
    def force_eviction(self, tier, key=None):
        """
        Manually evict entries from a specific tier
        
        Args:
            tier: "local", "cpu", or "remote"
            key: Specific key to evict, or None to evict based on policy
        """
        if tier == "local":
            cache = self.engine.local_cache
        elif tier == "cpu":
            cache = self.engine.cpu_cache
        elif tier == "remote":
            cache = self.engine.remote_cache
        
        if key:
            cache.remove(key)
        else:
            cache.evict_lru()  # Evict least recently used


# =============================================================================
# Advanced: Custom Eviction Policies
# =============================================================================

class CustomEvictionPolicy:
    """
    Custom eviction policy that considers both recency and frequency
    """
    
    def __init__(self, recency_weight=0.7, frequency_weight=0.3):
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.access_counts = {}
        self.last_access = {}
    
    def on_access(self, key):
        """Track access patterns"""
        import time
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        self.last_access[key] = time.time()
    
    def get_eviction_candidate(self, candidates):
        """
        Select the best candidate for eviction based on weighted score
        """
        import time
        current_time = time.time()
        
        scores = {}
        for key in candidates:
            recency_score = current_time - self.last_access.get(key, 0)
            frequency_score = 1.0 / (self.access_counts.get(key, 1))
            
            scores[key] = (
                self.recency_weight * recency_score +
                self.frequency_weight * frequency_score
            )
        
        # Return key with highest score (oldest + least frequent)
        return max(scores, key=scores.get)


# =============================================================================
# Example: Complete Setup
# =============================================================================

if __name__ == "__main__":
    # Choose your backend configuration
    # config = configure_s3_backend()
    # config = configure_s3_ecs_backend()
    config = configure_redis_backend()
    # config = configure_redis_cluster_backend()
    
    # Initialize engine
    engine = initialize_lmcache_engine(config)
    
    # Setup TTL manager
    ttl_manager = TTLManager(engine)
    
    # Example: Adjust TTL dynamically based on usage patterns
    stats = ttl_manager.get_tier_stats()
    if stats["remote"]["hit_rate"] > 0.8:
        # If remote cache is hit frequently, increase its TTL
        ttl_manager.set_dynamic_ttl("remote", 172800)  # 48 hours
    
    print("LMCache engine initialized with multi-tier storage and TTL")
    print(f"Configuration: {config.remote_backend} backend")
    print(f"TTL settings: Local={config.local_ttl}s, "
          f"CPU={config.cpu_ttl}s, Remote={config.remote_ttl}s")
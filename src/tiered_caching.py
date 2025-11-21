import time
import pickle
import os
import sys
import traceback
import math
from typing import Optional, Dict, Any

from caching_heuristics import compute_ttl, select_backend
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
        
        
        
        
        
        print("[DEBUG][TieredCache] generate_and_manage() start", file=sys.stderr)
        print("[DEBUG][TieredCache] prompt_len:", len(prompt), file=sys.stderr)
        print("[DEBUG][TieredCache] sampling_params:", sampling_params, file=sys.stderr)

        try:
            eng = getattr(self.llm, "llm_engine", None)
            print("[DEBUG][TieredCache] llm_engine present:", eng is not None, file=sys.stderr)
            if eng is not None:
                core = getattr(eng, "engine_core", None)
                print("[DEBUG][TieredCache] engine_core type:", type(core), file=sys.stderr)
                res = getattr(core, "resources", None)
                print("[DEBUG][TieredCache] engine resources:", res, file=sys.stderr)
        except Exception:
            print("[DEBUG][TieredCache] error introspecting engine:", file=sys.stderr)
            traceback.print_exc()

        try:
            outputs = self.llm.generate([prompt], sampling_params)
        except Exception as e:
            print("[DEBUG][TieredCache] EXCEPTION in self.llm.generate:", repr(e), file=sys.stderr)
            traceback.print_exc()
            # Optionally re-check engine_dead flag if reachable:
            try:
                eng = getattr(self.llm, "llm_engine", None)
                core = getattr(eng, "engine_core", None) if eng is not None else None
                print("[DEBUG][TieredCache] engine_dead after exception:",
                    getattr(core, "resources", None).engine_dead
                    if core is not None and getattr(core, "resources", None) is not None
                    else "unknown",
                    file=sys.stderr)
            except Exception:
                traceback.print_exc()
            raise  # re-raise so your existing error handling still runs

        print("[DEBUG][TieredCache] generate() returned", file=sys.stderr)
        output_text = outputs[0].outputs[0].text







        # Generate (LMCache automatically stores KV tensors)
        outputs = self.llm.generate([prompt], sampling_params)
        output_text = outputs[0].outputs[0].text
        
        # After generation, apply TTL heuristics and move KV if needed.
        self._apply_heuristics_and_move(prompt, metadata)
        
        return output_text

    def _apply_heuristics_and_move(self, prompt, metadata):
        """Query LMCache, compute TTL/backend heuristics, and move KV if needed.
        
        Remote tier is disabled; moves are between 'gpu' and 'disk'.
        """
        try:
            # Step 1: Tokenize and lookup current KV location
            tokens = self.controller.tokenize(prompt)
            current_layout = self.controller.lookup(tokens)
            
            if not current_layout.get("found"):
                # KV not yet cached (shouldn't happen after generate, but handle gracefully)
                return
            
            current_backend = current_layout.get("lmcache_default_instance", [None])[0]
            
            # Step 2: Compute TTL and preferred backend based on heuristics (gpu/disk only)
            ttl = compute_ttl(metadata)
            preferred_backend = select_backend(metadata)
            
            # Respect env guard to avoid orchestrated moves
            if os.getenv("LMCACHE_DISABLE_OFFLOAD", "0") == "1":
                print(f"✓ GPU-only mode. TTL: {ttl}s (no orchestrated move)")
                return

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
    

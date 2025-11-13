# test_inference.py
import time
from vllm import LLM, SamplingParams


# 1. Define the model ID
model_id = "TheBloke/deepseek-coder-6.7B-instruct-GPTQ"

# 2. Define the LMCache configuration
kv_cache_config = {
    "kv_connector": "LMCacheConnectorV1",
    "kv_role": "kv_both"
}

# 3. Initialize the vLLM engine with LMCache
print("Loading model...")
llm = LLM(
    model=model_id,
    kv_transfer_config=kv_cache_config,
    gpu_memory_utilization=0.8,
    dtype="float16", # Explicitly set dtype to float16, as bfloat16 is not supported by T4
    quantization="gptq" # Explicitly disable quantization to avoid mxfp4
)
print("Model loaded.")

# 4. Define sampling parameters
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

# 5. Define prompts to test caching
prompts = [
    ("What is the capital of France?"),

    ("What is the capital of France?")
]

# --- Run Generations ---

# Run the first prompt (will be slower and populate the cache)
print("\n--- Running first prompt (populating cache) ---")
start_time = time.time()
outputs = llm.generate([prompts[0]], sampling_params)
end_time = time.time()

print(f"Time taken: {end_time - start_time:.2f} seconds")
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {generated_text}\n")


# Run the second prompt (will be faster due to cached prefix)
print("\n--- Running second prompt (using cache) ---")
start_time = time.time()
outputs = llm.generate([prompts[1]], sampling_params)
end_time = time.time()

print(f"Time taken: {end_time - start_time:.2f} seconds")
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {generated_text}\n")
import requests

class LMCacheController:
    def __init__(self, host="localhost", port=9000, model = "gemma-3-270m"):
        self.url = f"http://{host}:{port}"
        self.vllm_url = f"http://{host}:{8000}"
        self.model = model

    def lookup(self, tokens):
        resp = requests.post(f"{self.url}/lookup", json={"tokens": tokens})
        resp.raise_for_status()
        return resp.json()
    
    def tokenize(self, prompt):
        resp = requests.post(f"{self.vllm_url}/tokenize", json={"model": self.model, "prompt": prompt})
        resp.raise_for_status()
        return resp.json()
    
    def move(self, old_position, new_position):
        resp = requests.post(f"{self.url}/move", json={"old_position": old_position, "new_position": new_position})
        resp.raise_for_status()
        return resp.json()

# Usage
controller = LMCacheController()
tokens = [128000, 849, 21435, 279]
layout_info = controller.lookup(tokens)
print(layout_info)
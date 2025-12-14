from typing import List, Optional

class KVCache:
    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.caches: List[Optional[dict]] = [None] * n_layers

    def get(self, layer_idx: int) -> Optional[dict]:
        return self.caches[layer_idx]

    def update(self, layer_idx: int, cache: dict):
        self.caches[layer_idx] = cache

    def reset(self):
        """Clear all cached keys/values (e.g. for new conversation)"""
        self.caches = [None] * self.n_layers

    def is_empty(self) -> bool:
        return all(c is None for c in self.caches)
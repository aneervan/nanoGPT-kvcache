from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch


KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


class KVCacheStore:
    def __init__(self) -> None:
        self._store: Dict[Tuple[int, ...], KVCache] = {}

    def _detach_to_cpu(self, kvcache: KVCache) -> KVCache:
        return [(k.detach().cpu(), v.detach().cpu()) for k, v in kvcache]

    def add(self, token_ids: Iterable[int], kvcache: KVCache) -> Tuple[int, ...]:
        key = tuple(token_ids)
        self._store[key] = self._detach_to_cpu(kvcache)
        return key

    def get(self, token_ids: Iterable[int], device: torch.device | str) -> Optional[KVCache]:
        key = tuple(token_ids)
        cached = self._store.get(key)
        if cached is None:
            return None
        return [(k.to(device), v.to(device)) for k, v in cached]

    def __contains__(self, token_ids: Iterable[int]) -> bool:
        return tuple(token_ids) in self._store

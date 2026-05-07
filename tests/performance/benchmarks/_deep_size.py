"""Deep-size helper for benchmark tests.

Exposes ``deep_sizeof(obj)`` which walks nested container contents (dict
keys/values, list/tuple/set/frozenset items) and sums their allocated
``sys.getsizeof`` footprints. This is required for cache-memory NFR gates
(e.g., ``test_cache_memory_bounded``) because ``sys.getsizeof(dict)`` alone
returns only the shallow allocation and underestimates real usage by 3-5x
for dict-of-frozenset payloads.

Originally lived inline in ``test_token_expansion.py``; promoted to a shared
helper when ADR-008 added cache-memory assertions that needed the same
measurement without duplicating the walker.
"""

from __future__ import annotations

import sys
from typing import Any


def deep_sizeof(obj: Any, _seen: set[int] | None = None) -> int:
    """Recursive ``sys.getsizeof`` — accounts for nested container contents."""
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return 0
    _seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_sizeof(k, _seen) + deep_sizeof(v, _seen) for k, v in obj.items())
    elif isinstance(obj, list | tuple | set | frozenset):
        size += sum(deep_sizeof(item, _seen) for item in obj)
    return size

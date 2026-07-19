"""Schema-9 router seam for guided full-pipeline planning.

The production endpoint lands in the next task.  Keeping its router separate
lets that controller be added without moving any signed handlers in
``guided.py``.
"""

from fastapi import APIRouter

router = APIRouter()

__all__ = ["router"]

"""Composer rules for elspeth-lints."""

from __future__ import annotations

from elspeth_lints.rules.composer.catch_order import RULE as CATCH_ORDER_RULE
from elspeth_lints.rules.composer.exception_channel import RULE as EXCEPTION_CHANNEL_RULE

__all__ = ["CATCH_ORDER_RULE", "EXCEPTION_CHANNEL_RULE"]

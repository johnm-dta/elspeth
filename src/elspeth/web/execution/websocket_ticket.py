"""Short-lived opaque tickets for execution-progress WebSocket auth."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Lock
from uuid import UUID

from elspeth.web.auth.models import UserIdentity

_TICKET_BYTES = 32


@dataclass(frozen=True, slots=True)
class WebSocketTicket:
    """Opaque, single-use credential bound to one run and user."""

    ticket: str
    run_id: str
    user: UserIdentity
    expires_at: datetime


class WebSocketTicketStore:
    """Process-local store for short-lived, single-use WebSocket tickets."""

    def __init__(self, ttl_seconds: int = 30) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl = timedelta(seconds=ttl_seconds)
        self._tickets: dict[str, WebSocketTicket] = {}
        self._lock = Lock()

    def issue(
        self,
        *,
        run_id: str | UUID,
        user: UserIdentity,
        now: datetime | None = None,
    ) -> WebSocketTicket:
        """Issue a new ticket for one WebSocket connection attempt."""
        issued_at = self._now(now)
        expires_at = issued_at + self._ttl
        with self._lock:
            self._purge_expired_locked(issued_at)
            ticket_value = self._new_ticket_value()
            while ticket_value in self._tickets:
                ticket_value = self._new_ticket_value()
            ticket = WebSocketTicket(
                ticket=ticket_value,
                run_id=str(run_id),
                user=user,
                expires_at=expires_at,
            )
            self._tickets[ticket.ticket] = ticket
        return ticket

    def consume(
        self,
        *,
        ticket: str,
        run_id: str | UUID,
        now: datetime | None = None,
    ) -> UserIdentity | None:
        """Consume a ticket and return its bound user when valid."""
        if not ticket.strip():
            return None
        checked_at = self._now(now)
        with self._lock:
            self._purge_expired_locked(checked_at)
            record = self._tickets.pop(ticket, None)
        if record is None:
            return None
        if record.expires_at <= checked_at:
            return None
        if record.run_id != str(run_id):
            return None
        return record.user

    @staticmethod
    def _new_ticket_value() -> str:
        return secrets.token_urlsafe(_TICKET_BYTES)

    @staticmethod
    def _now(now: datetime | None) -> datetime:
        if now is None:
            return datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        return now.astimezone(UTC)

    def _purge_expired_locked(self, now: datetime) -> None:
        expired = [ticket for ticket, record in self._tickets.items() if record.expires_at <= now]
        for ticket in expired:
            del self._tickets[ticket]

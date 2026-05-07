"""Freeze-contract regression tests for the blob exception family.

Pattern parity with ``tests/unit/web/composer/test_service.py``'s
convergence/plugin-crash freeze tests: every declared attribute MUST
be frozen after construction, and the exception-chain machinery
(``__cause__`` / ``__context__`` / ``__suppress_context__`` /
``__notes__``) MUST remain writable so ``raise X from Y`` and
``add_note()`` continue to work.

See ``src/elspeth/web/blobs/protocol.py`` — the block comment above
``_guard_frozen_attr`` explains why the blob family intentionally
omits the ``capture()`` classmethod gateway used by the composer
family.  These tests do NOT assert ``capture()`` exists.
"""

from __future__ import annotations

import pytest

from elspeth.web.blobs.protocol import (
    BlobActiveRunError,
    BlobContentMissingError,
    BlobError,
    BlobIntegrityError,
    BlobNotFoundError,
    BlobQuotaExceededError,
    BlobStateError,
)


class TestBlobNotFoundErrorFreezeContract:
    def test_declared_attrs_frozen_after_construction(self) -> None:
        exc = BlobNotFoundError("blob-123")
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.blob_id = "other-blob"  # type: ignore[misc]

    def test_exception_chain_machinery_remains_writable(self) -> None:
        """__cause__, __context__, __suppress_context__, __notes__ must all work.

        Freezing these would break ``raise X from Y`` and ``add_note()``
        which is why the freeze guard only targets declared ``_FROZEN_ATTRS``.
        """
        root = RuntimeError("underlying")
        exc = BlobNotFoundError("blob-xyz")
        exc.__cause__ = root
        exc.__suppress_context__ = True
        exc.add_note("operator triage hint")

        assert exc.__cause__ is root
        assert exc.__suppress_context__ is True
        assert "operator triage hint" in exc.__notes__


class TestBlobActiveRunErrorFreezeContract:
    def test_declared_attrs_frozen_after_construction(self) -> None:
        exc = BlobActiveRunError("blob-1", run_id="run-1")
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.blob_id = "other-blob"  # type: ignore[misc]
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.run_id = "other-run"  # type: ignore[misc]

    def test_secondary_args_keyword_only(self) -> None:
        """``run_id`` MUST be keyword-only at the raise site.

        The composer family uses positional first-arg + keyword-only
        secondary-arg for self-documenting raise sites (``raise
        BlobActiveRunError(blob_id, run_id=run)`` tells the reader
        which identifier is the subject and which is context).  A
        positional ``run_id`` regression would weaken that contract.
        """
        with pytest.raises(TypeError):
            BlobActiveRunError("blob-1", "run-1")  # type: ignore[misc]


class TestBlobQuotaExceededErrorFreezeContract:
    def test_declared_attrs_frozen_after_construction(self) -> None:
        exc = BlobQuotaExceededError("sess-1", current_bytes=100, limit_bytes=50)
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.session_id = "other"  # type: ignore[misc]
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.current_bytes = 999  # type: ignore[misc]
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.limit_bytes = 0  # type: ignore[misc]

    def test_secondary_args_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            BlobQuotaExceededError("sess-1", 100, 50)  # type: ignore[misc]


class TestBlobStateErrorFreezeContract:
    def test_declared_attrs_frozen_after_construction(self) -> None:
        exc = BlobStateError("blob-1", message="status wrong")
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.blob_id = "other"  # type: ignore[misc]

    def test_message_arg_keyword_only(self) -> None:
        """``message`` is keyword-only to prevent positional confusion.

        Legacy raise sites sometimes pass a positional second arg that
        looks like a blob identifier but is actually the message body;
        keyword-only forces the distinction at the call site.
        """
        with pytest.raises(TypeError):
            BlobStateError("blob-1", "status wrong")  # type: ignore[misc]


class TestBlobIntegrityErrorFreezeContract:
    def test_declared_attrs_frozen_after_construction(self) -> None:
        exc = BlobIntegrityError("blob-1", expected="a" * 64, actual="b" * 64)
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.blob_id = "other"  # type: ignore[misc]
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.expected_hash = "c" * 64  # type: ignore[misc]
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.actual_hash = "d" * 64  # type: ignore[misc]

    def test_secondary_args_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            BlobIntegrityError("blob-1", "a" * 64, "b" * 64)  # type: ignore[misc]


class TestBlobContentMissingErrorFreezeContract:
    def test_declared_attrs_frozen_after_construction(self) -> None:
        exc = BlobContentMissingError("blob-1", storage_path="/tmp/blob.csv")
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.blob_id = "other"  # type: ignore[misc]
        with pytest.raises(AttributeError, match="frozen after construction"):
            exc.storage_path = "/tmp/other.csv"  # type: ignore[misc]

    def test_storage_path_arg_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            BlobContentMissingError("blob-1", "/tmp/blob.csv")  # type: ignore[misc]


class TestBlobExceptionFamilyInvariants:
    """Family-level invariants that must hold across every subclass."""

    _FAMILY: tuple[type[Exception], ...] = (
        BlobNotFoundError,
        BlobActiveRunError,
        BlobQuotaExceededError,
        BlobStateError,
        BlobIntegrityError,
        BlobContentMissingError,
    )

    def test_every_family_member_inherits_blob_error(self) -> None:
        for cls in self._FAMILY:
            assert issubclass(cls, BlobError), f"{cls.__name__} must inherit BlobError for family-wide catches."

    def test_every_family_member_declares_frozen_attrs(self) -> None:
        """Every exception in the family MUST declare ``_FROZEN_ATTRS``.

        Drift guard: a new blob exception added without the guard
        would inherit ``object.__setattr__`` and silently lack the
        freeze contract. The family invariant prevents that from
        slipping through review.
        """
        for cls in self._FAMILY:
            attrs = getattr(cls, "_FROZEN_ATTRS", None)
            assert isinstance(attrs, frozenset), (
                f"{cls.__name__} must declare ``_FROZEN_ATTRS: ClassVar[frozenset[str]]`` "
                "to match the blob exception family freeze contract. See "
                "blobs/protocol.py::_guard_frozen_attr for the shared guard."
            )
            assert len(attrs) > 0, (
                f"{cls.__name__}._FROZEN_ATTRS is empty — if the exception carries no "
                "payload it should still declare at least one frozen attribute "
                "(typically ``blob_id``) for audit correlation."
            )

    def test_every_family_member_has_overridden_setattr(self) -> None:
        """Every exception MUST override ``__setattr__`` with the freeze guard.

        Declaring ``_FROZEN_ATTRS`` without the ``__setattr__``
        override would make the declaration decorative — attribute
        reassignment would still succeed.  This test pins that both
        halves of the contract are present.
        """
        for cls in self._FAMILY:
            own_setattr = cls.__dict__.get("__setattr__")
            assert own_setattr is not None, (
                f"{cls.__name__} must define its own ``__setattr__`` (delegating to "
                "``_guard_frozen_attr``) to enforce the ``_FROZEN_ATTRS`` contract."
            )

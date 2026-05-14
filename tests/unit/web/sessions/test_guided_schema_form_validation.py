from __future__ import annotations

import pytest
from fastapi import HTTPException

from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.sessions.routes import _reject_hidden_field_submissions


def test_blob_ref_field_rejects_invalid_uuid_string() -> None:
    knobs = {
        "fields": [
            {
                "name": "source_blob",
                "label": "Source blob",
                "kind": "blob-ref",
                "required": True,
                "nullable": False,
            }
        ]
    }

    with pytest.raises(HTTPException) as exc_info:
        _reject_hidden_field_submissions(
            knobs,
            {"source_blob": "not-a-uuid"},
            recorder=BufferingRecorder(),
            composition_version=1,
            actor="alice",
            session_id="session-1",
            plugin_kind="recipe",
            plugin_name="example",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "invalid_blob_ref"
    assert exc_info.value.detail["field"] == "source_blob"

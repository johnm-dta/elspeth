"""Property tests for the text sink byte contract."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from elspeth.plugins.sinks.text_sink import TextSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory


def _config(path: Path) -> dict[str, object]:
    return {"path": str(path), "field": "line_text", "encoding": "utf-8", "mode": "write", "schema": {"mode": "observed"}}


@given(
    st.lists(
        st.text(alphabet=st.characters(blacklist_characters="\r\n", blacklist_categories=("Cs",))),
        min_size=1,
        max_size=50,
    )
)
def test_text_sink_bytes_and_hash_match_all_lines(values: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "out.txt"
        sink = inject_write_failure(TextSink(_config(path)))
        result = sink.write(
            [{"line_text": value} for value in values],
            make_context(landscape=make_factory().plugin_audit_writer()),
        )
        sink.close()

        expected = "".join(f"{value}\n" for value in values).encode()
        assert path.read_bytes() == expected
        assert result.artifact.content_hash == hashlib.sha256(expected).hexdigest()


@given(st.one_of(st.integers(), st.booleans(), st.none(), st.binary(max_size=20), st.text().map(lambda value: value + "\nsecret")))
def test_rejected_values_never_appear_in_output(value: object) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "out.txt"
        sink = inject_write_failure(TextSink(_config(path)))
        result = sink.write(
            [{"line_text": "safe"}, {"line_text": value}],
            make_context(landscape=make_factory().plugin_audit_writer()),
        )

        assert path.read_bytes() == b"safe\n"
        assert len(result.diversions) == 1


@given(st.lists(st.none(), max_size=20))
def test_empty_or_all_diverted_batches_do_not_create_target(values: list[None]) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "out.txt"
        sink = inject_write_failure(TextSink(_config(path)))
        rows = [{"line_text": value} for value in values]
        result = sink.write(rows, make_context(landscape=make_factory().plugin_audit_writer()))

        assert not path.exists()
        assert result.artifact.size_bytes == 0
        assert result.artifact.content_hash == hashlib.sha256(b"").hexdigest()

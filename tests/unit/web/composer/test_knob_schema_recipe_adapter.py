import typing

import pytest

from elspeth.contracts.composer_slots import SlotSpec, SlotType
from elspeth.web.catalog.knob_schema import lower_slot_specs_to_knob_schema


def test_blob_id_slot_lowers_to_blob_ref():
    slots = {"source_blob": SlotSpec(slot_type="blob_id", required=True, description="Source CSV")}
    ks = lower_slot_specs_to_knob_schema(slots)
    f = ks["fields"][0]
    assert f["kind"] == "blob-ref"
    assert f["required"] is True


def test_str_list_slot_lowers_to_string_list():
    slots = {"keys": SlotSpec(slot_type="str_list", required=False, description="Merge keys")}
    ks = lower_slot_specs_to_knob_schema(slots)
    f = ks["fields"][0]
    assert f["kind"] == "string-list"
    assert f["item_kind"] == "text"


@pytest.mark.parametrize("slot_type", typing.get_args(SlotType))
def test_every_slot_type_has_a_mapping(slot_type):
    """A new SlotType member fails this totality test instead of runtime."""
    slots = {"x": SlotSpec(slot_type=slot_type, required=False, description="X")}
    ks = lower_slot_specs_to_knob_schema(slots)
    assert len(ks["fields"]) == 1
    assert "kind" in ks["fields"][0]

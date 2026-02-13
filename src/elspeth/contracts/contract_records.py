"""Audit record types for schema contracts.

These types bridge SchemaContract (runtime) to Landscape storage (JSON serialization).
The pattern:
- Runtime: SchemaContract with Python types
- Storage: ContractAuditRecord with string type names
- Restore: to_schema_contract() converts back with integrity check
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from elspeth.contracts.errors import (
    ContractViolation,
    ExtraFieldViolation,
    MissingFieldViolation,
    TypeMismatchViolation,
)
from elspeth.contracts.type_normalization import CONTRACT_TYPE_MAP

if TYPE_CHECKING:
    from elspeth.contracts.schema_contract import FieldContract, SchemaContract


@dataclass(frozen=True, slots=True)
class FieldAuditRecord:
    """Audit record for a single field in a schema contract.

    Immutable after creation - stores type information as strings
    for JSON serialization in the Landscape audit trail.

    Attributes:
        normalized_name: Dict key / Python identifier (e.g., "important_data")
        original_name: Display name from source (e.g., "'Important - Data !!'")
        python_type: Python type name as string (e.g., "int", "str")
        required: Whether field must be present in row
        source: "declared" (from config) or "inferred" (from first row observation)
    """

    normalized_name: str
    original_name: str
    python_type: str
    required: bool
    source: Literal["declared", "inferred"]

    @classmethod
    def from_field_contract(cls, fc: FieldContract) -> FieldAuditRecord:
        """Create FieldAuditRecord from FieldContract.

        Converts the Python type to its string name for serialization.

        Args:
            fc: The FieldContract to convert

        Returns:
            FieldAuditRecord with type name as string
        """
        return cls(
            normalized_name=fc.normalized_name,
            original_name=fc.original_name,
            python_type=fc.python_type.__name__,
            required=fc.required,
            source=fc.source,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dict suitable for JSON serialization
        """
        return {
            "normalized_name": self.normalized_name,
            "original_name": self.original_name,
            "python_type": self.python_type,
            "required": self.required,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ContractAuditRecord:
    """Audit record for a schema contract.

    Stores schema contract information in a format suitable for
    JSON serialization in the Landscape audit trail.

    Attributes:
        mode: Schema enforcement mode (FIXED, FLEXIBLE, OBSERVED)
        locked: True after first row processed (types frozen)
        version_hash: Deterministic hash for integrity verification
        fields: Immutable tuple of FieldAuditRecord instances
    """

    mode: Literal["FIXED", "FLEXIBLE", "OBSERVED"]
    locked: bool
    version_hash: str
    fields: tuple[FieldAuditRecord, ...]

    @classmethod
    def from_contract(cls, contract: SchemaContract) -> ContractAuditRecord:
        """Create ContractAuditRecord from SchemaContract.

        Args:
            contract: The SchemaContract to convert

        Returns:
            ContractAuditRecord suitable for JSON serialization
        """
        sorted_fields = sorted(contract.fields, key=lambda fc: fc.normalized_name)
        return cls(
            mode=contract.mode,
            locked=contract.locked,
            version_hash=contract.version_hash(),
            fields=tuple(FieldAuditRecord.from_field_contract(fc) for fc in sorted_fields),
        )

    def to_json(self) -> str:
        """Serialize to canonical JSON.

        Uses canonical_json for deterministic serialization,
        ensuring identical contracts produce identical JSON.

        Returns:
            Canonical JSON string
        """
        from elspeth.core.canonical import canonical_json

        # Canonicalize field order so semantically equivalent contracts serialize
        # identically regardless of upstream insertion/merge ordering.
        sorted_fields = sorted(self.fields, key=lambda f: f.normalized_name)
        data = {
            "mode": self.mode,
            "locked": self.locked,
            "version_hash": self.version_hash,
            "fields": [f.to_dict() for f in sorted_fields],
        }
        return canonical_json(data)

    @classmethod
    def from_json(cls, json_str: str) -> ContractAuditRecord:
        """Restore ContractAuditRecord from JSON string.

        Args:
            json_str: JSON string from to_json()

        Returns:
            Restored ContractAuditRecord
        """
        data = json.loads(json_str)
        return cls(
            mode=data["mode"],
            locked=data["locked"],
            version_hash=data["version_hash"],
            fields=tuple(
                FieldAuditRecord(
                    normalized_name=f["normalized_name"],
                    original_name=f["original_name"],
                    python_type=f["python_type"],
                    required=f["required"],
                    source=f["source"],
                )
                for f in data["fields"]
            ),
        )

    def to_schema_contract(self) -> SchemaContract:
        """Convert back to SchemaContract with integrity verification.

        Verifies the restored contract's hash matches the stored hash
        to ensure audit integrity (Tier 1 requirement).

        Returns:
            Restored SchemaContract

        Raises:
            KeyError: If a field has an unknown python_type
            ValueError: If hash verification fails (integrity violation)
        """
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        fields = tuple(
            FieldContract(
                normalized_name=f.normalized_name,
                original_name=f.original_name,
                python_type=CONTRACT_TYPE_MAP[f.python_type],  # KeyError on unknown = correct!
                required=f.required,
                source=f.source,
            )
            for f in self.fields
        )

        contract = SchemaContract(
            mode=self.mode,
            fields=fields,
            locked=self.locked,
        )

        # Verify integrity (Tier 1 audit requirement)
        actual_hash = contract.version_hash()
        if actual_hash != self.version_hash:
            raise ValueError(
                f"Contract integrity violation: hash mismatch. "
                f"Expected {self.version_hash}, got {actual_hash}. "
                f"Audit record may be corrupted or from different version."
            )

        return contract


@dataclass(frozen=True, slots=True)
class ValidationErrorWithContract:
    """Audit record for a schema validation error.

    Captures contract violation details in a format suitable for
    the Landscape audit trail.

    Attributes:
        violation_type: Type of violation (type_mismatch, missing_field, extra_field)
        normalized_field_name: Internal field name used by code
        original_field_name: Original field name from external data
        expected_type: Expected type name (for type_mismatch), None otherwise
        actual_type: Actual type name (for type_mismatch), None otherwise
    """

    violation_type: Literal["type_mismatch", "missing_field", "extra_field"]
    normalized_field_name: str
    original_field_name: str
    expected_type: str | None
    actual_type: str | None

    @classmethod
    def from_violation(cls, violation: ContractViolation) -> ValidationErrorWithContract:
        """Create ValidationErrorWithContract from ContractViolation.

        Args:
            violation: The ContractViolation to convert

        Returns:
            ValidationErrorWithContract with violation details

        Raises:
            ValueError: If violation is of an unknown type
        """
        if isinstance(violation, TypeMismatchViolation):
            return cls(
                violation_type="type_mismatch",
                normalized_field_name=violation.normalized_name,
                original_field_name=violation.original_name,
                expected_type=violation.expected_type.__name__,
                actual_type=violation.actual_type.__name__,
            )
        elif isinstance(violation, MissingFieldViolation):
            return cls(
                violation_type="missing_field",
                normalized_field_name=violation.normalized_name,
                original_field_name=violation.original_name,
                expected_type=None,
                actual_type=None,
            )
        elif isinstance(violation, ExtraFieldViolation):
            return cls(
                violation_type="extra_field",
                normalized_field_name=violation.normalized_name,
                original_field_name=violation.original_name,
                expected_type=None,
                actual_type=None,
            )
        else:
            raise ValueError(f"Unknown violation type: {type(violation).__name__}")

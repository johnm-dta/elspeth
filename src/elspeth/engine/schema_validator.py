# src/elspeth/engine/schema_validator.py
"""Optional pipeline schema validation.

Checks that plugin schemas are compatible across pipeline stages.
This is opt-in validation - the pipeline still runs without it.
"""

from __future__ import annotations

from elspeth.plugins.schemas import PluginSchema


def validate_pipeline_schemas(
    source_output: type[PluginSchema] | None,
    transform_inputs: list[type[PluginSchema] | None],
    transform_outputs: list[type[PluginSchema] | None],
    sink_inputs: list[type[PluginSchema] | None],
) -> list[str]:
    """Validate schema compatibility across pipeline stages.

    Checks that each consumer's required fields are provided by its producer.
    This catches configuration errors before the pipeline runs.

    The validation flow:
    1. Source output -> First transform input
    2. Transform[i] output -> Transform[i+1] input (for each adjacent pair)
    3. Final transform output -> Each sink input

    If there are no transforms, source output is validated directly against sink inputs.

    Args:
        source_output: Schema of source output (None = dynamic/skip validation)
        transform_inputs: Schemas of transform inputs (in order)
        transform_outputs: Schemas of transform outputs (in order)
        sink_inputs: Schemas of sink inputs

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    # Skip validation if source schema is None (dynamic schema)
    if source_output is None:
        return errors

    # Determine the "current producer" schema - starts with source output
    current_producer: type[PluginSchema] = source_output

    # Check source -> first transform
    if transform_inputs and transform_inputs[0] is not None:
        missing = _get_missing_required_fields(current_producer, transform_inputs[0])
        if missing:
            errors.append(
                f"Source output missing fields required by transform[0]: {missing}"
            )

    # Check transform chain
    for i in range(len(transform_outputs)):
        # Update current producer to this transform's output (if known)
        transform_output = transform_outputs[i]
        if transform_output is not None:
            current_producer = transform_output

        # Check against next transform's input (if there is one)
        if i + 1 < len(transform_inputs):
            next_input = transform_inputs[i + 1]
            if next_input is not None:
                missing = _get_missing_required_fields(current_producer, next_input)
                if missing:
                    errors.append(
                        f"Transform[{i}] output missing fields required by transform[{i + 1}]: {missing}"
                    )

    # Check final transform -> sinks
    # If there are transforms, use last transform output; otherwise use source output
    final_producer: type[PluginSchema] = source_output
    if transform_outputs:
        last_output = transform_outputs[-1]
        final_producer = last_output if last_output is not None else current_producer

    for j, sink_input in enumerate(sink_inputs):
        if sink_input is None:
            continue
        missing = _get_missing_required_fields(final_producer, sink_input)
        if missing:
            errors.append(
                f"Final transform output missing fields required by sink[{j}]: {missing}"
            )

    return errors


def _get_missing_required_fields(
    producer: type[PluginSchema],
    consumer: type[PluginSchema],
) -> set[str]:
    """Get required fields in consumer that are missing from producer.

    Args:
        producer: Schema that produces data
        consumer: Schema that consumes data

    Returns:
        Set of field names that are required by consumer but not in producer
    """
    producer_fields = set(producer.model_fields.keys())
    required_fields = {
        name for name, field in consumer.model_fields.items() if field.is_required()
    }
    return required_fields - producer_fields

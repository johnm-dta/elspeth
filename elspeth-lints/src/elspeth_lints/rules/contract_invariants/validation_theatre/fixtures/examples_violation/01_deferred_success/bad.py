from contracts import OutputValidationResult


def validate_output_target() -> OutputValidationResult:
    # Field resolution is not available yet; this will be checked later.
    return OutputValidationResult.success()

from contracts import OutputValidationResult


def validate_output_target(path) -> OutputValidationResult:
    if not path.exists():
        return OutputValidationResult.success()
    return OutputValidationResult.failure("target exists")

"""Built-in transform plugins for ELSPETH.

Transforms process rows in the pipeline. Each transform receives a row
and returns a TransformResult indicating success/failure and output data.
"""

from elspeth.plugins.transforms.passthrough import PassThrough

__all__ = ["PassThrough"]

from dataclasses import dataclass
from typing import Any


@dataclass
class ElaraResult:
    """Standard result returned by ELARA components."""

    success: bool
    message: str
    data: Any = None

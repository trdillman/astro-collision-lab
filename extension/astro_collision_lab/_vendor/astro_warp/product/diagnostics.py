"""The finding value type shared by the compiler's error and advisory channels.

Separate from compiler.py because model.py and the runtime also produce findings and must
not import the compiler. ADR 0037 section 6 reads this channel for the HUD.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    ERROR = "error"        # blocks compilation
    ADVISORY = "advisory"  # surfaced, never blocks


@dataclass(frozen=True)
class Finding:
    severity: Severity
    code: str      # stable identifier, e.g. "mass-contrast-float32"
    message: str   # actionable text naming the observed value and the remedy

    def __str__(self) -> str:
        return self.message

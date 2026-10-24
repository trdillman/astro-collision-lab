"""Idempotent lifecycle accounting used by tests and Blender glue."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegistrationLedger:
    """Tracks registration calls without tying tests to Blender APIs."""

    timer_registered: bool = False
    draw_registered: bool = False
    timer_register_calls: int = 0
    draw_register_calls: int = 0
    timer_unregister_calls: int = 0
    draw_unregister_calls: int = 0

    def register_timer(self) -> bool:
        if self.timer_registered:
            return False
        self.timer_registered = True
        self.timer_register_calls += 1
        return True

    def unregister_timer(self) -> bool:
        if not self.timer_registered:
            return False
        self.timer_registered = False
        self.timer_unregister_calls += 1
        return True

    def register_draw(self) -> bool:
        if self.draw_registered:
            return False
        self.draw_registered = True
        self.draw_register_calls += 1
        return True

    def unregister_draw(self) -> bool:
        if not self.draw_registered:
            return False
        self.draw_registered = False
        self.draw_unregister_calls += 1
        return True

    @property
    def duplicate_timer_registrations(self) -> int:
        return max(0, self.timer_register_calls - 1)

    @property
    def duplicate_draw_registrations(self) -> int:
        return max(0, self.draw_register_calls - 1)

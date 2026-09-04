"""Suppressing repeat sightings of the same plate.

Two different duplicates exist, and they need different mechanisms:

*Within one pass of a vehicle*, a plate is visible for dozens of frames.
That is Phase 6's job — track the vehicle and emit one voted result per
track — and it is not what this module does.

*Across passes*, the same car returns: it circles a car park, waits at a
barrier, or the tracker loses and re-acquires it. No tracker can suppress
that, because the second sighting genuinely is a new track. A cooldown
window does, and that is what lives here.

The window is a judgement call, not a constant of nature. Too short and a
car idling at a barrier fills the log; too long and a vehicle that legally
re-enters goes unrecorded. The default of five minutes suits car-park and
gate footage; a motorway camera would want far less.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta

DEFAULT_COOLDOWN = timedelta(minutes=5)

# A long run must not grow memory without bound. The cap is on distinct
# plates held, and eviction is oldest-first — an entry old enough to be
# evicted is almost always past its cooldown anyway.
DEFAULT_MAX_TRACKED = 10_000


@dataclass(frozen=True)
class Suppression:
    """Why a sighting was suppressed, for the run summary."""

    plate: str
    first_seen: datetime
    last_seen: datetime
    count: int


class Deduplicator:
    """Accepts a plate the first time, then suppresses it for `cooldown`.

    The cooldown restarts on each *suppressed* sighting, so a car sitting in
    view for twenty minutes produces one row rather than one row per window.
    """

    def __init__(
        self,
        cooldown: timedelta = DEFAULT_COOLDOWN,
        *,
        max_tracked: int = DEFAULT_MAX_TRACKED,
    ) -> None:
        if cooldown < timedelta(0):
            raise ValueError(f"cooldown must not be negative, got {cooldown}")
        if max_tracked < 1:
            raise ValueError(f"max_tracked must be at least 1, got {max_tracked}")

        self.cooldown = cooldown
        self.max_tracked = max_tracked
        # Ordered by last sighting, so eviction is a popitem from the front.
        self._seen: OrderedDict[str, Suppression] = OrderedDict()
        self.suppressed_count = 0

    def accept(self, plate: str, when: datetime) -> bool:
        """Should this sighting be logged?

        Args:
            plate: the normalized plate string.
            when: sighting time. Must be consistent in awareness across calls —
                mixing naive and aware datetimes raises from the comparison.

        Returns:
            True the first time a plate is seen and again once its cooldown
            has elapsed; False while it is still within the window.
        """
        previous = self._seen.get(plate)

        if previous is not None and when - previous.last_seen < self.cooldown:
            # Refresh the window: a vehicle continuously in view should not
            # produce a second row the moment the first cooldown expires.
            self._seen[plate] = Suppression(
                plate=plate,
                first_seen=previous.first_seen,
                last_seen=when,
                count=previous.count + 1,
            )
            self._seen.move_to_end(plate)
            self.suppressed_count += 1
            return False

        self._seen[plate] = Suppression(
            plate=plate,
            first_seen=previous.first_seen if previous else when,
            last_seen=when,
            count=1 if previous is None else previous.count + 1,
        )
        self._seen.move_to_end(plate)
        self._evict()
        return True

    def _evict(self) -> None:
        while len(self._seen) > self.max_tracked:
            self._seen.popitem(last=False)

    def tracked(self) -> int:
        return len(self._seen)

    def history(self, plate: str) -> Suppression | None:
        return self._seen.get(plate)

    def reset(self) -> None:
        self._seen.clear()
        self.suppressed_count = 0

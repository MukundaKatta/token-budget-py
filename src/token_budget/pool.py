"""Core BudgetPool implementation."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


class BudgetExceeded(Exception):
    """Raised when a record() or reserve() would push past a cap.

    Attributes:
        axis: which axis was breached, "tokens" or "usd"
        attempted: the candidate value (current + delta) that would breach
        cap: the configured cap on that axis
    """

    def __init__(self, axis: str, attempted: float, cap: float):
        self.axis = axis
        self.attempted = attempted
        self.cap = cap
        super().__init__(f"budget exceeded on {axis}: attempted {attempted:.6g} > cap {cap:.6g}")


@dataclass(frozen=True)
class BudgetSnapshot:
    """Point-in-time view of pool usage. Returned by `BudgetPool.snapshot()`."""

    tokens_used: int
    usd_used: float
    token_cap: int | None
    usd_cap: float | None
    tokens_remaining: int | None
    usd_remaining: float | None


class BudgetPool:
    """Thread-safe shared token + USD budget.

    Either axis can be left as None (unbounded). Both axes are tracked
    independently: a record breaches if EITHER cap would be exceeded.

    Methods:
      * `record(tokens, usd)` - commit consumption; raise BudgetExceeded on breach.
      * `reserve(tokens, usd)` - context manager for two-phase commit.
      * `try_reserve` / `release` - manual two-phase commit.
      * `snapshot()` - read current totals atomically.
      * `reset()` - zero both axes (useful in tests; also for time-window rollovers).
    """

    def __init__(
        self,
        token_cap: int | None = None,
        usd_cap: float | None = None,
    ) -> None:
        if token_cap is not None and token_cap < 0:
            raise ValueError("token_cap must be >= 0 or None")
        if usd_cap is not None and usd_cap < 0:
            raise ValueError("usd_cap must be >= 0 or None")
        self._token_cap = token_cap
        self._usd_cap = usd_cap
        self._tokens = 0
        self._usd = 0.0
        # also count reserved-but-not-committed
        self._tokens_reserved = 0
        self._usd_reserved = 0.0
        # bumped on reset() so reservations made before a reset become inert
        # (committing/releasing them must not corrupt the post-reset counters)
        self._generation = 0
        self._lock = threading.Lock()

    # ---- single-phase ----

    def record(self, *, tokens: int = 0, usd: float = 0.0) -> None:
        """Commit `tokens` and `usd` to the pool. Raises BudgetExceeded if
        either axis would breach. On breach, no change is applied."""
        if tokens < 0 or usd < 0:
            raise ValueError("record amounts must be non-negative")
        with self._lock:
            self._check_capacity(tokens, usd)
            self._tokens += tokens
            self._usd += usd

    # ---- two-phase ----

    def try_reserve(self, *, tokens: int = 0, usd: float = 0.0) -> Reservation:
        """Atomically check capacity and earmark the amount. Caller must
        either `commit()` (final usage may differ) or `release()` the
        reservation. Raises BudgetExceeded on breach."""
        if tokens < 0 or usd < 0:
            raise ValueError("reserve amounts must be non-negative")
        with self._lock:
            self._check_capacity(tokens, usd)
            self._tokens_reserved += tokens
            self._usd_reserved += usd
            generation = self._generation
        return Reservation(_pool=self, _tokens=tokens, _usd=usd, _generation=generation)

    @contextmanager
    def reserve(self, *, tokens: int = 0, usd: float = 0.0) -> Iterator[Reservation]:
        """Context manager wrapping try_reserve/release.

        If the `with` block exits without calling reservation.commit(),
        the reservation is auto-released (refunded). If commit() was called,
        no release happens on exit.
        """
        r = self.try_reserve(tokens=tokens, usd=usd)
        try:
            yield r
        finally:
            if not r._completed:
                self._release_reservation(r)

    def _commit_reservation(self, r: Reservation, actual_tokens: int, actual_usd: float) -> None:
        if r._completed:
            raise RuntimeError("reservation already committed or released")
        if actual_tokens < 0 or actual_usd < 0:
            raise ValueError("actual amounts must be non-negative")
        with self._lock:
            if r._generation != self._generation:
                # The pool was reset() after this reservation was made; its hold
                # has already been dropped. Don't double-subtract the reserved
                # counters (that would drive them negative). Still record the
                # actual usage against the current generation, subject to caps.
                try:
                    self._check_capacity(actual_tokens, actual_usd)
                except BudgetExceeded:
                    r._completed = True
                    raise
                self._tokens += actual_tokens
                self._usd += actual_usd
                r._completed = True
                return
            # release the reservation first
            self._tokens_reserved -= r._tokens
            self._usd_reserved -= r._usd
            # check capacity for the actual commit amount. if it breaches,
            # mark the reservation completed so the context manager doesn't
            # try to release again - we already released above.
            try:
                self._check_capacity(actual_tokens, actual_usd)
            except BudgetExceeded:
                r._completed = True
                raise
            self._tokens += actual_tokens
            self._usd += actual_usd
            r._completed = True

    def _release_reservation(self, r: Reservation) -> None:
        if r._completed:
            return
        with self._lock:
            if r._generation == self._generation:
                # Only refund if the hold is still live. After a reset() the
                # reserved counters were already cleared, so subtracting again
                # would drive them negative.
                self._tokens_reserved -= r._tokens
                self._usd_reserved -= r._usd
            r._completed = True

    # ---- introspection ----

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                tokens_used=self._tokens,
                usd_used=self._usd,
                token_cap=self._token_cap,
                usd_cap=self._usd_cap,
                tokens_remaining=(
                    self._token_cap - self._tokens - self._tokens_reserved
                    if self._token_cap is not None
                    else None
                ),
                usd_remaining=(
                    self._usd_cap - self._usd - self._usd_reserved
                    if self._usd_cap is not None
                    else None
                ),
            )

    def reset(self) -> None:
        """Zero both axes. Drops any outstanding reservations too.

        Reservations created before the reset are invalidated: a later
        commit() on one records its actual usage against the fresh window,
        and a later release() is a harmless no-op. Either way the reserved
        counters stay consistent.
        """
        with self._lock:
            self._tokens = 0
            self._usd = 0.0
            self._tokens_reserved = 0
            self._usd_reserved = 0.0
            self._generation += 1

    # ---- internal ----

    def _check_capacity(self, tokens: int, usd: float) -> None:
        """Caller must hold the lock."""
        if self._token_cap is not None:
            attempted = self._tokens + self._tokens_reserved + tokens
            if attempted > self._token_cap:
                raise BudgetExceeded("tokens", attempted, self._token_cap)
        if self._usd_cap is not None:
            attempted_usd = self._usd + self._usd_reserved + usd
            if attempted_usd > self._usd_cap:
                raise BudgetExceeded("usd", attempted_usd, self._usd_cap)


class Reservation:
    """An outstanding reserve()'d slice of the budget.

    Call `commit(tokens=..., usd=...)` with the actual amounts after the
    LLM call returns, or let the `with pool.reserve(...)` block end without
    committing to refund the reservation.
    """

    __slots__ = ("_pool", "_tokens", "_usd", "_generation", "_completed")

    def __init__(self, _pool: BudgetPool, _tokens: int, _usd: float, _generation: int = 0) -> None:
        self._pool = _pool
        self._tokens = _tokens
        self._usd = _usd
        self._generation = _generation
        self._completed = False

    def commit(self, *, tokens: int | None = None, usd: float | None = None) -> None:
        """Commit the reservation with the actual usage. Pass None for an
        axis to use the reserved amount unchanged.

        If actual > reserved on a capped axis, the extra is checked against
        capacity and may raise BudgetExceeded - but the reservation is still
        consumed (released) in that case to avoid orphaning the slot."""
        actual_tokens = self._tokens if tokens is None else tokens
        actual_usd = self._usd if usd is None else usd
        self._pool._commit_reservation(self, actual_tokens, actual_usd)

    def release(self) -> None:
        """Refund the reservation. Safe to call multiple times - no-op after
        the first."""
        self._pool._release_reservation(self)

    @property
    def reserved_tokens(self) -> int:
        return self._tokens

    @property
    def reserved_usd(self) -> float:
        return self._usd

    @property
    def completed(self) -> bool:
        return self._completed

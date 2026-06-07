import threading

import pytest

from token_budget import BudgetExceeded, BudgetPool, BudgetSnapshot

# ---------- basic recording ----------


def test_unbounded_pool_accepts_anything():
    p = BudgetPool()
    p.record(tokens=10**9, usd=10**9)
    snap = p.snapshot()
    assert snap.tokens_used == 10**9
    assert snap.usd_used == 10**9
    assert snap.tokens_remaining is None
    assert snap.usd_remaining is None


def test_record_under_cap():
    p = BudgetPool(token_cap=100, usd_cap=1.0)
    p.record(tokens=50, usd=0.25)
    snap = p.snapshot()
    assert snap.tokens_used == 50
    assert snap.tokens_remaining == 50
    assert snap.usd_remaining == pytest.approx(0.75)


def test_record_raises_on_token_breach():
    p = BudgetPool(token_cap=100)
    p.record(tokens=80)
    with pytest.raises(BudgetExceeded) as exc:
        p.record(tokens=30)
    assert exc.value.axis == "tokens"
    assert exc.value.cap == 100
    assert exc.value.attempted == 110
    # state unchanged
    assert p.snapshot().tokens_used == 80


def test_record_raises_on_usd_breach():
    p = BudgetPool(usd_cap=1.0)
    p.record(usd=0.9)
    with pytest.raises(BudgetExceeded) as exc:
        p.record(usd=0.2)
    assert exc.value.axis == "usd"
    assert p.snapshot().usd_used == pytest.approx(0.9)


def test_either_axis_alone_works():
    only_tokens = BudgetPool(token_cap=100)
    only_tokens.record(tokens=50, usd=1_000_000.0)  # usd unlimited
    only_usd = BudgetPool(usd_cap=1.0)
    only_usd.record(tokens=10**9, usd=0.5)


def test_record_rejects_negative():
    p = BudgetPool()
    with pytest.raises(ValueError):
        p.record(tokens=-1)
    with pytest.raises(ValueError):
        p.record(usd=-0.01)


def test_invalid_caps_raise():
    with pytest.raises(ValueError):
        BudgetPool(token_cap=-1)
    with pytest.raises(ValueError):
        BudgetPool(usd_cap=-1.0)


# ---------- reservations ----------


def test_reserve_then_commit_actual_amounts():
    p = BudgetPool(token_cap=1000, usd_cap=1.0)
    with p.reserve(tokens=500, usd=0.5) as r:
        # while reserved, remaining reflects the hold
        snap = p.snapshot()
        assert snap.tokens_remaining == 500
        assert snap.usd_remaining == pytest.approx(0.5)
        # commit with real numbers
        r.commit(tokens=420, usd=0.35)
    final = p.snapshot()
    assert final.tokens_used == 420
    assert final.usd_used == pytest.approx(0.35)
    assert final.tokens_remaining == 580


def test_reserve_auto_release_on_no_commit():
    p = BudgetPool(token_cap=100)
    with p.reserve(tokens=50) as r:
        assert r.reserved_tokens == 50
        # do NOT commit -> reservation should be released on block exit
    snap = p.snapshot()
    assert snap.tokens_used == 0
    assert snap.tokens_remaining == 100


def test_explicit_release():
    p = BudgetPool(token_cap=100)
    r = p.try_reserve(tokens=80)
    r.release()
    snap = p.snapshot()
    assert snap.tokens_remaining == 100
    # double release is safe
    r.release()


def test_try_reserve_raises_when_no_room():
    p = BudgetPool(token_cap=100)
    p.record(tokens=80)
    with pytest.raises(BudgetExceeded):
        p.try_reserve(tokens=30)


def test_reservation_counts_against_remaining():
    p = BudgetPool(token_cap=100)
    p.record(tokens=40)
    r = p.try_reserve(tokens=40)
    # 40 used + 40 reserved -> remaining = 20
    assert p.snapshot().tokens_remaining == 20
    with pytest.raises(BudgetExceeded):
        p.record(tokens=30)
    r.release()
    assert p.snapshot().tokens_remaining == 60


def test_commit_with_extra_above_reservation_breaches():
    p = BudgetPool(token_cap=100)
    with p.reserve(tokens=50) as r:
        # 50 reserved, but actual was 80, and we have only 100 total. fine.
        r.commit(tokens=80)
    assert p.snapshot().tokens_used == 80
    # now reserve again and overshoot
    with pytest.raises(BudgetExceeded), p.reserve(tokens=10) as r:
        r.commit(tokens=30)  # 80 already used + 30 = 110 > 100


def test_commit_passes_through_reserved_amounts_by_default():
    p = BudgetPool(token_cap=100, usd_cap=1.0)
    with p.reserve(tokens=10, usd=0.05) as r:
        r.commit()  # use reserved amounts as-is
    snap = p.snapshot()
    assert snap.tokens_used == 10
    assert snap.usd_used == pytest.approx(0.05)


def test_commit_after_completion_raises():
    p = BudgetPool(token_cap=100)
    r = p.try_reserve(tokens=10)
    r.commit()
    with pytest.raises(RuntimeError):
        r.commit()


# ---------- reset + snapshot ----------


def test_reset_zeroes_everything():
    p = BudgetPool(token_cap=100, usd_cap=1.0)
    p.record(tokens=50, usd=0.5)
    p.try_reserve(tokens=20)
    p.reset()
    snap = p.snapshot()
    assert snap.tokens_used == 0
    assert snap.usd_used == 0.0
    assert snap.tokens_remaining == 100


def test_commit_stale_reservation_after_reset_records_against_fresh_window():
    # Regression: reset() drops the reserved counters; committing a reservation
    # made before the reset must not double-subtract them (which used to drive
    # tokens_reserved negative and corrupt tokens_remaining).
    p = BudgetPool(token_cap=100)
    r = p.try_reserve(tokens=40)
    p.reset()
    r.commit(tokens=40)
    snap = p.snapshot()
    assert snap.tokens_used == 40
    assert snap.tokens_remaining == 60  # 100 - 40 used - 0 reserved


def test_release_stale_reservation_after_reset_is_noop():
    p = BudgetPool(token_cap=100)
    r = p.try_reserve(tokens=40)
    p.reset()
    r.release()  # must not push tokens_reserved negative
    snap = p.snapshot()
    assert snap.tokens_used == 0
    assert snap.tokens_remaining == 100


def test_reserve_context_auto_release_after_reset_is_clean():
    p = BudgetPool(token_cap=100)
    with p.reserve(tokens=40):
        p.reset()  # invalidates the live reservation mid-block
    snap = p.snapshot()
    assert snap.tokens_used == 0
    assert snap.tokens_remaining == 100


def test_stale_commit_still_respects_fresh_cap():
    p = BudgetPool(token_cap=100)
    r = p.try_reserve(tokens=40)
    p.reset()
    p.record(tokens=90)
    with pytest.raises(BudgetExceeded):
        r.commit(tokens=40)  # 90 + 40 = 130 > 100
    assert p.snapshot().tokens_used == 90


def test_snapshot_is_a_value_type():
    p = BudgetPool(token_cap=100)
    p.record(tokens=10)
    s = p.snapshot()
    assert isinstance(s, BudgetSnapshot)
    # mutating the pool does not mutate the snapshot
    p.record(tokens=10)
    assert s.tokens_used == 10


# ---------- concurrency ----------


def test_concurrent_records_do_not_overrun_cap():
    p = BudgetPool(token_cap=1000)
    successes = []
    failures = []

    def worker():
        try:
            p.record(tokens=10)
            successes.append(1)
        except BudgetExceeded:
            failures.append(1)

    threads = [threading.Thread(target=worker) for _ in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # exactly 100 of the 200 should have succeeded (1000 / 10)
    assert len(successes) == 100
    assert len(failures) == 100
    assert p.snapshot().tokens_used == 1000

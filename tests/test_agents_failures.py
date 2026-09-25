"""Specification for primer.agents.failures: why ambitious agents fail, and the fix for each failure."""

import pytest

from primer.agents.failures import (
    FAILURES,
    CircuitBreaker,
    CircuitOpen,
    chain_success,
    is_looping,
    max_steps_for,
    retry_with_backoff,
)


class TestCompoundingError:
    def test_given_ten_steps_each_95_percent_reliable_the_whole_task_succeeds_about_60_percent_of_the_time(self):
        # 0.95 ** 10 = 0.5987
        assert chain_success(0.95, 10) == pytest.approx(0.599, abs=1e-3)

    def test_given_twenty_such_steps_the_task_succeeds_about_36_percent_of_the_time(self):
        # 0.95 ** 20 = 0.3585
        assert chain_success(0.95, 20) == pytest.approx(0.358, abs=1e-3)

    def test_given_95_percent_steps_and_a_90_percent_target_at_most_two_steps_fit(self):
        # 0.95 ** 2 = 0.9025 clears 0.90; 0.95 ** 3 = 0.857 does not.
        assert max_steps_for(0.95, target=0.90) == 2


class FlakyService:
    """Fails with a timeout a set number of times, then answers."""

    def __init__(self, failures: int, error: type[Exception] = TimeoutError):
        self.failures, self.error, self.calls = failures, error, 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error("upstream timed out")
        return "ok"


class TestRetryWithBackoff:
    def test_given_two_timeouts_then_success_the_call_succeeds_on_the_third_attempt(self):
        svc = FlakyService(failures=2)
        assert retry_with_backoff(svc, sleep=lambda s: None) == "ok"
        assert svc.calls == 3

    def test_given_two_timeouts_the_waits_double_from_the_base_delay(self):
        waits = []
        retry_with_backoff(FlakyService(failures=2), base_delay=0.1, sleep=waits.append)
        assert waits == pytest.approx([0.1, 0.2])

    def test_given_many_failures_the_wait_never_exceeds_the_cap(self):
        waits = []
        with pytest.raises(TimeoutError):
            retry_with_backoff(FlakyService(failures=99), max_attempts=8, base_delay=1.0, max_delay=10.0, sleep=waits.append)
        assert max(waits) == 10.0

    def test_given_a_service_that_never_recovers_the_last_error_is_raised_after_the_final_attempt(self):
        svc = FlakyService(failures=99)
        with pytest.raises(TimeoutError):
            retry_with_backoff(svc, max_attempts=4, sleep=lambda s: None)
        assert svc.calls == 4

    def test_given_an_error_that_retrying_cannot_fix_it_is_raised_immediately(self):
        # A bad request fails the same way every time; retrying only adds load.
        svc = FlakyService(failures=99, error=ValueError)
        with pytest.raises(ValueError):
            retry_with_backoff(svc, sleep=lambda s: None)
        assert svc.calls == 1


class TestCircuitBreaker:
    def make(self):
        now = [0.0]
        return CircuitBreaker(failure_threshold=3, reset_timeout=30.0, clock=lambda: now[0]), now

    def failing(self):
        raise TimeoutError("down")

    def test_given_three_consecutive_failures_the_breaker_opens(self):
        breaker, _ = self.make()
        for _ in range(3):
            with pytest.raises(TimeoutError):
                breaker.call(self.failing)
        assert breaker.state == "open"

    def test_given_an_open_breaker_calls_fail_fast_without_touching_the_service(self):
        breaker, _ = self.make()
        for _ in range(3):
            with pytest.raises(TimeoutError):
                breaker.call(self.failing)
        svc = FlakyService(failures=0)
        with pytest.raises(CircuitOpen):
            breaker.call(svc)
        assert svc.calls == 0

    def test_given_the_reset_timeout_has_passed_one_trial_call_goes_through_and_success_closes_the_breaker(self):
        breaker, now = self.make()
        for _ in range(3):
            with pytest.raises(TimeoutError):
                breaker.call(self.failing)
        now[0] = 31.0
        assert breaker.call(FlakyService(failures=0)) == "ok"
        assert breaker.state == "closed"

    def test_given_the_trial_call_fails_the_breaker_opens_again(self):
        breaker, now = self.make()
        for _ in range(3):
            with pytest.raises(TimeoutError):
                breaker.call(self.failing)
        now[0] = 31.0
        with pytest.raises(TimeoutError):
            breaker.call(self.failing)
        assert breaker.state == "open"


class TestLoopDetection:
    def test_given_the_same_tool_called_with_the_same_arguments_three_times_in_a_row_the_agent_is_looping(self):
        calls = [("search", {"q": "vpn"}), ("search", {"q": "vpn"}), ("search", {"q": "vpn"})]
        assert is_looping(calls, repeats=3)

    def test_given_the_same_tool_with_different_arguments_the_agent_is_making_progress(self):
        calls = [("search", {"q": "vpn"}), ("search", {"q": "vpn error"}), ("search", {"q": "ERR-4012"})]
        assert not is_looping(calls, repeats=3)


class TestCatalogue:
    def test_given_the_catalogue_every_failure_names_a_symptom_a_fix_and_the_lesson_that_implements_it(self):
        assert all(f.symptom and f.fix and f.module.startswith("primer.") for f in FAILURES)

    def test_given_the_catalogue_it_covers_the_ten_classic_failures(self):
        assert len(FAILURES) == 10

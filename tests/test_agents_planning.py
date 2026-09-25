"""Specification for primer.agents.planning: long tasks succeed by being short, checked and resumable."""

import pytest

from primer.agents.planning import end_to_end_success


class TestCompoundingError:
    def test_given_steps_that_each_succeed_95_percent_of_the_time_ten_in_a_row_succeed_about_60_percent(self):
        # 0.95^10 = 0.5987
        assert end_to_end_success(0.95, 10) == pytest.approx(0.599, abs=1e-3)

    def test_given_steps_that_each_succeed_95_percent_of_the_time_twenty_in_a_row_succeed_about_36_percent(self):
        # 0.95^20 = 0.3585
        assert end_to_end_success(0.95, 20) == pytest.approx(0.358, abs=1e-3)


class TestVerificationAndRetries:
    def test_given_95_percent_steps_a_check_that_catches_every_failure_and_one_retry_make_each_step_99_75_percent(self):
        # 0.95 + 0.05 · 1.0 · 0.95 = 0.9975
        from primer.agents.planning import step_success

        assert step_success(0.95, detect=1.0, retries=1) == pytest.approx(0.9975)

    def test_given_perfect_checks_and_one_retry_ten_steps_succeed_about_97_5_percent_instead_of_60(self):
        # 0.9975^10 = 0.9753
        from primer.agents.planning import step_success

        assert end_to_end_success(step_success(0.95, detect=1.0, retries=1), 10) == pytest.approx(0.9753, abs=1e-4)

    def test_given_a_check_that_misses_half_the_failures_ten_steps_succeed_only_about_77_percent(self):
        # Retries only help failures you notice: (0.95 + 0.05 · 0.5 · 0.95)^10 = 0.97375^10 = 0.766
        from primer.agents.planning import step_success

        assert end_to_end_success(step_success(0.95, detect=0.5, retries=1), 10) == pytest.approx(0.766, abs=1e-3)


class TestCheckpoints:
    def test_given_checkpoints_finishing_ten_95_percent_steps_takes_about_10_5_step_runs(self):
        # Each step is retried on its own until it passes: 10 / 0.95 = 10.53
        from primer.agents.planning import expected_step_runs

        assert expected_step_runs(0.95, 10, checkpoints=True) == pytest.approx(10.53, abs=0.01)

    def test_given_no_checkpoints_a_failure_restarts_from_step_one_and_it_takes_about_13_4_step_runs(self):
        # (1 - 0.95^10) / (0.05 · 0.95^10) = 0.4013 / 0.02994 = 13.41
        from primer.agents.planning import expected_step_runs

        assert expected_step_runs(0.95, 10, checkpoints=False) == pytest.approx(13.41, abs=0.01)


class TestDecomposition:
    # Q3 data: INV-102 (Globex, 1200) was paid 1100; INV-104 (Umbrella, 450) was never paid.

    def test_given_the_q3_invoices_and_payments_the_reconciliation_lists_the_short_payment_and_the_unpaid_invoice(self):
        from primer.agents.planning import reconcile_q3, run_subtasks

        report = run_subtasks(reconcile_q3())
        assert report.outputs["list_mismatches"] == [
            {"invoice": "INV-102", "vendor": "Globex", "invoiced": 1200, "paid": 1100},
            {"invoice": "INV-104", "vendor": "Umbrella", "invoiced": 450, "paid": 0},
        ]

    def test_given_a_finished_reconciliation_the_summary_names_both_mismatches(self):
        from primer.agents.planning import reconcile_q3, run_subtasks

        assert run_subtasks(reconcile_q3()).outputs["draft_summary"] == (
            "Q3 reconciliation: 4 invoices, 3 payments, 2 mismatches (INV-102 short by 100, INV-104 unpaid)."
        )

    def test_given_a_payments_fetch_that_times_out_once_only_that_step_runs_again(self):
        from primer.agents.planning import PAYMENTS, reconcile_q3, run_subtasks

        calls = []

        def flaky_payments():
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("payments API timed out")
            return list(PAYMENTS)

        report = run_subtasks(reconcile_q3(fetch_payments=flaky_payments))
        assert report.attempts == {"fetch_invoices": 1, "fetch_payments": 2, "match": 1, "list_mismatches": 1, "draft_summary": 1}

    def test_given_a_step_that_keeps_failing_its_check_the_run_stops_there_and_later_steps_never_run(self):
        from primer.agents.planning import reconcile_q3, run_subtasks

        report = run_subtasks(reconcile_q3(fetch_payments=lambda: []), max_retries=2)
        assert (report.status, report.failed_step, report.problem) == ("failed", "fetch_payments", "no payments returned for Q3")
        assert report.attempts == {"fetch_invoices": 1, "fetch_payments": 3}


class TestPlanAndExecute:
    # The first plan calls the invoices API v1, which has been retired. The planner learns that only by trying.

    def test_given_a_step_that_fails_because_its_api_was_retired_the_planner_revises_the_plan_and_the_goal_is_met(self):
        from primer.agents.planning import demo_planner, plan_and_execute

        run = plan_and_execute(demo_planner(), "Reconcile Q3 invoices")
        assert (run.status, run.replans) == ("done", 1)

    def test_given_a_revised_plan_work_already_completed_is_not_repeated(self):
        from primer.agents.planning import demo_planner, plan_and_execute

        run = plan_and_execute(demo_planner(), "Reconcile Q3 invoices")
        assert run.executed == [
            ("fetch_payments", "ok"),
            ("fetch_invoices_v1", "failed: invoices API v1 was retired on 2026-07-01; use fetch_invoices_v2"),
            ("fetch_invoices_v2", "ok"),
            ("match", "ok"),
            ("draft_summary", "ok"),
        ]

    def test_given_a_planner_that_never_adapts_the_run_gives_up_after_the_replan_limit(self):
        from primer.agents.llm import ScriptedLLM
        from primer.agents.planning import plan_and_execute

        stubborn = ScriptedLLM(['{"steps": ["fetch_invoices_v1", "match"]}'])
        run = plan_and_execute(stubborn, "Reconcile Q3 invoices", max_replans=2)
        assert (run.status, run.replans) == ("failed", 2)


class TestReflectionVersusExternalChecks:
    # The first draft treats every year divisible by 4 as a leap year, which is wrong for 1900.

    def test_given_a_leap_year_function_that_forgets_century_years_the_models_self_review_approves_it(self):
        from primer.agents.planning import BUGGY_LEAP, confident_critic, self_review

        assert self_review(confident_critic(), BUGGY_LEAP) == (True, "Looks correct: years divisible by 4 are leap years.")

    def test_given_the_same_function_running_real_test_cases_catches_the_year_1900(self):
        from primer.agents.planning import BUGGY_LEAP, run_checks

        assert run_checks(BUGGY_LEAP) == ["is_leap(1900) returned True, expected False"]

    def test_given_the_corrected_function_every_test_case_passes(self):
        from primer.agents.planning import FIXED_LEAP, run_checks

        assert run_checks(FIXED_LEAP) == []

    def test_given_test_failures_fed_back_to_the_model_its_second_draft_passes(self):
        from primer.agents.planning import fixing_coder, generate_until_checks_pass

        code, attempts = generate_until_checks_pass(fixing_coder(), "Write is_leap(year).")
        assert attempts == 2


class TestFigures:
    def test_given_the_lesson_it_draws_compounding_error_the_effect_of_checks_and_the_cost_of_restarts(self):
        pytest.importorskip("matplotlib")
        from primer.agents.planning import figures

        assert sorted(figures()) == ["checkpoints", "compounding", "verification"]

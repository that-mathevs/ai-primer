"""Specification for primer.agents.cost: paying less per successful task without getting worse."""

import asyncio

import pytest

from primer.agents.cost import (
    CACHE_PAIRS,
    BudgetExceeded,
    ResponseCache,
    SemanticCache,
    TaskBudget,
    TenantSpend,
    batch_cost,
    cost_per_success_with_cleanup,
    cost_per_success_with_retries,
    five_x_plan,
    request_cost,
    route,
    routing_savings,
    run_parallel,
    run_sequential,
    semantic_cache_sweep,
)


class TestRequestPricing:
    # Illustrative prices: large model $5 in / $25 out per million tokens; cached reads at 10%.
    def test_given_10k_input_and_500_output_tokens_on_the_large_model_the_call_costs_6_25_cents(self):
        # 10,000 × 5 / 1M = 0.05; 500 × 25 / 1M = 0.0125.
        assert request_cost("large", 10_000, 500) == pytest.approx(0.0625)

    def test_given_8k_of_the_input_served_from_cache_the_same_call_costs_2_65_cents(self):
        # 8,000 × 0.5 / 1M + 2,000 × 5 / 1M + 0.0125 = 0.004 + 0.010 + 0.0125.
        assert request_cost("large", 10_000, 500, cached_tokens=8_000) == pytest.approx(0.0265)

    def test_given_the_batch_interface_the_same_call_costs_half(self):
        assert batch_cost([("large", 10_000, 500)]) == pytest.approx(0.03125)

    def test_given_equal_token_counts_output_costs_five_times_input(self):
        assert request_cost("large", 0, 1000) == pytest.approx(5 * request_cost("large", 1000, 0))


class TestModelRouting:
    def test_given_a_short_classification_task_it_goes_to_the_small_model(self):
        assert route("Classify this ticket as billing, technical or other: 'my card was charged twice'") == "small"

    def test_given_a_multi_step_planning_task_it_goes_to_the_large_model(self):
        assert route("Plan the migration of our billing system and analyze the risks of each step") == "large"

    def test_given_the_sample_workload_routing_cuts_spend_by_more_than_half(self):
        assert routing_savings()["savings"] > 0.5


class TestResponseCaching:
    def test_given_the_same_question_with_different_spacing_and_case_the_exact_cache_hits(self):
        cache = ResponseCache()
        cache.put("How do I reset my password?", "Use the self-service portal.")
        assert cache.get("  how do i reset my PASSWORD? ") == "Use the self-service portal."

    def test_given_a_paraphrase_the_exact_cache_misses(self):
        cache = ResponseCache()
        cache.put("How do I reset my password?", "Use the self-service portal.")
        assert cache.get("I forgot my password, how do I recover it?") is None


class TestSemanticCaching:
    def test_given_a_paraphrase_above_the_threshold_the_semantic_cache_returns_the_stored_answer(self):
        cache = SemanticCache(threshold=0.85)
        cache.put("How do I reset my password?", "Use the self-service portal.")
        assert cache.get("I forgot my password, how do I recover it?") == "Use the self-service portal."

    def test_given_a_different_question_about_the_same_topic_a_high_threshold_still_returns_the_wrong_answer(self):
        # "sick days" and "vacation days" share almost every concept: this is the semantic cache's core risk.
        cache = SemanticCache(threshold=0.95)
        cache.put("How many vacation days do I get?", "20 days of PTO per year.")
        assert cache.get("How many sick days do I get?") == "20 days of PTO per year."

    def test_given_different_error_codes_the_identifier_guard_prevents_a_wrong_hit(self):
        cache = SemanticCache(threshold=0.5)
        cache.put("What does ERR-4012 mean?", "The VPN tunnel failed.")
        assert cache.get("What does ERR-4013 mean?") is None

    def test_given_a_negated_request_the_negation_guard_prevents_a_wrong_hit(self):
        cache = SemanticCache(threshold=0.5)
        cache.put("Cancel my order", "Order cancelled.")
        assert cache.get("Don't cancel my order") is None

    def test_given_an_entry_older_than_its_time_to_live_it_is_not_served(self):
        now = [0.0]
        cache = SemanticCache(threshold=0.85, ttl_seconds=60, clock=lambda: now[0])
        cache.put("How do I reset my password?", "Use the self-service portal.")
        now[0] = 61
        assert cache.get("How do I reset my password?") is None

    def test_given_a_lower_threshold_more_wrong_answers_are_served(self):
        low, high = semantic_cache_sweep([0.6, 0.95])
        assert low["wrong_hit_rate"] > high["wrong_hit_rate"]

    def test_given_the_labelled_pairs_there_are_both_paraphrases_and_near_misses(self):
        assert {label for _, _, label in CACHE_PAIRS} == {True, False}


class TestParallelToolCalls:
    def test_given_three_independent_100ms_tools_running_them_together_takes_about_one_tool_time(self):
        seq = asyncio.run(run_sequential([0.1, 0.1, 0.1]))
        par = asyncio.run(run_parallel([0.1, 0.1, 0.1]))
        assert seq["seconds"] >= 0.3
        assert par["seconds"] < 0.2

    def test_given_parallel_calls_the_results_come_back_in_request_order(self):
        assert asyncio.run(run_parallel([0.03, 0.01, 0.02]))["results"] == ["tool-0", "tool-1", "tool-2"]


class TestBudgets:
    def test_given_a_three_step_budget_the_fourth_step_is_refused(self):
        budget = TaskBudget(max_steps=3, max_tokens=10_000)
        for _ in range(3):
            budget.charge(100)
        with pytest.raises(BudgetExceeded):
            budget.charge(100)

    def test_given_a_token_budget_a_step_that_would_exceed_it_is_refused(self):
        budget = TaskBudget(max_steps=10, max_tokens=1_000)
        budget.charge(900)
        with pytest.raises(BudgetExceeded):
            budget.charge(200)

    def test_given_a_tenant_over_its_monthly_cap_the_spend_tracker_raises_a_cap_alert(self):
        spend = TenantSpend(monthly_cap=10.0)
        spend.record("acme", cost=6.0, tokens=1000)
        assert "cap" in spend.record("acme", cost=5.0, tokens=1000)

    def test_given_a_task_using_ten_times_the_usual_tokens_the_tracker_raises_an_anomaly_alert(self):
        # A sudden jump in tokens per task usually means a loop or a bad deploy.
        spend = TenantSpend(monthly_cap=1_000.0)
        for tokens in [1000, 1100, 900, 1050, 950, 1000]:
            spend.record("acme", cost=0.01, tokens=tokens)
        assert "anomaly" in spend.record("acme", cost=0.1, tokens=10_000)


class TestUnitEconomics:
    def test_given_retries_until_success_a_cheap_model_that_succeeds_60_percent_costs_0_33_cents_per_success(self):
        # 0.002 / 0.60
        assert cost_per_success_with_retries(0.002, 0.60) == pytest.approx(0.003333, abs=1e-6)

    def test_given_human_cleanup_of_failures_the_cheap_model_costs_80_cents_per_task(self):
        # 0.002 + 0.40 × 2.00
        assert cost_per_success_with_cleanup(0.002, 0.60, human_fix_cost=2.00) == pytest.approx(0.802)

    def test_given_human_cleanup_of_failures_the_large_model_costs_11_cents_per_task(self):
        # 0.010 + 0.05 × 2.00
        assert cost_per_success_with_cleanup(0.010, 0.95, human_fix_cost=2.00) == pytest.approx(0.110)


class TestFiveTimesPlan:
    def test_given_the_sample_workload_the_ranked_levers_together_cut_cost_at_least_five_fold(self):
        steps = five_x_plan()
        assert steps[0]["cost"] / steps[-1]["cost"] >= 5

    def test_given_the_plan_the_free_levers_come_before_the_model_change(self):
        # Caching and trimming don't change the model, so they can't hurt quality; try them first.
        names = [s["lever"] for s in five_x_plan()]
        assert names.index("prompt caching") < names.index("route easy tasks to the small model")


class TestTheParallelFigureIsStable:
    def test_given_three_100ms_calls_the_figure_draws_them_end_to_end_then_all_at_once(self):
        import matplotlib

        matplotlib.use("Agg")
        from primer.agents.cost import figures

        # The figure draws the schedule, so it is identical on every build; the demo and the tests measure it.
        bars = [(p.get_x(), p.get_width()) for p in figures()["parallel"].axes[0].patches]
        assert bars == [(0, 100), (100, 100), (200, 100), (0, 100), (0, 100), (0, 100)]


class TestTheSemanticCacheSweep:
    def test_given_thresholds_from_0_4_up_the_cache_serves_more_wrong_answers_than_right_ones(self):
        from primer.agents.cost import semantic_cache_sweep

        sweep = semantic_cache_sweep([0.4, 0.95])
        # At 0.4: 57% of paraphrases hit, 60% of different-intent questions wrongly hit. At 0.95: 14% vs 40%.
        assert [(round(s["useful_hit_rate"], 2), round(s["wrong_hit_rate"], 2)) for s in sweep] == [(0.57, 0.6), (0.14, 0.4)]

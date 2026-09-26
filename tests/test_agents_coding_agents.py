"""Specification for primer.agents.coding_agents: agents that edit, run and test code, and agents that drive a screen."""

import sys

import pytest

from primer.agents.agent_loop import ToolError
from primer.agents.coding_agents import (
    EXAMPLE_AGENT,
    INJECTION_NOTICE,
    LEAP_BREAKS_2000,
    LEAP_FIXED,
    MEDIAN_CASES,
    MEDIAN_FIXED,
    MEDIAN_SPECIAL_CASED,
    MINI_BENCH,
    SHIFT_NOTICE,
    TASK,
    TOY_REPO,
    Case,
    Limits,
    Workspace,
    benchmark,
    careful_fixer,
    chance_within,
    context_tokens,
    cost_per_resolved,
    fix_until_green,
    form_filling_policy,
    grade,
    gullible_screen_policy,
    image_tokens,
    make_signup_screen,
    memorized_clicks_policy,
    overconfident_fixer,
    pass_at_k,
    resolved_rate,
    run_cases,
    run_computer_agent,
    run_sandboxed,
    screenshot_tokens,
)
from primer.agents.llm import ScriptedLLM, ToolCall


def _fix(policy, max_steps=10):
    workspace = Workspace(TOY_REPO, MEDIAN_CASES)
    return workspace, fix_until_green(ScriptedLLM(policy), workspace, TASK, max_steps=max_steps)


def _task(task_id):
    return next(t for t in MINI_BENCH if t.id == task_id)


class TestWhyCodeSuitsAgents:
    def test_given_a_40_percent_fix_rate_and_tests_to_check_each_try_three_tries_succeed_78_percent_of_the_time(self):
        # 1 - 0.6³ = 1 - 0.216.
        assert chance_within(0.4, 3) == pytest.approx(0.784)

    def test_given_a_single_try_the_chance_is_just_the_fix_rate(self):
        assert chance_within(0.4, 1) == pytest.approx(0.4)

    def test_given_a_failing_test_the_report_names_the_call_what_it_returned_and_what_was_expected(self):
        # sorted([4, 1, 3, 2]) is [1, 2, 3, 4]; the buggy median takes index 4 // 2 = 2, which holds 3.
        assert run_cases(TOY_REPO, MEDIAN_CASES).failures[0] == "median([4, 1, 3, 2]) returned 3, expected 2.5"


class TestRunningTheTests:
    def test_given_the_buggy_median_the_odd_length_cases_pass_and_the_even_length_ones_fail(self):
        report = run_cases(TOY_REPO, MEDIAN_CASES)
        assert (report.passed, report.total) == (2, 4)

    def test_given_code_that_raises_the_report_names_the_exception_and_its_message(self):
        report = run_cases({"m.py": "def first():\n    return [][0]\n"}, [Case("first()", 1)])
        assert report.failures == ["first() raised IndexError: list index out of range"]

    def test_given_a_patch_with_a_syntax_error_the_report_names_the_file_and_line(self):
        report = run_cases({"m.py": "def f(:\n    return 1\n"}, [Case("f()", 1)])
        assert report.failures[0].startswith("m.py line 1: SyntaxError")

    def test_given_two_cases_each_runs_in_a_fresh_namespace_so_one_cannot_leak_into_the_next(self):
        # If state leaked, the second call would see two appends and return 2.
        code = "calls = []\ndef count():\n    calls.append(1)\n    return len(calls)\n"
        assert run_cases({"c.py": code}, [Case("count()", 1), Case("count()", 1)]).green


class TestEditingFiles:
    def test_given_old_text_that_appears_once_the_edit_replaces_it(self):
        workspace = Workspace(TOY_REPO, MEDIAN_CASES)
        workspace.edit_file("stats.py", "return xs[len(xs) // 2]", "return xs[0]")
        assert "    return xs[0]\n" in workspace.files["stats.py"]

    def test_given_old_text_that_is_not_in_the_file_the_error_says_to_copy_the_lines_exactly(self):
        with pytest.raises(ToolError, match="copy the lines exactly"):
            Workspace(TOY_REPO, MEDIAN_CASES).edit_file("stats.py", "return xs[mid]", "return 0")

    def test_given_old_text_that_matches_twice_the_error_asks_for_more_surrounding_lines(self):
        # "len(xs)" is in both median and mean: replacing both would be a guess about which one was meant.
        with pytest.raises(ToolError, match="appears 2 times"):
            Workspace(TOY_REPO, MEDIAN_CASES).edit_file("stats.py", "len(xs)", "len(values)")

    def test_given_a_file_that_does_not_exist_the_error_lists_the_real_ones(self):
        with pytest.raises(ToolError, match="stats.py"):
            Workspace(TOY_REPO, MEDIAN_CASES).read_file("stat.py")


class TestTheEditRunTestLoop:
    def test_given_a_model_that_tests_searches_reads_and_patches_the_tests_go_green_at_step_7(self):
        _, result = _fix(careful_fixer)
        assert (result.outcome, result.steps) == ("green", 7)

    def test_given_the_careful_run_its_steps_are_test_search_read_edit_test_edit_test(self):
        _, result = _fix(careful_fixer)
        assert result.actions == ["run_tests", "search", "read_file", "edit_file", "run_tests", "edit_file", "run_tests"]

    def test_given_the_first_patch_is_off_by_one_the_next_test_run_names_the_index_error(self):
        # The wrong patch reads xs[mid + 1]; for a two-item list that is past the end.
        workspace, _ = _fix(careful_fixer)
        assert "median([5, 1]) raised IndexError: list index out of range" in workspace.reports[1].failures

    def test_given_the_careful_run_the_passing_count_goes_two_two_four(self):
        # The wrong patch passes no more cases than the bug did, but its message changes, and that guides the fix.
        workspace, _ = _fix(careful_fixer)
        assert [r.passed for r in workspace.reports] == [2, 2, 4]

    def test_given_a_model_that_says_fixed_without_running_the_tests_the_harness_does_not_take_its_word(self):
        _, result = _fix(overconfident_fixer, max_steps=4)
        assert result.outcome == "out_of_budget"

    def test_given_a_model_that_claims_success_the_harness_replies_with_the_failing_tests(self):
        _, result = _fix(overconfident_fixer, max_steps=4)
        assert "median([4, 1, 3, 2]) returned 3.5, expected 2.5" in result.transcript[4]["content"]

    def test_given_a_model_that_never_fixes_anything_the_run_stops_at_the_step_budget(self):
        _, result = _fix([ToolCall("", "run_tests", {})], max_steps=3)
        assert (result.outcome, result.steps) == ("out_of_budget", 3)


class TestFindingTheRightFiles:
    def test_given_a_search_for_a_definition_it_returns_the_file_and_line_number(self):
        assert Workspace(TOY_REPO, MEDIAN_CASES).search("def median") == "stats.py:1: def median(xs):"

    def test_given_more_matches_than_the_cap_the_rest_are_counted_not_dumped(self):
        hits = Workspace(TOY_REPO, MEDIAN_CASES, max_hits=2).search("def ").splitlines()
        assert len(hits) == 3 and hits[-1].endswith("more matches; use a more specific pattern.")

    def test_given_the_careful_run_the_only_file_it_reads_is_stats_py(self):
        workspace, _ = _fix(careful_fixer)
        assert workspace.files_read == ["stats.py"]

    def test_given_the_careful_run_it_puts_under_a_fifth_of_the_repository_into_context(self):
        workspace, _ = _fix(careful_fixer)
        assert workspace.context_chars < sum(len(s) for s in TOY_REPO.values()) / 5

    def test_given_500_files_of_400_tokens_a_dump_is_200000_tokens_and_search_then_read_is_850(self):
        # Dump: 500 · 400. Targeted: 50 tokens of search hits plus 2 files · 400.
        assert context_tokens(500, 400, files_read=2, search_tokens=50) == {"dump": 200_000, "targeted": 850}


class TestTheSandbox:
    def test_given_a_well_behaved_function_it_returns_its_value(self):
        assert run_sandboxed("def double(x):\n    return x * 2\n", "double(21)").value == 42

    def test_given_an_infinite_loop_it_is_stopped_on_the_first_line_past_the_budget(self):
        result = run_sandboxed("while True:\n    pass\n", limits=Limits(max_lines=1000))
        assert (result.ok, result.stopped_by, result.lines_run) == (False, "lines", 1001)

    def test_given_a_loop_that_catches_every_exception_it_is_still_stopped(self):
        # The limit is a BaseException, so `except Exception` in model-written code cannot swallow it.
        code = "while True:\n    try:\n        pass\n    except Exception:\n        pass\n"
        assert run_sandboxed(code, limits=Limits(max_lines=1000)).stopped_by == "lines"

    def test_given_a_wall_clock_limit_a_runaway_loop_is_stopped_by_time(self):
        result = run_sandboxed("while True:\n    pass\n", limits=Limits(max_lines=10**12, max_seconds=0.05))
        assert result.stopped_by == "time" and result.seconds < 1

    def test_given_code_that_keeps_allocating_it_is_stopped_at_the_memory_limit(self):
        # Each [0] * 1000 is a new 8 KB list; a constant string would be one object appended over and over.
        code = "hoard = []\nwhile True:\n    hoard.append([0] * 1000)\n"
        assert run_sandboxed(code, limits=Limits(max_bytes=1_000_000)).stopped_by == "memory"

    def test_given_code_that_imports_a_module_it_fails_before_it_can_reach_the_network(self):
        assert run_sandboxed("import socket\n").error == "ImportError: __import__ not found"

    def test_given_code_that_opens_a_file_it_fails_because_open_is_not_provided(self):
        assert run_sandboxed("open('/etc/passwd')\n").error == "NameError: name 'open' is not defined"

    def test_given_the_restricted_namespace_introspection_still_reaches_every_class_the_interpreter_loaded(self):
        # Why an in-process sandbox is a lesson, not a boundary: real systems isolate with a process, container or VM.
        result = run_sandboxed("", "len(().__class__.__base__.__subclasses__())")
        assert result.ok and result.value > 100

    def test_given_the_sandbox_finishes_the_previous_tracer_is_restored(self):
        # Debuggers and coverage tools install tracers too; the sandbox must hand theirs back.
        before = sys.gettrace()
        run_sandboxed("while True:\n    pass\n", limits=Limits(max_lines=100))
        assert sys.gettrace() is before


class TestEvaluatingCodingAgents:
    def test_given_a_correct_patch_every_hidden_test_passes_so_the_task_is_resolved(self):
        assert grade(_task("median-even"), {"stats.py": MEDIAN_FIXED}).resolved

    def test_given_a_patch_that_special_cases_the_visible_inputs_the_visible_tests_pass(self):
        assert run_cases({**TOY_REPO, "stats.py": MEDIAN_SPECIAL_CASED}, MEDIAN_CASES).green

    def test_given_a_patch_that_special_cases_the_visible_inputs_the_hidden_tests_leave_it_unresolved(self):
        assert not grade(_task("median-even"), {"stats.py": MEDIAN_SPECIAL_CASED}).resolved

    def test_given_a_patch_that_fixes_1900_but_breaks_2000_the_pass_to_pass_tests_leave_it_unresolved(self):
        outcome = grade(_task("leap-century"), {"dates.py": LEAP_BREAKS_2000})
        assert (outcome.fail_to_pass.green, outcome.pass_to_pass.green, outcome.resolved) == (True, False, False)

    def test_given_the_correct_leap_year_rule_the_task_is_resolved(self):
        assert grade(_task("leap-century"), {"dates.py": LEAP_FIXED}).resolved

    def test_given_the_example_agent_two_of_the_three_tasks_are_resolved(self):
        assert resolved_rate(benchmark(EXAMPLE_AGENT)) == pytest.approx(2 / 3)

    def test_given_10_samples_with_3_correct_pass_at_1_is_0_3(self):
        # 1 - C(7, 1) / C(10, 1) = 1 - 7/10.
        assert pass_at_k(10, 3, 1) == pytest.approx(0.3)

    def test_given_10_samples_with_3_correct_pass_at_5_is_1_minus_21_over_252(self):
        # C(7, 5) = 21 ways to draw five all-wrong samples, out of C(10, 5) = 252 ways to draw five.
        assert pass_at_k(10, 3, 5) == pytest.approx(1 - 21 / 252)

    def test_given_no_correct_samples_pass_at_k_is_zero_at_any_k(self):
        assert pass_at_k(10, 0, 5) == 0.0

    def test_given_fewer_wrong_samples_than_draws_every_draw_holds_a_correct_one(self):
        assert pass_at_k(10, 8, 5) == 1.0

    def test_given_ten_attempts_at_60_cents_with_four_resolved_each_resolved_task_costs_1_50(self):
        # $6.00 spent, 4 tasks resolved.
        assert cost_per_resolved([0.60] * 10, [True] * 4 + [False] * 6) == pytest.approx(1.50)

    def test_given_nothing_resolved_the_cost_per_resolved_task_is_infinite(self):
        assert cost_per_resolved([0.60] * 3, [False] * 3) == float("inf")


class TestDrivingAScreen:
    def test_given_the_signup_form_its_screenshot_shows_each_control_as_a_row_of_text(self):
        rows = make_signup_screen().screenshot().splitlines()
        assert rows[2].startswith("Name:") and rows[6].startswith("[ Submit ]")

    def test_given_a_click_on_a_field_it_takes_focus_and_typed_text_lands_in_it(self):
        screen = make_signup_screen()
        screen.click(10, 2)
        screen.type_text("Ada")
        assert "Ada" in screen.screenshot().splitlines()[2]

    def test_given_a_click_on_empty_space_nothing_happens_and_no_error_is_raised(self):
        # A GUI fails silently where an API would return an error.
        screen = make_signup_screen()
        before = screen.screenshot()
        screen.click(40, 1)
        assert screen.screenshot() == before

    def test_given_an_agent_that_looks_before_each_action_it_completes_the_form(self):
        run = run_computer_agent(ScriptedLLM(form_filling_policy), make_signup_screen(), "Sign Ada up.")
        assert run.outcome == "done"

    def test_given_the_looking_agent_it_takes_8_model_calls_and_7_screenshots(self):
        # One look, six actions (click, type, click, type, tick, submit), one final answer.
        run = run_computer_agent(ScriptedLLM(form_filling_policy), make_signup_screen(), "Sign Ada up.")
        assert (run.steps, run.screenshots) == (8, 7)

    def test_given_a_layout_shift_the_agent_that_looks_again_still_completes_the_form(self):
        run = run_computer_agent(ScriptedLLM(form_filling_policy), make_signup_screen(SHIFT_NOTICE), "Sign Ada up.")
        assert run.outcome == "done"

    def test_given_the_original_layout_replayed_coordinates_complete_the_form(self):
        run = run_computer_agent(ScriptedLLM(memorized_clicks_policy), make_signup_screen(), "Sign Ada up.")
        assert run.outcome == "done"

    def test_given_a_layout_shift_replayed_coordinates_miss_and_nothing_is_submitted(self):
        run = run_computer_agent(ScriptedLLM(memorized_clicks_policy), make_signup_screen(SHIFT_NOTICE), "Sign Ada up.")
        assert run.outcome == "form"

    def test_given_an_instruction_written_on_screen_an_unguarded_gullible_agent_deletes_the_account(self):
        screen = make_signup_screen(INJECTION_NOTICE)
        run = run_computer_agent(ScriptedLLM(gullible_screen_policy), screen, "Sign Ada up.", guard=False)
        assert run.outcome == "deleted"

    def test_given_a_guard_on_destructive_controls_the_injected_click_is_blocked(self):
        screen = make_signup_screen(INJECTION_NOTICE)
        run = run_computer_agent(ScriptedLLM(gullible_screen_policy), screen, "Sign Ada up.")
        assert run.blocked == ["Delete account"]

    def test_given_the_blocked_click_comes_back_as_an_error_the_agent_returns_to_its_task_and_finishes(self):
        screen = make_signup_screen(INJECTION_NOTICE)
        run = run_computer_agent(ScriptedLLM(gullible_screen_policy), screen, "Sign Ada up.")
        assert run.outcome == "done"

    def test_given_a_1280_by_800_screenshot_cut_into_32_pixel_patches_it_costs_1000_tokens(self):
        # (1280 / 32) · (800 / 32) = 40 · 25.
        assert screenshot_tokens(1280, 800, 32) == 1000

    def test_given_six_actions_at_that_size_the_run_sends_7000_image_tokens(self):
        # Six actions each return a screenshot, plus the first look: seven screenshots.
        assert image_tokens(6, 1280, 800, 32) == 7000


class TestFigures:
    def test_given_the_lesson_it_draws_its_seven_figures(self):
        pytest.importorskip("matplotlib")
        from primer.agents.coding_agents import figures

        assert sorted(figures()) == [
            "context_tokens", "fix_loop_trace", "gui_vs_api", "layout_shift", "pass_at_k", "retries_with_checker",
            "sandbox_limits",
        ]

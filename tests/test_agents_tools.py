"""Specification for primer.agents.tools: tools a model can pick correctly and call safely."""

import pytest

from primer.agents.tools import ToolRegistry, validate

ADD_SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}


def _registry():
    reg = ToolRegistry()
    reg.register("add", "Add two integers. Use for arithmetic on whole numbers.", ADD_SCHEMA, lambda a, b: a + b)
    return reg


class TestToolDefinitions:
    def test_given_a_registered_function_its_definition_has_the_name_description_and_input_schema(self):
        assert _registry().definitions() == [
            {"name": "add", "description": "Add two integers. Use for arithmetic on whole numbers.", "input_schema": ADD_SCHEMA}
        ]

    def test_given_strict_mode_each_definition_is_marked_strict_and_forbids_extra_fields(self):
        # With strict tool use the API guarantees the arguments match the schema exactly.
        definition = _registry().definitions(strict=True)[0]
        assert definition["strict"] is True
        assert definition["input_schema"]["additionalProperties"] is False


PAYMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "amount": {"type": "number", "minimum": 0.01, "description": "amount in major units, e.g. 12.50"},
        "currency": {"type": "string", "enum": ["EUR", "GBP", "USD"]},
        "pay_on": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$", "description": "date as YYYY-MM-DD"},
        "quantity": {"type": "integer", "maximum": 100},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["amount", "currency"],
    "additionalProperties": False,
}


class TestArgumentValidation:
    def test_given_a_missing_required_argument_the_error_names_it(self):
        assert validate({"amount": 5}, PAYMENT_SCHEMA) == ["$: missing required field 'currency'"]

    def test_given_text_where_a_whole_number_is_expected_the_error_names_the_field_and_both_types(self):
        errors = validate({"amount": 5, "currency": "USD", "quantity": "3"}, PAYMENT_SCHEMA)
        assert errors == ["$.quantity: expected integer, got string ('3')"]

    def test_given_true_where_a_whole_number_is_expected_it_is_rejected_as_a_boolean(self):
        # In Python True == 1, so a naive isinstance(x, int) check would accept it.
        assert validate({"amount": 5, "currency": "USD", "quantity": True}, PAYMENT_SCHEMA) == [
            "$.quantity: expected integer, got boolean (True)"
        ]

    def test_given_the_lessons_misspelt_payment_every_problem_comes_back_at_once_with_its_fix(self):
        # The worked example: one typo, one lowercase currency, one plain-English date, so four fixes.
        from primer.agents.tools import PAYMENT_SCHEMA as LESSON_PAYMENT_SCHEMA

        assert validate({"ammount": 5, "currency": "usd", "pay_on": "next friday"}, LESSON_PAYMENT_SCHEMA) == [
            "$: missing required field 'amount'",
            "$: unexpected field 'ammount' (allowed: amount, currency, pay_on)",
            "$.currency: must be one of ['EUR', 'GBP', 'USD'], got 'usd'",
            "$.pay_on: 'next friday' does not match the required format. Expected: date as YYYY-MM-DD",
        ]

    def test_given_a_value_outside_the_allowed_set_the_error_lists_the_allowed_values(self):
        assert validate({"amount": 5, "currency": "usd"}, PAYMENT_SCHEMA) == [
            "$.currency: must be one of ['EUR', 'GBP', 'USD'], got 'usd'"
        ]

    def test_given_a_date_in_the_wrong_format_the_error_shows_the_expected_format(self):
        # "invalid input" leaves the model guessing; the schema's own description tells it what to send.
        assert validate({"amount": 5, "currency": "USD", "pay_on": "next friday"}, PAYMENT_SCHEMA) == [
            "$.pay_on: 'next friday' does not match the required format. Expected: date as YYYY-MM-DD"
        ]

    def test_given_an_amount_below_the_minimum_the_error_states_the_bound(self):
        assert validate({"amount": -5, "currency": "USD"}, PAYMENT_SCHEMA) == ["$.amount: must be >= 0.01, got -5"]

    def test_given_a_quantity_above_the_maximum_the_error_states_the_bound(self):
        assert validate({"amount": 5, "currency": "USD", "quantity": 250}, PAYMENT_SCHEMA) == [
            "$.quantity: must be <= 100, got 250"
        ]

    def test_given_a_misspelled_field_the_error_names_it_and_lists_the_allowed_fields(self):
        # A typo like "ammount" would otherwise be silently ignored and the real field reported missing.
        assert validate({"ammount": 5, "currency": "USD"}, PAYMENT_SCHEMA) == [
            "$: missing required field 'amount'",
            "$: unexpected field 'ammount' (allowed: amount, currency, pay_on, quantity, tags)",
        ]

    def test_given_a_list_with_one_wrong_item_the_error_points_at_that_item(self):
        assert validate({"amount": 5, "currency": "USD", "tags": ["travel", 7]}, PAYMENT_SCHEMA) == [
            "$.tags[1]: expected string, got integer (7)"
        ]

    def test_given_valid_arguments_there_are_no_errors(self):
        assert validate({"amount": 12.5, "currency": "EUR", "pay_on": "2026-10-02", "tags": ["q3"]}, PAYMENT_SCHEMA) == []


class TestCallingATool:
    def test_given_valid_arguments_the_tool_runs_and_its_result_comes_back_as_text(self):
        outcome = _registry().call("add", {"a": 2, "b": 3})
        assert (outcome.status, outcome.content) == ("ok", "5")

    def test_given_invalid_arguments_the_tool_does_not_run_and_every_problem_is_reported_at_once(self):
        # Reporting all problems together lets the model fix its call in one retry instead of several.
        outcome = _registry().call("add", {"a": "two"})
        assert outcome.status == "invalid"
        assert outcome.content == (
            "Invalid arguments for add:\n- $: missing required field 'b'\n- $.a: expected integer, got string ('two')"
        )

    def test_given_a_tool_name_that_does_not_exist_the_reply_lists_the_tools_that_do(self):
        outcome = _registry().call("multiply", {"a": 2, "b": 3})
        assert (outcome.status, outcome.content) == ("unknown_tool", "Unknown tool 'multiply'. Available: add.")


REFUND_SCHEMA = {
    "type": "object",
    "properties": {"customer_id": {"type": "string", "pattern": r"^C-\d+$"}, "amount": {"type": "number", "minimum": 0.01}},
    "required": ["customer_id", "amount"],
}


def _payments(**overrides):
    """A registry with one write tool that refunds a customer, recording every refund in a ledger."""
    ledger = []
    customers = {"C-100", "C-200"}

    def refund(customer_id, amount):
        ledger.append((customer_id, amount))
        return f"refunded {amount:.2f} to {customer_id}"

    reg = ToolRegistry()
    options = dict(
        check=lambda a: [] if a["customer_id"] in customers else [f"No customer {a['customer_id']!r}. Look the id up with find_customer first."],
        scopes=frozenset({"payments:write"}),
        side_effect="write",
    )
    options.update(overrides)
    reg.register("refund_payment", "Refund a customer payment.", REFUND_SCHEMA, refund, **options)
    return reg, ledger


ALL_ACCESS = frozenset({"payments:write"})


class TestSemanticValidation:
    def test_given_a_well_formed_id_for_a_customer_that_does_not_exist_the_refund_never_runs(self):
        # "C-999" passes the schema's pattern; only a lookup can tell it isn't a real customer.
        reg, ledger = _payments()
        outcome = reg.call("refund_payment", {"customer_id": "C-999", "amount": 20}, credentials=ALL_ACCESS)
        assert (outcome.status, ledger) == ("rejected", [])
        assert outcome.content == "No customer 'C-999'. Look the id up with find_customer first."


class TestLeastPrivilege:
    def test_given_credentials_that_only_read_tickets_a_refund_is_denied_and_never_runs(self):
        # Scope each agent's credentials to its job, so a tricked agent can't do what it was never allowed to.
        reg, ledger = _payments()
        outcome = reg.call("refund_payment", {"customer_id": "C-100", "amount": 20}, credentials=frozenset({"tickets:read"}))
        assert (outcome.status, ledger) == ("denied", [])
        assert outcome.content == "Permission denied: refund_payment needs scope 'payments:write'; this agent has ['tickets:read']."

    def test_given_credentials_with_the_right_scope_the_refund_runs(self):
        reg, ledger = _payments()
        reg.call("refund_payment", {"customer_id": "C-100", "amount": 20}, credentials=ALL_ACCESS)
        assert ledger == [("C-100", 20)]


class TestIdempotency:
    def test_given_a_retried_refund_with_the_same_idempotency_key_the_money_moves_once(self):
        # Networks time out after the write succeeds; the retry must not refund twice.
        reg, ledger = _payments()
        args = {"customer_id": "C-100", "amount": 20}
        reg.call("refund_payment", args, credentials=ALL_ACCESS, idempotency_key="req-1")
        retry = reg.call("refund_payment", args, credentials=ALL_ACCESS, idempotency_key="req-1")
        assert ledger == [("C-100", 20)]
        assert (retry.status, retry.content) == ("duplicate", "refunded 20.00 to C-100")

    def test_given_two_different_idempotency_keys_both_refunds_happen(self):
        reg, ledger = _payments()
        args = {"customer_id": "C-100", "amount": 20}
        reg.call("refund_payment", args, credentials=ALL_ACCESS, idempotency_key="req-1")
        reg.call("refund_payment", args, credentials=ALL_ACCESS, idempotency_key="req-2")
        assert len(ledger) == 2


class TestDryRun:
    def test_given_a_dry_run_the_reply_describes_the_refund_and_no_money_moves(self):
        reg, ledger = _payments()
        outcome = reg.call("refund_payment", {"customer_id": "C-100", "amount": 20}, credentials=ALL_ACCESS, dry_run=True)
        assert ledger == []
        assert (outcome.status, outcome.content) == (
            "dry_run",
            'Dry run: would call refund_payment with {"amount": 20, "customer_id": "C-100"}. Nothing was changed.',
        )


class TestHumanApproval:
    def test_given_an_irreversible_action_and_no_approver_it_waits_for_approval_and_does_not_run(self):
        reg, ledger = _payments(side_effect="irreversible")
        outcome = reg.call("refund_payment", {"customer_id": "C-100", "amount": 20}, credentials=ALL_ACCESS)
        assert (outcome.status, ledger) == ("awaiting_approval", [])

    def test_given_an_approver_who_declines_the_refund_does_not_run_and_the_model_is_told_why(self):
        reg, ledger = _payments(side_effect="irreversible")
        outcome = reg.call(
            "refund_payment", {"customer_id": "C-100", "amount": 20}, credentials=ALL_ACCESS, approver=lambda name, args: False
        )
        assert (outcome.status, ledger) == ("declined", [])
        assert outcome.content == "A human declined refund_payment. Do not retry; tell the user it was not approved."

    def test_given_an_approver_who_agrees_the_refund_runs(self):
        reg, ledger = _payments(side_effect="irreversible")
        reg.call("refund_payment", {"customer_id": "C-100", "amount": 20}, credentials=ALL_ACCESS, approver=lambda n, a: True)
        assert ledger == [("C-100", 20)]

    def test_given_a_rule_that_refunds_over_500_need_approval_a_refund_of_40_runs_without_asking(self):
        reg, ledger = _payments(needs_approval=lambda a: a["amount"] > 500)
        assert reg.call("refund_payment", {"customer_id": "C-100", "amount": 40}, credentials=ALL_ACCESS).status == "ok"

    def test_given_a_rule_that_refunds_over_500_need_approval_a_refund_of_900_waits(self):
        reg, ledger = _payments(needs_approval=lambda a: a["amount"] > 500)
        outcome = reg.call("refund_payment", {"customer_id": "C-100", "amount": 900}, credentials=ALL_ACCESS)
        assert (outcome.status, ledger) == ("awaiting_approval", [])


class TestToolDescriptions:
    # A stand-in model that picks the tool whose name and description share the most words with the
    # request. Real models are far smarter, but they choose from the same text, and fail the same way.

    def test_given_vague_overlapping_descriptions_a_model_picks_the_right_tool_for_2_of_6_requests(self):
        from primer.agents.tools import LABELED_REQUESTS, VAGUE_TOOLS, selection_accuracy

        assert selection_accuracy(VAGUE_TOOLS, LABELED_REQUESTS) == (2, 6)

    def test_given_precise_descriptions_that_say_when_to_use_each_tool_it_picks_right_for_6_of_6(self):
        from primer.agents.tools import LABELED_REQUESTS, PRECISE_TOOLS, selection_accuracy

        assert selection_accuracy(PRECISE_TOOLS, LABELED_REQUESTS) == (6, 6)


class TestFewerHigherLevelTools:
    def test_given_calls_that_each_succeed_97_percent_of_the_time_five_in_a_row_succeed_about_86_percent(self):
        # 0.97^5 = 0.8587: five low-level calls to build an invoice vs. one create_invoice call.
        from primer.agents.tools import chain_success

        assert chain_success(0.97, 5) == pytest.approx(0.8587, abs=1e-4)

    def test_given_one_high_level_call_at_97_percent_the_whole_job_succeeds_97_percent_of_the_time(self):
        from primer.agents.tools import chain_success

        assert chain_success(0.97, 1) == pytest.approx(0.97)


class TestDynamicToolLoading:
    def test_given_a_catalog_of_30_tools_a_locked_out_request_loads_the_password_reset_tool(self):
        # Past a few dozen tools, picking gets worse; send the model only the few that fit the request.
        from primer.agents.tools import TOOL_CATALOG, select_tools

        assert len(TOOL_CATALOG) == 30
        chosen = [d["name"] for d in select_tools("I forgot my password and I'm locked out", TOOL_CATALOG, k=5)]
        assert "reset_password" in chosen

    def test_given_a_limit_of_5_only_5_tool_definitions_are_sent_to_the_model(self):
        from primer.agents.tools import TOOL_CATALOG, select_tools

        assert len(select_tools("submit my car mileage expense", TOOL_CATALOG, k=5)) == 5

    def test_given_an_expense_request_the_best_match_is_the_expense_tool(self):
        from primer.agents.tools import TOOL_CATALOG, select_tools

        assert select_tools("submit my car mileage expense", TOOL_CATALOG, k=5)[0]["name"] == "submit_expense"


class TestFigures:
    def test_given_the_lesson_it_draws_compounding_reliability_selection_accuracy_and_tool_similarity(self):
        pytest.importorskip("matplotlib")
        from primer.agents.tools import figures

        assert sorted(figures()) == ["compounding", "selection_accuracy", "tool_similarity"]

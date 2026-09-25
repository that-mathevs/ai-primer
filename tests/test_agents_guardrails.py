"""Specification for primer.agents.guardrails: input, output and action checks, and injection-proof design."""

from primer.agents.guardrails import (
    INBOX,
    PARAPHRASED_ATTACK,
    ActionPolicy,
    attack_outcomes,
    detect_injection,
    find_pii,
    groundedness,
    guarded_answer,
    luhn_valid,
    policy_violations,
    redact_pii,
    run_naive_agent,
    run_separated_agent,
    validate_schema,
)


class TestInjectionHeuristics:
    def test_given_an_email_saying_ignore_previous_instructions_the_detector_flags_an_override(self):
        rules = [f.rule for f in detect_injection(INBOX[1]["body"])]
        assert "override" in rules

    def test_given_an_ordinary_business_email_the_detector_finds_nothing(self):
        assert detect_injection(INBOX[0]["body"]) == []

    def test_given_the_same_attack_paraphrased_the_detector_misses_it(self):
        # This is the honest limit of pattern matching: rewording defeats it.
        assert detect_injection(PARAPHRASED_ATTACK) == []


class TestLuhnChecksum:
    def test_given_the_standard_visa_test_number_the_checksum_passes(self):
        assert luhn_valid("4111 1111 1111 1111")

    def test_given_one_digit_changed_the_checksum_fails(self):
        assert not luhn_valid("4111 1111 1111 1112")

    def test_given_fewer_than_thirteen_digits_it_is_not_a_card_number(self):
        assert not luhn_valid("4111 1111 11")


class TestPIIRedaction:
    def test_given_an_email_address_it_becomes_a_typed_placeholder(self):
        assert redact_pii("write to jo@corp.example today") == "write to [EMAIL] today"

    def test_given_a_phone_number_it_becomes_a_typed_placeholder(self):
        assert redact_pii("call (555) 123-4567 now") == "call [PHONE] now"

    def test_given_a_valid_card_number_it_becomes_a_typed_placeholder(self):
        assert redact_pii("card 4111-1111-1111-1111 on file") == "card [CARD] on file"

    def test_given_a_sixteen_digit_order_number_that_fails_luhn_it_is_left_alone(self):
        # Validation is what stops a PII filter from redacting every long ID.
        assert find_pii("order 1234 5678 9012 3456") == []


class TestSchemaValidation:
    SCHEMA = {
        "type": "object",
        "required": ["order_id", "amount"],
        "additionalProperties": False,
        "properties": {"order_id": {"type": "string"}, "amount": {"type": "number", "minimum": 0}},
    }

    def test_given_a_well_formed_refund_there_are_no_errors(self):
        assert validate_schema({"order_id": "A100", "amount": 40}, self.SCHEMA) == []

    def test_given_an_amount_sent_as_a_string_the_error_names_the_field_and_expected_type(self):
        assert validate_schema({"order_id": "A100", "amount": "40"}, self.SCHEMA) == ["$.amount: expected number, got str"]

    def test_given_a_missing_field_the_error_names_it(self):
        assert validate_schema({"amount": 1}, self.SCHEMA) == ["$: missing required field 'order_id'"]

    def test_given_an_unexpected_field_it_is_rejected(self):
        assert validate_schema({"order_id": "A1", "amount": 1, "note": "x"}, self.SCHEMA) == ["$: unexpected field 'note'"]

    def test_given_a_boolean_where_a_number_belongs_it_is_rejected(self):
        # Python treats True as the integer 1; a schema must not.
        assert validate_schema({"order_id": "A1", "amount": True}, self.SCHEMA) == ["$.amount: expected number, got bool"]


class TestOutputPolicy:
    def test_given_an_answer_that_promises_a_guarantee_the_policy_flags_it(self):
        assert policy_violations("We guarantee a refund.") == ["no_guarantees"]

    def test_given_an_answer_that_reveals_a_password_the_policy_flags_it(self):
        assert policy_violations("The admin password is hunter2") == ["no_credentials"]


class TestGroundedness:
    SOURCES = ["Full-time employees accrue 20 days of PTO per year. Unused PTO up to 5 days rolls over."]

    def test_given_an_answer_restating_the_source_every_claim_is_supported(self):
        g = groundedness("Employees accrue 20 days of PTO per year [hr-001].", self.SOURCES)
        assert g["score"] == 1.0

    def test_given_one_invented_claim_out_of_two_half_the_answer_is_grounded(self):
        g = groundedness("Employees accrue 20 days of PTO per year. Managers get unlimited sabbaticals.", self.SOURCES)
        assert g["score"] == 0.5
        assert g["unsupported"] == ["Managers get unlimited sabbaticals."]

    def test_given_a_grounded_answer_containing_an_email_address_the_layered_check_blocks_it(self):
        verdict = guarded_answer("Employees accrue 20 days of PTO per year, ask hr@corp.example.", self.SOURCES)
        assert not verdict["ok"]
        assert verdict["reasons"] == ["pii: answer contains personal data"]


class TestActionPolicy:
    def test_given_a_tool_the_agent_was_not_granted_the_call_is_denied(self):
        assert ActionPolicy(allowed_tools={"refund"}).check("send_email", {}).verdict == "deny"

    def test_given_a_small_refund_it_is_allowed(self):
        assert ActionPolicy(allowed_tools={"refund"}).check("refund", {"amount": 40}).verdict == "allow"

    def test_given_a_refund_over_the_approval_threshold_a_human_must_approve(self):
        assert ActionPolicy(allowed_tools={"refund"}, approval_over=100).check("refund", {"amount": 120}).verdict == "needs_approval"

    def test_given_refunds_that_together_exceed_the_spend_cap_the_last_one_is_denied(self):
        policy = ActionPolicy(allowed_tools={"refund"}, spend_cap=100, approval_over=1000)
        policy.record({"amount": 80})
        assert policy.check("refund", {"amount": 30}).verdict == "deny"

    def test_given_an_email_to_an_outside_domain_a_human_must_approve(self):
        policy = ActionPolicy(allowed_tools={"send_email"}, internal_domains={"example.com"})
        assert policy.check("send_email", {"to": "x@evil.example"}).verdict == "needs_approval"

    def test_given_an_irreversible_tool_a_human_must_approve_even_with_no_amount(self):
        assert ActionPolicy(allowed_tools={"delete_records"}).check("delete_records", {}).verdict == "needs_approval"


class TestPrivilegeSeparation:
    def test_given_a_single_agent_that_reads_email_and_can_send_the_injection_exfiltrates_invoices(self):
        sent = run_naive_agent(INBOX)
        assert [s["to"] for s in sent] == ["attacker@evil.example"]

    def test_given_reader_and_actor_are_separated_nothing_is_sent_to_the_attacker(self):
        assert run_separated_agent(INBOX)["sent"] == []

    def test_given_reader_and_actor_are_separated_the_injected_forward_request_is_blocked_and_visible(self):
        assert {"action": "forward", "target": "attacker@evil.example"} in run_separated_agent(INBOX)["blocked"]

    def test_given_the_user_only_asked_for_a_summary_even_a_benign_reply_waits_for_the_user(self):
        # Actions proposed by email content never run on the email's say-so, however harmless.
        assert {"action": "reply", "target": "dana@example.com"} in run_separated_agent(INBOX)["blocked"]

    def test_given_every_attack_variant_only_the_naive_design_leaks(self):
        outcomes = attack_outcomes()
        assert all(o["naive_leaked"] for o in outcomes)
        assert not any(o["separated_leaked"] for o in outcomes)

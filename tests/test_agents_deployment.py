"""Specification for primer.agents.deployment: letting an agent act in the real world, one earned step at a time."""

import pytest

from primer.agents.deployment import (
    AuditLog,
    AutonomyController,
    CanaryController,
    KillSwitch,
    PromptRegistry,
    TokenBucket,
    in_canary,
    shadow_agreement,
)


class TestShadowMode:
    AGENT = ["refund", "refund", "reply", "reply", "reply", "refund", "reply", "refund", "reply", "refund"]
    HUMAN = ["refund", "escalate", "reply", "reply", "reply", "refund", "reply", "escalate", "reply", "refund"]

    def test_given_eight_matching_decisions_out_of_ten_overall_agreement_is_eighty_percent(self):
        assert shadow_agreement(self.AGENT, self.HUMAN)["overall"] == pytest.approx(0.8)

    def test_given_the_same_run_refund_decisions_agree_three_times_out_of_five(self):
        # Broken down by what the agent proposed, the weak spot is visible: refunds.
        assert shadow_agreement(self.AGENT, self.HUMAN)["by_agent_action"]["refund"] == pytest.approx(3 / 5)

    def test_given_the_same_run_reply_decisions_agree_every_time(self):
        assert shadow_agreement(self.AGENT, self.HUMAN)["by_agent_action"]["reply"] == 1.0


class TestGraduatedAutonomy:
    def test_given_a_new_agent_it_starts_in_shadow_mode(self):
        assert AutonomyController().level == "shadow"

    def test_given_48_of_50_shadow_decisions_matching_humans_it_is_promoted_to_human_approval(self):
        c = AutonomyController(min_decisions=50, min_agreement=0.95)
        for i in range(50):
            c.record_shadow(agreed=i >= 2)
        assert c.level == "approve_each"

    def test_given_46_of_50_shadow_decisions_matching_it_stays_in_shadow_mode(self):
        c = AutonomyController(min_decisions=50, min_agreement=0.95)
        for i in range(50):
            c.record_shadow(agreed=i >= 4)
        assert c.level == "shadow"

    def test_given_49_of_50_actions_approved_by_reviewers_it_may_act_alone_on_low_risk_actions(self):
        c = AutonomyController(level="approve_each", min_decisions=50, min_approval_rate=0.98)
        for i in range(50):
            c.record_approval(approved=i != 0)
        assert c.level == "auto_low_risk"

    def test_given_autonomy_high_risk_actions_still_need_a_person(self):
        c = AutonomyController(level="auto_low_risk")
        assert not c.may_act_alone(risk="high")
        assert c.may_act_alone(risk="low")

    def test_given_quality_drift_while_autonomous_it_drops_back_to_human_approval(self):
        c = AutonomyController(level="auto_low_risk")
        c.record_quality(success_rate=0.91, baseline=0.97, max_drop=0.03)
        assert c.level == "approve_each"

    def test_given_an_incident_while_autonomous_it_drops_back_to_human_approval(self):
        c = AutonomyController(level="auto_low_risk")
        c.record_incident("refunded the wrong customer")
        assert c.level == "approve_each"


class TestPromptsAsCode:
    def test_given_the_same_prompt_text_twice_it_gets_the_same_version(self):
        # Content-addressed: the version is a fingerprint of the text itself.
        reg = PromptRegistry()
        assert reg.register("support", "Be concise.") == reg.register("support", "Be concise.")

    def test_given_an_edited_prompt_it_gets_a_new_version(self):
        reg = PromptRegistry()
        assert reg.register("support", "Be concise.") != reg.register("support", "Be concise and friendly.")

    def test_given_a_rollback_the_previous_text_is_active_again(self):
        reg = PromptRegistry()
        reg.register("support", "Be concise.")
        reg.register("support", "Always be maximally helpful.")
        reg.rollback("support")
        assert reg.active_text("support") == "Be concise."

    def test_given_a_version_its_name_carries_the_first_eight_hex_digits_of_the_text_hash(self):
        # sha256("Be concise.") begins ab018bdb; checked independently with `printf "Be concise." | shasum -a 256`.
        assert PromptRegistry().register("support", "Be concise.") == "support@ab018bdb"


class TestCanaryAssignment:
    def test_given_the_same_user_the_assignment_never_changes(self):
        assert in_canary("user-42", percent=10) == in_canary("user-42", percent=10)

    def test_given_ten_percent_about_one_user_in_ten_is_in_the_canary(self):
        share = sum(in_canary(f"user-{i}", percent=10) for i in range(10_000)) / 10_000
        assert share == pytest.approx(0.10, abs=0.01)

    def test_given_a_user_in_the_5_percent_canary_they_stay_in_it_at_25_percent(self):
        # Growing the canary only adds users; nobody flips back and forth between versions.
        members = [f"user-{i}" for i in range(2_000) if in_canary(f"user-{i}", percent=5)]
        assert all(in_canary(u, percent=25) for u in members)


class TestCanaryRollout:
    def test_given_the_canary_matches_control_the_rollout_advances_to_the_next_step(self):
        c = CanaryController(steps=[1, 5, 25, 100], max_drop=0.02, min_samples=100)
        c.observe(control_successes=95, control_total=100, canary_successes=95, canary_total=100)
        assert c.step() == 5

    def test_given_the_canary_is_five_points_worse_the_rollout_rolls_back_to_zero(self):
        c = CanaryController(steps=[1, 5, 25, 100], max_drop=0.02, min_samples=100)
        c.observe(control_successes=95, control_total=100, canary_successes=90, canary_total=100)
        assert c.step() == 0
        assert c.rolled_back

    def test_given_too_few_canary_samples_the_rollout_waits(self):
        c = CanaryController(steps=[1, 5, 25, 100], max_drop=0.02, min_samples=100)
        c.observe(control_successes=950, control_total=1000, canary_successes=9, canary_total=10)
        assert c.step() == 1


class TestKillSwitch:
    def test_given_a_tenant_switched_off_only_that_tenant_is_blocked(self):
        ks = KillSwitch()
        ks.disable_tenant("acme")
        assert (ks.allowed("acme"), ks.allowed("globex")) == (False, True)

    def test_given_the_global_switch_every_tenant_is_blocked(self):
        ks = KillSwitch()
        ks.disable_all()
        assert not ks.allowed("globex")


class TestRateLimiting:
    def make(self):
        now = [0.0]
        return TokenBucket(rate_per_second=1.0, capacity=5, clock=lambda: now[0]), now

    def test_given_a_full_bucket_of_five_a_burst_of_five_actions_is_allowed(self):
        bucket, _ = self.make()
        assert [bucket.allow() for _ in range(5)] == [True] * 5

    def test_given_an_empty_bucket_the_sixth_action_is_refused(self):
        bucket, _ = self.make()
        for _ in range(5):
            bucket.allow()
        assert not bucket.allow()

    def test_given_two_seconds_of_refill_at_one_per_second_two_more_actions_are_allowed(self):
        bucket, now = self.make()
        for _ in range(5):
            bucket.allow()
        now[0] = 2.0
        assert [bucket.allow() for _ in range(3)] == [True, True, False]


class TestTamperEvidentAuditLog:
    def build(self):
        log = AuditLog()
        log.append(actor="refund-agent", action="refund", on_behalf_of="dana", inputs={"order": "A200", "amount": 40})
        log.append(actor="refund-agent", action="refund", on_behalf_of="lee", inputs={"order": "A201", "amount": 15})
        log.append(actor="refund-agent", action="escalate", on_behalf_of="sam", inputs={"order": "A300"})
        return log

    def test_given_an_untouched_log_verification_finds_no_tampering(self):
        assert self.build().verify() is None

    def test_given_someone_edits_an_amount_verification_points_at_that_entry(self):
        log = self.build()
        log.entries[1]["inputs"]["amount"] = 1500
        assert log.verify() == 1

    def test_given_someone_deletes_an_entry_verification_detects_the_break(self):
        log = self.build()
        del log.entries[1]
        assert log.verify() == 1

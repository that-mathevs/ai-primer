"""Specification for primer.ml.structured_output: constrained decoding that guarantees valid output."""

import json
import re

import numpy as np
import pytest

from primer.ml.structured_output import (
    AGE_PATTERN,
    AGE_REFERENCE,
    ENUM_PATTERN,
    ENUM_VOCAB,
    EOS,
    NO_AGE_REFERENCE,
    NULLABLE_AGE_SCHEMA,
    PERSON_REFERENCE,
    PERSON_SCHEMA,
    VOCAB,
    RegexConstraint,
    SchemaChecker,
    SchemaConstraint,
    ToyModel,
    allowed_tokens,
    chance_all_valid,
    chance_still_failing,
    compile_pattern,
    distortion_example,
    expected_attempts,
    forced_choice,
    mask_cost,
    mask_table,
    masked_softmax,
    sample_answer,
    tokenizations,
    validity_experiment,
)


def _parses(text):
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def _samples(n, constraint=None, seed=0):
    rng = np.random.default_rng(seed)
    model = ToyModel(AGE_REFERENCE)
    return [sample_answer(model, rng, constraint).text for _ in range(n)]


class TestAskingNicelyIsNotEnough:
    def test_given_ten_tokens_each_right_98_percent_of_the_time_the_whole_output_is_valid_82_percent_of_the_time(self):
        # 0.98 ** 10 = 0.817, worked by hand in the lesson.
        assert chance_all_valid(0.98, 10) == pytest.approx(0.817, abs=0.001)

    def test_given_a_longer_output_the_chance_that_every_token_is_right_falls(self):
        assert chance_all_valid(0.98, 100) < chance_all_valid(0.98, 10)

    def test_given_the_toy_model_unconstrained_a_sizeable_share_of_its_answers_break_the_pattern(self):
        # Python's own re module is the referee, independent of the lesson's machinery.
        texts = _samples(200)
        invalid = sum(re.fullmatch(AGE_PATTERN, t) is None for t in texts) / len(texts)
        assert 0.15 < invalid < 0.6

    def test_given_a_chatty_model_some_unconstrained_answers_open_with_a_preamble(self):
        assert any(t.startswith("Sure") for t in _samples(200))


class TestMaskingTheLogits:
    def test_given_logits_2_1_half_and_0_with_only_the_last_two_allowed_they_share_62_and_38_percent(self):
        # e^0.5 / (e^0.5 + e^0) = 1.649 / 2.649 = 0.622, the lesson's worked example.
        probs = masked_softmax(np.array([2.0, 1.0, 0.5, 0.0]), np.array([False, False, True, True]))
        assert probs == pytest.approx([0.0, 0.0, 0.622, 0.378], abs=0.001)

    def test_given_a_mask_every_forbidden_token_gets_exactly_zero_probability(self):
        probs = masked_softmax(np.array([9.0, -3.0, 4.0]), np.array([False, True, True]))
        assert probs[0] == 0.0

    def test_given_a_mask_the_allowed_tokens_keep_their_odds_against_each_other(self):
        # Masking removes options; it never reorders the ones left. Odds of 0.5 apart in logits stay e^0.5 = 1.65 to 1.
        probs = masked_softmax(np.array([2.0, 1.0, 0.5, 0.0]), np.array([False, False, True, True]))
        assert probs[2] / probs[3] == pytest.approx(1.6487, abs=1e-4)

    def test_given_the_shared_vocabulary_the_end_of_output_token_comes_last(self):
        assert VOCAB[-1] == EOS


class TestFromAPatternToAStateMachine:
    def test_given_the_enum_cat_car_dog_the_machine_accepts_exactly_those_three_words(self):
        dfa = compile_pattern(ENUM_PATTERN)
        words = ["cat", "car", "dog", "ca", "cats", "do", "cog", ""]
        assert [w for w in words if dfa.accepts(w)] == ["cat", "car", "dog"]

    def test_given_an_optional_sign_and_digits_the_machine_needs_three_states(self):
        # Start, "saw a minus", "saw at least one digit": drawn by hand in the lesson.
        assert compile_pattern("-?[0-9]+").n_states == 3

    def test_given_random_strings_the_machine_agrees_with_pythons_re_module(self):
        rng = np.random.default_rng(5)
        dfa = compile_pattern(AGE_PATTERN)
        alphabet = list('{}"age: 0123456789x')
        strings = ['{"age": 42}', '{"age": 7}', '{"age": }'] + [
            "".join(rng.choice(alphabet, size=rng.integers(0, 14))) for _ in range(300)
        ]
        assert [dfa.accepts(s) for s in strings] == [re.fullmatch(AGE_PATTERN, s) is not None for s in strings]


class TestWhichTokensMayComeNext:
    def test_given_the_start_state_a_multi_character_token_is_allowed_when_the_machine_can_consume_all_of_it(self):
        dfa = compile_pattern(ENUM_PATTERN)
        assert allowed_tokens(dfa, dfa.walk(dfa.start, ""), ENUM_VOCAB) == ["c", "d", "ca", "cat", "do", "dog"]

    def test_given_the_prefix_ca_only_t_and_r_may_follow(self):
        dfa = compile_pattern(ENUM_PATTERN)
        assert allowed_tokens(dfa, dfa.walk(dfa.start, "ca"), ENUM_VOCAB) == ["t", "r"]

    def test_given_a_finished_word_only_the_end_token_may_follow(self):
        dfa = compile_pattern(ENUM_PATTERN)
        assert allowed_tokens(dfa, dfa.walk(dfa.start, "dog"), ENUM_VOCAB) == [EOS]

    def test_given_a_precomputed_table_each_row_matches_checking_every_token_on_the_fly(self):
        dfa = compile_pattern(AGE_PATTERN)
        table = mask_table(dfa, VOCAB)
        on_the_fly = [[t in allowed_tokens(dfa, s, VOCAB) for t in VOCAB] for s in range(dfa.n_states)]
        assert table.tolist() == on_the_fly


class TestConstrainedDecodingWithAPattern:
    def test_given_the_age_pattern_every_constrained_answer_matches_it(self):
        texts = _samples(200, RegexConstraint(AGE_PATTERN, VOCAB))
        assert [t for t in texts if re.fullmatch(AGE_PATTERN, t) is None] == []

    def test_given_the_age_pattern_the_constrained_answers_never_open_with_a_preamble(self):
        assert not any(t.startswith("Sure") for t in _samples(200, RegexConstraint(AGE_PATTERN, VOCAB)))


class TestNestingNeedsAStack:
    def test_given_a_prefix_inside_three_containers_the_stack_holds_one_frame_for_each(self):
        assert SchemaChecker({}).open_containers('{"a": [1, {"b": ') == ["object", "array", "object"]

    def test_given_the_innermost_object_closes_its_frame_comes_off_the_stack(self):
        assert SchemaChecker({}).open_containers('{"a": [1, {"b": 2}') == ["object", "array"]

    def test_given_a_closing_bracket_that_does_not_match_the_last_one_opened_the_prefix_is_dead(self):
        # The array opened last, so only "]" may close it; "}" is a mismatch no continuation can repair.
        assert SchemaChecker({}).status('{"a": [1}') == "dead"


class TestCanThisPrefixStillBeCompleted:
    def test_given_a_prefix_that_can_still_be_finished_the_checker_says_open(self):
        assert SchemaChecker(PERSON_SCHEMA).status('{"name": "Ada", "ag') == "open"

    def test_given_a_whole_object_with_every_required_field_the_checker_says_complete(self):
        assert SchemaChecker(PERSON_SCHEMA).status('{"name": "Ada", "age": 36}') == "complete"

    def test_given_a_key_the_schema_does_not_list_the_prefix_dies_at_its_first_wrong_letter(self):
        checker = SchemaChecker(PERSON_SCHEMA)
        assert [checker.status(p) for p in ('{"na', '{"nx')] == ["open", "dead"]

    def test_given_a_required_field_still_missing_the_object_cannot_close(self):
        assert SchemaChecker(PERSON_SCHEMA).status('{"name": "Ada"}') == "dead"

    def test_given_an_integer_field_a_decimal_point_is_refused(self):
        assert SchemaChecker(PERSON_SCHEMA).status('{"age": 3.') == "dead"

    def test_given_a_number_json_forbids_a_leading_zero(self):
        assert SchemaChecker(PERSON_SCHEMA).status('{"age": 01') == "dead"

    def test_given_an_enum_only_the_listed_words_can_be_spelled(self):
        checker = SchemaChecker(PERSON_SCHEMA)
        assert [checker.status(p) for p in ('{"pets": ["c', '{"pets": ["x')] == ["open", "dead"]

    def test_given_any_json_every_prefix_of_a_valid_document_stays_alive(self):
        docs = [{"a": [1, {"b": 2}], "c": "d"}, [True, False, None, -3.25], {"x": {"y": {"z": []}}}, "hi", 0]
        checker = SchemaChecker({})
        texts = [json.dumps(d) for d in docs]
        assert [t[:i] for t in texts for i in range(len(t) + 1) if checker.status(t[:i]) == "dead"] == []

    def test_given_random_text_whatever_the_checker_calls_complete_python_can_parse(self):
        # json.loads is the independent referee: "complete" must never be a false promise.
        rng = np.random.default_rng(11)
        checker = SchemaChecker({})
        alphabet = list('{}[]":, 0123-.abtruefalsn')
        texts = ["".join(rng.choice(alphabet, size=rng.integers(1, 12))) for _ in range(3000)]
        complete = [t for t in texts if checker.status(t) == "complete"]
        assert len(complete) > 20
        assert [t for t in complete if not _parses(t)] == []


class TestConstrainedDecodingWithASchema:
    def test_given_the_person_schema_every_constrained_answer_that_finishes_parses_and_fits_the_schema(self):
        from primer.agents.tools import validate

        rng = np.random.default_rng(0)
        model = ToyModel(PERSON_REFERENCE)
        constraint = SchemaConstraint(PERSON_SCHEMA, VOCAB)
        answers = [sample_answer(model, rng, constraint) for _ in range(30)]
        assert [a.text for a in answers if a.finished and validate(json.loads(a.text), PERSON_SCHEMA)] == []

    def test_given_a_token_budget_too_small_the_answer_is_cut_off_valid_so_far_but_unfinished(self):
        # The guarantee covers every prefix; only finishing makes a whole value. Budgets must leave room.
        rng = np.random.default_rng(0)
        answer = sample_answer(ToyModel(PERSON_REFERENCE), rng, SchemaConstraint(PERSON_SCHEMA, VOCAB), max_tokens=5)
        assert not answer.finished
        assert SchemaChecker(PERSON_SCHEMA).status(answer.text) == "open"

    def test_given_a_finished_value_only_the_end_token_is_allowed(self):
        mask = SchemaConstraint(PERSON_SCHEMA, VOCAB).mask('{"name": "Ada", "age": 36}')
        assert [t for t, ok in zip(VOCAB, mask) if ok] == [EOS]


class TestWhatTheMaskCostsInQuality:
    def test_given_a_model_that_wanted_null_forcing_an_integer_throws_away_80_percent_of_its_belief(self):
        # null 0.80, "3" 0.12, "5" 0.08; only digits allowed. By hand: 0.12 / 0.20 = 0.6.
        kept, removed = forced_choice({"null": 0.80, "3": 0.12, "5": 0.08}, {"3", "5"})
        assert removed == pytest.approx(0.80)
        assert kept == pytest.approx({"3": 0.6, "5": 0.4})

    def test_given_a_schema_that_admits_null_the_honest_answer_is_allowed_again(self):
        assert SchemaChecker(NULLABLE_AGE_SCHEMA).status('{"age": null}') == "complete"

    def test_given_a_locally_tempting_first_token_masking_commits_to_it_nine_times_in_ten(self):
        # A: 0.9 then y: 0.01; B: 0.1 then y: 0.9. Only answers ending in y are valid.
        assert distortion_example()["masked"]["Ay"] == pytest.approx(0.9)

    def test_given_the_same_model_its_own_odds_among_valid_answers_give_that_path_only_9_percent(self):
        # P(Ay | valid) = 0.9 * 0.01 / (0.9 * 0.01 + 0.1 * 0.9) = 0.009 / 0.099 = 0.0909.
        assert distortion_example()["conditional"]["Ay"] == pytest.approx(0.0909, abs=1e-4)


class TestTokenBoundaries:
    def test_given_the_text_42_the_toy_vocabulary_can_spell_it_two_ways(self):
        assert sorted(tokenizations("42", VOCAB)) == [["4", "2"], ["42"]]

    def test_given_a_whole_answer_the_mask_allows_many_spellings_the_model_rarely_saw(self):
        assert len(tokenizations(AGE_REFERENCE, VOCAB)) > 10


class TestWhatTheMaskCostsInTime:
    def test_given_a_hundred_steps_the_naive_mask_walks_every_character_of_every_token_every_step(self):
        # n * |V| * L = 100 * 1000 * 4.
        assert mask_cost(n_steps=100, vocab_size=1000, avg_len=4, n_states=10)["naive"] == 400_000

    def test_given_a_precomputed_table_the_work_is_paid_once_per_state_plus_one_row_read_per_step(self):
        # S * |V| * L + n * |V| = 10 * 1000 * 4 + 100 * 1000.
        assert mask_cost(n_steps=100, vocab_size=1000, avg_len=4, n_states=10)["table"] == 140_000


class TestWhenRetryingIsEnough:
    def test_given_answers_valid_95_percent_of_the_time_retrying_needs_1_05_attempts_on_average(self):
        assert expected_attempts(0.95) == pytest.approx(1.053, abs=0.001)

    def test_given_answers_valid_two_times_in_three_three_tries_all_fail_about_4_percent_of_the_time(self):
        # The toy model's 66.5% on the age answer: (1 - 0.665) ** 3 = 0.0376.
        assert chance_still_failing(0.665, 3) == pytest.approx(0.0376, abs=1e-4)

    def test_given_a_longer_answer_the_unconstrained_model_breaks_it_more_often(self):
        rows = {(r["task"], r["constrained"]): r for r in validity_experiment(n=100)}
        assert rows[("person schema", False)]["valid"] < rows[("age pattern", False)]["valid"]

    def test_given_constrained_decoding_no_finished_answer_breaks_the_rules_on_either_task(self):
        rows = [r for r in validity_experiment(n=100) if r["constrained"]]
        assert [(r["task"], r["chatter"], r["broken"]) for r in rows] == [("age pattern", 0, 0), ("person schema", 0, 0)]


class TestJsonModeAndToolArguments:
    def test_given_json_mode_every_finished_answer_parses_even_when_a_required_field_is_missing(self):
        rng = np.random.default_rng(0)
        model = ToyModel(NO_AGE_REFERENCE)
        answers = [sample_answer(model, rng, SchemaConstraint({}, VOCAB)) for _ in range(20)]
        parsed = [json.loads(a.text) for a in answers if a.finished]
        assert any("age" not in p for p in parsed if isinstance(p, dict))

    def test_given_the_schema_a_model_that_wanted_to_skip_age_is_made_to_write_one(self):
        rng = np.random.default_rng(0)
        model = ToyModel(NO_AGE_REFERENCE)
        answers = [sample_answer(model, rng, SchemaConstraint(PERSON_SCHEMA, VOCAB)) for _ in range(20)]
        assert all("age" in json.loads(a.text) for a in answers if a.finished)

    def test_given_payment_arguments_the_shape_can_be_guaranteed_while_the_amount_is_still_wrong(self):
        from primer.agents.tools import PAYMENT_SCHEMA, validate

        # Decoding enforces shape; "minimum" is a business rule it leaves to validation after.
        arguments = '{"amount": 0, "currency": "USD"}'
        assert SchemaChecker(PAYMENT_SCHEMA).status(arguments) == "complete"
        assert validate(json.loads(arguments), PAYMENT_SCHEMA) == ["$.amount: must be >= 0.01, got 0"]

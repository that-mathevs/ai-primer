"""Specification for primer.ml.tokenization: how text becomes the integers a model reads."""

import importlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from primer.ml.tokenization import (
    TRAINING_TEXT,
    ByteBPE,
    encode_char_bpe,
    estimate_cost,
    estimate_tokens,
    tokens_to_words,
    train_char_bpe,
    trained_tokenizer,
    wordpiece_scores,
)

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


@pytest.fixture(scope="module")
def tok() -> ByteBPE:
    return trained_tokenizer()


class TestBytePairEncodingTraining:
    def test_given_low_lower_lowest_the_first_merge_joins_l_and_o_seen_three_times(self):
        # l+o and o+w are tied at 3; the earliest pair in the text wins the tie.
        step = train_char_bpe("low lower lowest", 1)[0]
        assert (step.pair, step.count) == (("l", "o"), 3)

    def test_given_low_lower_lowest_the_second_merge_joins_lo_and_w_seen_three_times(self):
        step = train_char_bpe("low lower lowest", 2)[1]
        assert (step.pair, step.count) == (("lo", "w"), 3)

    def test_given_low_lower_lowest_the_third_merge_joins_low_and_e_seen_twice(self):
        step = train_char_bpe("low lower lowest", 3)[2]
        assert (step.pair, step.count) == (("low", "e"), 2)

    def test_after_three_merges_the_text_reads_low_lowe_r_lowe_s_t(self):
        assert train_char_bpe("low lower lowest", 3)[-1].text_after == "low · lowe r · lowe s t"

    def test_given_every_word_is_already_one_token_training_stops_early(self):
        assert len(train_char_bpe("a b a", 5)) == 0


class TestBytePairEncodingEncoding:
    MERGES = [("l", "o"), ("lo", "w"), ("low", "e")]

    def test_given_the_learned_merges_lowest_encodes_as_lowe_s_t(self):
        assert encode_char_bpe("lowest", self.MERGES) == ["lowe", "s", "t"]

    def test_given_an_unseen_word_the_learned_merges_still_apply_inside_it(self):
        assert encode_char_bpe("slow", self.MERGES) == ["s", "low"]

    def test_given_a_word_no_merge_applies_to_it_stays_as_characters(self):
        assert encode_char_bpe("cat", self.MERGES) == ["c", "a", "t"]


class TestWordPiece:
    def test_given_low_lower_lowest_wordpiece_prefers_s_and_t_which_never_appear_apart(self):
        # score = count(st) / (count(s)·count(t)) = 1 / (1·1) = 1.0, the highest of any pair.
        scores = wordpiece_scores("low lower lowest")
        assert max(scores, key=scores.get) == ("s", "t")
        assert scores[("s", "t")] == pytest.approx(1.0)

    def test_given_low_lower_lowest_wordpiece_scores_l_and_o_at_one_third(self):
        # count(lo)=3, count(l)=3, count(o)=3 -> 3/9.
        assert wordpiece_scores("low lower lowest")[("l", "o")] == pytest.approx(1 / 3)


class TestByteLevelBPE:
    def test_given_no_training_the_vocabulary_is_the_256_byte_values(self):
        assert ByteBPE().vocab_size == 256

    def test_given_an_untrained_tokenizer_each_utf8_byte_is_one_token(self):
        # "é" is 2 bytes in UTF-8, "🙂" is 4.
        assert len(ByteBPE().encode("é🙂")) == 6

    @pytest.mark.parametrize("text", ["naïve café 🙂", "東京タワー", "def f(x): return x**2", "\x00\x7f"])
    def test_given_any_text_decoding_the_encoding_returns_it_unchanged(self, tok, text):
        assert tok.decode(tok.encode(text)) == text

    def test_given_training_frequent_words_become_single_tokens(self, tok):
        assert len(tok.encode(" password")) == 1

    def test_given_training_the_vocabulary_grows_to_the_requested_size(self, tok):
        assert tok.vocab_size == 400


class TestTokenizerQuirks:
    def test_given_a_leading_space_cat_becomes_a_different_token(self, tok):
        # The pre-tokenizer glues the space onto the word, so " cat" and "cat" get unrelated ids.
        assert tok.encode(" cat") != tok.encode("cat")

    def test_given_1234_is_frequent_mid_sentence_it_is_one_token_while_12345_takes_two(self, tok):
        # The corpus only ever says " 1234", so " 12345" reuses that token plus a leftover "5".
        assert (tok.tokens(" 1234"), tok.tokens(" 12345")) == ([" 1234"], [" 1234", "5"])

    def test_given_the_same_number_without_its_leading_space_it_fragments_into_pieces(self, tok):
        assert len(tok.encode("1234")) > 1

    def test_given_strawberry_the_model_sees_fewer_tokens_than_its_ten_letters(self, tok):
        assert len(tok.encode(" strawberry")) < 10

    def test_given_strawberry_mid_sentence_it_splits_into_straw_and_berry(self, tok):
        # The lesson's table and demo: the corpus says " straw" and "berry" often, so two bricks, no letters.
        assert tok.tokens(" strawberry") == [" straw", "berry"]

    def test_given_an_english_trained_tokenizer_hindi_costs_more_tokens_per_character(self, tok):
        english = "The weather is nice today and we will go for a walk."
        hindi = "आज मौसम अच्छा है और हम टहलने जाएंगे।"
        assert len(tok.encode(hindi)) / len(hindi) > 2 * len(tok.encode(english)) / len(english)


class TestRulesOfThumb:
    def test_given_400_characters_of_prose_the_estimate_is_100_tokens(self):
        assert estimate_tokens("x" * 400) == 100

    def test_given_1000_tokens_the_text_is_about_750_words(self):
        assert tokens_to_words(1000) == 750

    def test_given_2000_input_and_500_output_tokens_at_3_and_15_dollars_per_million_the_call_costs_1_35_cents(self):
        # 2000/1e6·3 + 500/1e6·15 = 0.006 + 0.0075
        assert estimate_cost(2000, 500, 3, 15) == pytest.approx(0.0135)


class TestAgreementWithTiktoken:
    def test_given_english_prose_a_production_tokenizer_averages_three_to_six_characters_per_token(self):
        tiktoken = pytest.importorskip("tiktoken")
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:  # the encoding file is downloaded on first use; no network means skip
            pytest.skip("cl100k_base not cached locally")
        prose = "Pricing, latency and context limits are all measured in tokens, not words. " * 5
        assert 3 < len(prose) / len(enc.encode(prose)) < 6

    def test_given_a_production_tokenizer_a_leading_space_changes_the_token(self):
        tiktoken = pytest.importorskip("tiktoken")
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pytest.skip("cl100k_base not cached locally")
        assert enc.encode(" hello") != enc.encode("hello")


def _widget(expression: str) -> object:
    """Evaluate `expression` in Node with the widget's module as `m` and the lesson's data as `data`."""
    from primer.ml.tokenization import viz_data

    script = (
        "const m = require('./docs/assets/viz/tokenizer.js');"
        f"const data = {json.dumps(viz_data()['tokenizer'])};"
        f"console.log(JSON.stringify({expression}));"
    )
    return json.loads(subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True, cwd=ROOT).stdout)


# Seen and unseen English, contractions, digits, runs of spaces, accents, emoji,
# another script, whitespace Python and JavaScript disagree about (\x1f, \ufeff),
# and nothing at all.
WIDGET_TEXTS = [
    "The user asked the model to reset the password.",
    "Reset your password in the portal.",
    " strawberry",
    "Zyxwvut qwerty!! It's 12345 o'clock   now \t",
    "naïve café 🙂",
    "東京タワー",
    "\ufeffzero\x1fwidth\x85gap",
    "",
]


@pytest.mark.skipif(NODE is None, reason="needs Node to run the widget's tokenizer")
class TestTheTokenizerWidgetAgreesWithTheLesson:
    @pytest.mark.parametrize("merges", [0, 1, 10, 60, 144])
    def test_given_the_first_merges_the_widget_splits_text_into_the_pieces_a_tokenizer_trained_that_far_gives(self, merges):
        # BPE training is greedy, so a tokenizer trained to 256 + n entries knows
        # exactly the lesson's first n merges: the widget's slider, in Python.
        expected = [ByteBPE().train(TRAINING_TEXT, 256 + merges).tokens(t) for t in WIDGET_TEXTS]
        got = _widget(f"{json.dumps(WIDGET_TEXTS)}.map(t => m.tokenize(t, data.merges.slice(0, {merges})))")
        assert got == expected

    def test_given_no_merges_every_character_of_plain_english_is_its_own_token(self):
        assert _widget("m.tokenize(' the cat', [])") == [" ", "t", "h", "e", " ", "c", "a", "t"]

    def test_given_every_merge_the_lesson_learns_the_password_is_one_token(self):
        assert _widget("m.tokenize(' password', data.merges)") == [" password"]

    def test_given_an_emoji_the_widget_counts_characters_per_token_as_the_lesson_does(self):
        # Two emoji are two characters but eight UTF-8 bytes; JavaScript's .length would say four.
        assert _widget("m.charsPerToken('🙂🙃', data.merges)") == pytest.approx(2 / 8)


class TestTheTokenizerWidgetData:
    def test_given_the_lesson_its_widget_data_holds_every_learned_merge_in_order(self):
        from primer.ml.tokenization import viz_data

        merges = viz_data()["tokenizer"]["merges"]
        # 400 entries = 256 bytes + 144 merges; the first merge glues "h" (104) and "e" (101).
        assert (len(merges), merges[0]) == (144, [104, 101])

    def test_given_the_lesson_its_widget_data_is_small_plain_json(self):
        from primer.ml.tokenization import viz_data

        assert len(json.dumps(viz_data())) < 50_000

    def test_given_the_lesson_its_docstring_places_the_tokenizer_widget(self):
        doc = importlib.import_module("primer.ml.tokenization").__doc__
        assert '<div class="viz" data-viz="tokenizer" aria-label="' in doc

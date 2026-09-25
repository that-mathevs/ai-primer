"""Specification for the shared glossary and the hover-definition step of the docs site."""

import importlib
import json

import pytest

from primer.glossary import GLOSSARY, glossary_js
from tools.docsite import annotate_html


class TestTheGlossary:
    def test_given_every_entry_its_definition_is_plain_prose_of_at_most_three_sentences(self):
        # A tooltip longer than three sentences stops being a tooltip; the lesson is where depth lives.
        long = [t for t, e in GLOSSARY.items() if e.definition.count(". ") > 2]
        assert long == []

    def test_given_every_entry_its_lesson_is_a_real_module(self):
        for term, entry in GLOSSARY.items():
            if entry.lesson:
                importlib.import_module(entry.lesson)  # raises if the lesson doesn't exist

    def test_given_the_glossary_its_keys_are_lowercase_so_lookups_ignore_case(self):
        assert all(term == term.lower() for term in GLOSSARY)


class TestGlossaryForTheBrowser:
    def test_given_the_glossary_the_script_defines_one_global_the_pages_can_read(self):
        js = glossary_js()
        assert js.startswith("window.PRIMER_GLOSSARY = ")
        data = json.loads(js.removeprefix("window.PRIMER_GLOSSARY = ").rstrip(";\n"))
        assert data["softmax"]["lesson"] == "primer/ml/attention.html"


class TestHoverDefinitions:
    TERMS = {
        "attention": ("Each token looks at others.", "primer/ml/attention.html"),
        "multi-head attention": ("Several attentions in parallel.", "primer/ml/attention.html"),
        "softmax": ("Turns scores into shares.", "primer/ml/attention.html"),
    }

    def annotate(self, html, page="primer/ml/transformer.html"):
        return annotate_html(html, self.TERMS, page)

    def test_given_a_term_in_prose_its_first_mention_gets_a_hover_definition(self):
        out = self.annotate("<p>Softmax is used twice: softmax again.</p>")
        assert out.count('class="gl-term"') == 1
        assert 'data-tip="Turns scores into shares."' in out

    def test_given_a_longer_term_containing_a_shorter_one_the_longer_term_wins(self):
        out = self.annotate("<p>We use multi-head attention here.</p>")
        assert "Several attentions in parallel." in out
        assert "Each token looks at others." not in out

    def test_given_a_term_inside_code_or_a_heading_it_is_left_alone(self):
        # Wrapping code would corrupt it, and headings are navigation, not prose.
        html = "<h2>Softmax</h2><pre>softmax(x)</pre><code>softmax</code>"
        assert self.annotate(html) == html

    def test_given_a_term_inside_display_math_it_is_left_alone(self):
        # Wrapping part of a formula in a span breaks the math typesetter.
        html = "<p>$$\\text{softmax}(z)_i = e^{z_i}$$</p>"
        assert self.annotate(html) == html

    def test_given_a_term_inside_inline_math_it_is_left_alone_but_prose_beside_it_is_not(self):
        out = self.annotate("<p>the value $\\text{softmax}(x)$ comes from softmax</p>")
        assert "$\\text{softmax}(x)$" in out
        assert out.count('class="gl-term"') == 1 and out.endswith('softmax</span></p>')

    def test_given_display_math_split_across_tags_every_part_is_left_alone(self):
        html = "<p>$$ a = </p><p>\\text{softmax}(b) $$</p><p>softmax</p>"
        out = self.annotate(html)
        assert "<p>\\text{softmax}(b) $$</p>" in out and out.count('class="gl-term"') == 1

    def test_given_a_term_inside_a_link_it_is_left_alone(self):
        html = '<p><a href="x.html">softmax</a></p>'
        assert self.annotate(html) == html

    def test_given_a_term_as_part_of_a_longer_word_it_is_not_matched(self):
        assert 'class="gl-term"' not in self.annotate("<p>attentional drift</p>")

    def test_given_the_page_depth_the_lesson_link_is_relative_to_the_page(self):
        out = self.annotate("<p>softmax</p>", page="primer/ml/embeddings/ann.html")
        assert 'data-lesson="../attention.html"' in out


class TestScopedTerms:
    # "policy" means the model being trained in preference tuning, but a travel policy elsewhere.
    TERMS = {"policy": ("The model being trained.", "primer/ml/training_stages.html", ("primer/ml/training_stages",))}

    def test_given_a_scoped_term_on_a_page_inside_its_scope_it_gets_a_hover_definition(self):
        out = annotate_html("<p>the policy drifts</p>", self.TERMS, "primer/ml/training_stages.html")
        assert 'class="gl-term"' in out

    def test_given_a_scoped_term_on_a_page_outside_its_scope_it_is_left_alone(self):
        html = "<p>the travel policy says</p>"
        assert annotate_html(html, self.TERMS, "primer/agents/rag.html") == html

    def test_given_the_glossary_everyday_words_with_a_technical_meaning_are_scoped(self):
        for word in ("key", "value", "query", "rank", "recall", "policy", "patch", "filter", "kernel", "padding", "checkpoint", "span", "trace"):
            assert GLOSSARY[word].scope, f"'{word}' has an everyday meaning; scope it to the lessons that use the technical one"

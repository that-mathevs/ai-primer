"""The published site sends every old address to where the lesson or paper is read now, rumblr.com."""

from __future__ import annotations

from primer.curriculum import CURRICULUM
from tools.redirect_site import RUMBLR, redirect_page, redirects


class TestTheRedirects:
    def test_given_every_lesson_its_old_page_redirects_to_its_lesson_on_rumblr(self):
        pages = redirects()
        for lesson in CURRICULUM:
            slug = lesson.module.rsplit(".", 1)[-1].replace("_", "-")
            assert pages[lesson.module.replace(".", "/") + ".html"] == f"{RUMBLR}learn/{slug}", lesson.module

    def test_given_every_annotated_paper_its_old_page_redirects_to_its_copy_on_rumblr(self):
        pages = redirects()
        assert pages["papers/adam.html"] == f"{RUMBLR}papers/primer/adam"
        assert pages["papers/index.html"] == f"{RUMBLR}papers/"

    def test_given_an_address_the_old_site_never_had_the_reader_still_lands_on_rumblr(self):
        # GitHub Pages serves 404.html for any unknown path.
        assert redirects()["404.html"] == RUMBLR and redirects()["index.html"] == RUMBLR

    def test_given_a_redirect_page_it_forwards_at_once_and_offers_a_plain_link(self):
        page = redirect_page("https://rumblr.com/learn/attention")
        assert 'http-equiv="refresh" content="0; url=https://rumblr.com/learn/attention"' in page
        assert 'href="https://rumblr.com/learn/attention"' in page and 'rel="canonical"' in page

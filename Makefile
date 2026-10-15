PY ?= python3
# Tiny matrices run fastest single-threaded; multi-threaded BLAS thrashes on them.
# One export per line: make's `export A=1 B=2` would set A to the string "1 B=2".
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1
MODULES = $(shell find primer -name '*.py' ! -name '__init__.py' ! -name '_*' | sed 's|/|.|g; s|\.py$$||' | sort)

.PHONY: test spec readme demos figures docs redirects sitecheck browsercheck gate links clean

test:  ## run the full suite
	$(PY) -m pytest

spec:  ## print the suite as a readable specification
	$(PY) tools/spec.py

readme:  ## regenerate README's reading order from primer/curriculum.py
	$(PY) -m primer.curriculum

demos:  ## run every lesson's walkthrough
	@for m in $(MODULES); do \
	  if grep -q '^def demo' $$(echo $$m | tr . /).py; then \
	    echo "\n########## $$m"; $(PY) -m $$m > /dev/null && echo "ok" || exit 1; \
	  fi; done

figures:  ## render every lesson's figures() into docs/figures and check each is explained
	$(PY) tools/figures.py

redirects:  ## the published site: one redirect to rumblr.com per page the old site had -> docs/html
	$(PY) tools/redirect_site.py

docs: readme  ## a local HTML preview of the lessons -> docs/html (not published; rumblr.com is; uses pdoc, or uvx pdoc)
	$(PY) tools/docsite.py

sitecheck:  ## crawl the built site and fail on any broken internal link (run after make docs)
	$(PY) tools/sitecheck.py

browsercheck:  ## load every built page in headless Chrome: phone width, keyboard and screen-reader checks (after make docs)
	$(PY) tools/browsercheck.py

gate: test docs sitecheck browsercheck  ## everything a push to main must pass, in order, stopping at the first failure (stage new files first)

links:  ## check every URL in lessons, README and the paper catalog (needs network)
	$(PY) tools/links.py

clean:
	rm -rf docs/html .pytest_cache **/__pycache__

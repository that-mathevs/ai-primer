"""
# Part 2: building systems people rely on

Everything here runs offline against `primer.agents.llm.ScriptedLLM`, a
deterministic stand-in for a model, so every failure mode can be reproduced
on purpose. Swap in `primer.agents.llm.ClaudeLLM` to run the same code
against a real model.
"""

from primer.curriculum import reading_list as _reading_list

__doc__ += _reading_list("primer.agents")

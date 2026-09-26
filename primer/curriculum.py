"""
# Curriculum: the map of every lesson

The single source of truth for lesson order, titles and what each lesson
teaches. Everything navigational is generated from `CURRICULUM`:

* the reading-order tables in `README.md` (`make readme`; a test fails when
  they drift);
* the reading lists in each package's page (`primer.ml`, `primer.agents`, …);
* the site's home page, breadcrumbs and previous/next links (`make docs`).

To add a lesson, add one `Lesson` here, in reading order. Nothing else needs
editing by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

# Where the project lives. The README needs absolute links to the site (the
# built HTML isn't committed). The site builder reads the repository's address
# from git or GitHub Actions instead, so a fork's site links to the fork; this
# constant is only its fallback.
REPO_URL = "https://github.com/that-mathevs/ai-primer"
SITE_URL = "https://that-mathevs.github.io/ai-primer/"
BRANCH = "main"


def source_path(module: str) -> str:
    """The file a module lives in, relative to the repository root."""
    return module.replace(".", "/") + ".py"


def link_code_references(markdown: str, from_dir: str) -> str:
    """Make every code span that names something in this repository a link GitHub can follow.

    GitHub renders Markdown but not docstrings, so in a .md file a reference to
    code only helps a reader if it's a relative link. Linked, when written as code:

    * a module, or a name inside one: `primer.agents.llm`, `primer.agents.llm.ClaudeLLM`;
    * a command that runs a lesson: `python -m primer.ml.attention`;
    * a committed file or folder, written from the repository root (`primer/glossary.py`,
      `CLAUDE.md`) or from the Markdown file's own folder;
    * a class defined in exactly one module: `ScriptedLLM`.

    Anything else (build output such as `docs/html/index.html`, commands, other
    projects' names) stays plain. Existing links and fenced code blocks are untouched.

    Args:
        markdown: the text to link.
        from_dir: the directory the Markdown file sits in, relative to the repository root.
    """
    import os
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    here = root / from_dir

    def rel(target: Path) -> str:
        return os.path.relpath(target, here).replace(os.sep, "/")

    def target_for(code: str) -> Path | None:
        run = re.fullmatch(r"python -m (primer(?:\.\w+)+)", code)
        dotted = run.group(1) if run else code
        if re.fullmatch(r"primer(?:\.\w+)+", dotted):
            return _module_file(dotted)
        if re.fullmatch(r"[\w.-]+(?:/[\w.-]+)*/?", code) and ("/" in code or "." in code):
            for base in (root, here):
                if _is_committed(base / code):
                    return base / code
            return None
        if re.fullmatch(r"[A-Z]\w+(?:\(\))?", code):
            homes = _class_homes().get(code.removesuffix("()"), ())
            return _module_file(homes[0]) if len(homes) == 1 else None
        return None

    def link(m: re.Match) -> str:
        target = target_for(m.group(1))
        return f"[`{m.group(1)}`]({rel(target)})" if target else m.group(0)

    # A code span that is already a link's text is followed by "]"; one inside [..] is preceded by "[".
    span = re.compile(r"(?<!\[)`([^`\n]+)`(?!\])")
    pieces = re.split(r"(```.*?```)", markdown, flags=re.S)
    return "".join(piece if i % 2 else span.sub(link, piece) for i, piece in enumerate(pieces))


def _module_file(name: str):
    """The file that defines a dotted name: primer.agents.llm.ClaudeLLM -> primer/agents/llm.py."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        base = root.joinpath(*parts[:i])
        if base.with_suffix(".py").is_file():
            return base.with_suffix(".py")
        if (base / "__init__.py").is_file():
            return base / "__init__.py"
    return None


_COMMITTED: set[str] | None = None


def _is_committed(path) -> bool:
    """Whether GitHub will have this file or folder: it is tracked by git (or exists, outside a checkout)."""
    import subprocess
    from pathlib import Path

    global _COMMITTED
    root = Path(__file__).resolve().parent.parent
    if _COMMITTED is None:
        out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True)
        _COMMITTED = set(out.stdout.split("\n")) if out.returncode == 0 else set()
    try:
        rel = Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return False
    if not _COMMITTED:
        return Path(path).exists()
    return rel in _COMMITTED or any(f.startswith(rel.rstrip("/") + "/") for f in _COMMITTED)


_CLASS_HOMES: dict[str, list[str]] | None = None


def _class_homes() -> dict[str, list[str]]:
    """Every public class in the package, mapped to the modules that define one by that name."""
    import importlib
    import inspect
    import pkgutil

    import primer

    global _CLASS_HOMES
    if _CLASS_HOMES is None:
        _CLASS_HOMES = {}
        for info in pkgutil.walk_packages(primer.__path__, "primer."):
            if info.name.rsplit(".", 1)[-1].startswith("_"):
                continue
            mod = importlib.import_module(info.name)
            for n, o in vars(mod).items():
                if inspect.isclass(o) and o.__module__ == info.name and not n.startswith("_"):
                    _CLASS_HOMES.setdefault(n, []).append(info.name)
    return _CLASS_HOMES


def tests_for(module: str) -> str:
    """The test file that specifies a lesson (tests/test_<name>.py, test_emb_ or test_agents_)."""
    name = module.rsplit(".", 1)[-1]
    prefix = "emb_" if module.startswith("primer.ml.embeddings.") else "agents_" if module.startswith("primer.agents.") else ""
    return f"tests/test_{prefix}{name}.py"


@dataclass(frozen=True)
class Part:
    key: str
    title: str
    blurb: str


@dataclass(frozen=True)
class Lesson:
    module: str
    title: str
    outcome: str  # "What you'll be able to explain"
    part: str  # Part.key


PARTS: list[Part] = [
    Part("start", "Before you begin", "The notation every formula in this primer uses, decoded as short loops."),
    Part("ml", "Part 1: how the model works inside", "From a single neuron to a working transformer, and how models are trained and served."),
    Part("embeddings", "Embeddings, the centerpiece", "Vectors that capture meaning, and the search systems built on them."),
    Part("generative", "Generating images, audio and video", "Autoencoders, GANs, diffusion, and the multimodal models that connect them to language."),
    Part("agents", "Part 2: building systems people rely on", "Agents, tools, retrieval, memory, evaluation, safety, cost and deployment."),
]

CURRICULUM: list[Lesson] = [
    Lesson("primer.notation", "Math notation, from zero", "Every symbol in an ML formula, as a short loop", "start"),
    Lesson("primer.ml.big_picture", "The big picture", "What happens, end to end, when you send a prompt", "ml"),
    Lesson("primer.ml.neural_net", "Neural networks", "Neurons, activations, the forward pass, backprop by hand", "ml"),
    Lesson("primer.ml.optimizers", "Optimizers", "SGD, momentum, Adam/AdamW, learning-rate warmup and decay", "ml"),
    Lesson("primer.ml.deep_nets", "Training deep networks", "Vanishing/exploding gradients, residuals, normalization, initialization", "ml"),
    Lesson("primer.ml.attention", "Attention", "Queries, keys, values, softmax, masking, multi-head, GQA, O(n²)", "ml"),
    Lesson("primer.ml.positional", "Positional information", "Why order must be added, sinusoids and RoPE", "ml"),
    Lesson("primer.ml.transformer", "The transformer", "The block, a tiny GPT, parameter counts, mixture of experts", "ml"),
    Lesson("primer.ml.tokenization", "Tokenization", "BPE from scratch, byte-level tokens, why models miscount letters", "ml"),
    Lesson("primer.ml.training_stages", "Training stages", "Pretraining, SFT, RLHF and DPO, LoRA, fine-tuning vs. RAG", "ml"),
    Lesson("primer.ml.pretraining", "Pretraining at scale", "Data curation and deduplication, parallelism across GPUs, mixed precision", "ml"),
    Lesson("primer.ml.fine_tuning", "Fine-tuning in practice", "Preparing data, forgetting old skills, merging models", "ml"),
    Lesson("primer.ml.reinforcement", "Reinforcement learning", "Policy gradients from scratch, PPO, GRPO, reward hacking", "ml"),
    Lesson("primer.ml.reasoning", "Reasoning models", "Chain of thought, test-time compute, verifiers, learning to reason with RL", "ml"),
    Lesson("primer.ml.alignment", "Alignment and safety", "Constitutional AI, red-teaming, sycophancy, refusals", "ml"),
    Lesson("primer.ml.hardware", "The hardware underneath", "GPUs, the memory hierarchy, FLOPs vs. bandwidth, number formats", "ml"),
    Lesson("primer.ml.inference", "Inference", "Prefill vs. decode, the KV cache, sampling, speculative decoding, memory math", "ml"),
    Lesson("primer.ml.structured_output", "Structured output", "Constrained decoding: grammars and JSON schemas that guarantee valid output", "ml"),
    Lesson("primer.ml.efficient_architectures", "Long context and efficient architectures", "Sliding-window and sparse attention, state-space models, KV-cache compression", "ml"),
    Lesson("primer.ml.losses", "Loss functions", "Cross-entropy, perplexity, MSE/MAE, contrastive losses", "ml"),
    Lesson("primer.ml.metrics", "Metrics", "Precision/recall/F1, ROC-AUC, recall@k, MRR, nDCG, BLEU/ROUGE", "ml"),
    Lesson("primer.ml.benchmarks", "Reading benchmarks", "What benchmarks measure, contamination, leaderboards and arenas", "ml"),
    Lesson("primer.ml.regularization", "Overfitting and regularization", "Overfitting, early stopping, dropout, L1/L2, leakage", "ml"),
    Lesson("primer.ml.classical", "Trees and boosting", "Decision trees, random forests, gradient boosting, and when they still win", "ml"),
    Lesson("primer.ml.cnn_rnn", "CNNs and RNNs", "How convolutions see and recurrent nets remember, and why transformers won", "ml"),
    Lesson("primer.ml.interpretability", "Looking inside the model", "Probes, the logit lens, activation patching, superposition, sparse autoencoders", "ml"),
    Lesson("primer.ml.embeddings.word2vec", "Word embeddings", "Where embeddings came from, analogies, the \"bank\" problem", "embeddings"),
    Lesson("primer.ml.embeddings.similarity", "Similarity", "Cosine vs. dot vs. distance, normalization, anisotropy, thresholds", "embeddings"),
    Lesson("primer.ml.embeddings.contrastive", "Training embedding models", "Contrastive learning, hard negatives, CLIP", "embeddings"),
    Lesson("primer.ml.embeddings.compression", "Dimensions and compression", "Storage math, Matryoshka truncation, int8 and binary quantization", "embeddings"),
    Lesson("primer.ml.embeddings.ann", "Vector indexes", "Flat, IVF, PQ and HNSW from scratch, recall vs. latency", "embeddings"),
    Lesson("primer.ml.embeddings.retrieval", "Retrieval", "BM25, hybrid search with RRF, rerankers, ColBERT, chunking", "embeddings"),
    Lesson("primer.ml.embeddings.clustering", "Clustering and matching", "k-means, density clustering, dedup, routing, semantic caching", "embeddings"),
    Lesson("primer.ml.embeddings.operations", "Embeddings in production", "Model migrations, domain mismatch, measuring retrieval on its own", "embeddings"),
    Lesson("primer.ml.generative.autoencoders", "Autoencoders and VAEs", "Squeezing data into a code and back, and sampling new data from it", "generative"),
    Lesson("primer.ml.generative.gans", "GANs", "A forger against a detective: adversarial training, and why it is unstable", "generative"),
    Lesson("primer.ml.generative.diffusion", "Diffusion and flow matching", "Turning noise into images one small step at a time", "generative"),
    Lesson("primer.ml.generative.multimodal", "Multimodal models", "Images, audio and video into a language model", "generative"),
    Lesson("primer.agents.llm", "Talking to a model", "The message format, and what tool calling really is", "agents"),
    Lesson("primer.agents.orchestration", "Orchestration", "Workflows vs. agents, and the named patterns", "agents"),
    Lesson("primer.agents.agent_loop", "The agent loop", "A production agent loop: budgets, loop detection, recovery", "agents"),
    Lesson("primer.agents.tools", "Tools", "Tool design, validation, idempotency, approvals, least privilege", "agents"),
    Lesson("primer.agents.coding_agents", "Coding and computer-use agents", "Edit, run, test, repeat; sandboxes; driving a screen", "agents"),
    Lesson("primer.agents.mcp", "Model Context Protocol", "MCP on the wire, and its security risks", "agents"),
    Lesson("primer.agents.rag", "Retrieval-augmented generation", "RAG end to end, with citations and access control", "agents"),
    Lesson("primer.agents.context", "Context engineering", "What goes in the window, compression, cache-friendly layout", "agents"),
    Lesson("primer.agents.memory", "Memory", "Short- and long-term memory, tenant isolation, forgetting", "agents"),
    Lesson("primer.agents.planning", "Planning", "Plan-and-execute, decomposition, reflection, compounding error", "agents"),
    Lesson("primer.agents.evals", "Evaluation", "Golden sets, graders, LLM-as-judge calibration", "agents"),
    Lesson("primer.agents.guardrails", "Guardrails", "Prompt injection and privilege separation, PII, output checks", "agents"),
    Lesson("primer.agents.cost", "Cost and latency", "Routing, caching, batching, budgets, cost per successful task", "agents"),
    Lesson("primer.agents.observability", "Observability", "Traces, OpenTelemetry GenAI attributes, the improvement loop", "agents"),
    Lesson("primer.agents.deployment", "Safe deployment", "Shadow mode, graduated autonomy, canaries, kill switches, audit logs", "agents"),
    Lesson("primer.agents.failures", "Why the hard ones fail", "The common failure modes, and the fix for each", "agents"),
]

_BY_MODULE = {l.module: i for i, l in enumerate(CURRICULUM)}


def neighbours(module: str) -> tuple[Lesson | None, Lesson | None]:
    """(previous lesson, next lesson) in reading order; None at either end."""
    i = _BY_MODULE[module]
    return (CURRICULUM[i - 1] if i > 0 else None, CURRICULUM[i + 1] if i + 1 < len(CURRICULUM) else None)


def lessons_in(part_key: str) -> list[tuple[int, Lesson]]:
    """(number, lesson) pairs for one part, numbered across the whole curriculum."""
    return [(i, l) for i, l in enumerate(CURRICULUM) if l.part == part_key]


def readme_section() -> str:
    """The README's reading-order tables. Regenerate with `make readme`."""
    out = []
    for part in PARTS:
        out.append(f"### {part.title}\n\n{part.blurb}\n\n| # | Lesson | What you'll be able to explain | Read |\n|---|---|---|---|")
        out += [
            f"| {i} | [{l.title}]({source_path(l.module)}) | {l.outcome} | "
            f"[page]({SITE_URL}{l.module.replace('.', '/')}.html) · [tests]({tests_for(l.module)}) |"
            for i, l in lessons_in(part.key)
        ]
        out.append("")
    return link_code_references("\n".join(out) + "\n", '.')


def reading_list(package: str) -> str:
    """A markdown reading list of the lessons inside `package`, for its page."""
    rows = [
        f"{i}. `{l.module}`: **{l.title}.** {l.outcome}."
        for i, l in enumerate(CURRICULUM)
        if l.module.startswith(package + ".") and "." not in l.module[len(package) + 1 :]
    ]
    return "\n## Reading order\n\n" + "\n".join(rows) + "\n\nThe full map is in `primer.curriculum`.\n"


# ---------------------------------------------------------------------------
# Big questions: the macro map. Lessons are organized bottom-up; real
# conversations about AI systems start top-down with questions like these.
# Each one lists the lessons that answer it, in order, and the short version:
# the main ideas, in the order that builds understanding.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BigQuestion:
    question: str
    route: tuple[str, ...]  # lesson modules, in the order to read them
    in_brief: tuple[str, ...]  # the short version: the main ideas, in the order that builds them


_ML, _EMB, _AG = "primer.ml.", "primer.ml.embeddings.", "primer.agents."
_GEN = "primer.ml.generative."

BIG_QUESTIONS: list[BigQuestion] = [
    BigQuestion(
        "What happens, step by step, when I send a prompt to a language model?",
        ("primer.notation", _ML + "big_picture", _ML + "tokenization", _ML + "attention", _ML + "positional", _ML + "transformer", _ML + "inference"),
        (
            "Tokenizer: text becomes subword IDs; cost and context limits are counted in tokens.",
            "Embedding lookup turns each ID into a vector; position information is mixed in.",
            "Dozens of transformer blocks: attention mixes information across tokens, the feed-forward layer processes each token.",
            "The last position's vector becomes a score for every vocabulary token; softmax turns scores into probabilities.",
            "Sampling (temperature, top-p) picks one token, which is appended; the loop repeats until a stop token.",
            "Prefill processes the prompt in parallel; decode generates one token at a time, made cheap by the KV cache.",
        ),
    ),
    BigQuestion(
        "How does a neural network actually learn?",
        (_ML + "neural_net", _ML + "losses", _ML + "optimizers", _ML + "deep_nets", _ML + "regularization"),
        (
            "A forward pass makes a prediction; a loss turns 'how wrong' into one number.",
            "Backpropagation applies the chain rule to find every weight's gradient.",
            "An optimizer (SGD, Adam/AdamW) steps each weight against its gradient; the learning rate sets the step size.",
            "Depth brings vanishing and exploding gradients; residual connections, normalization and good initialization fix them.",
            "Watch validation loss: when it rises while training loss falls, the model is overfitting; regularize or stop early.",
        ),
    ),
    BigQuestion(
        "How does attention work, and why did transformers replace RNNs?",
        (_ML + "attention", _ML + "positional", _ML + "transformer", _ML + "cnn_rnn"),
        (
            "Each token forms a query, key and value; query-key dot products score relevance; softmax turns scores into weights; the output blends values.",
            "Scores are divided by the square root of d_k so softmax doesn't saturate and gradients keep flowing.",
            "A causal mask hides future tokens, which makes next-token training honest and generation cacheable.",
            "Multi-head attention runs several attentions in parallel; grouped-query attention shares keys and values to shrink the KV cache.",
            "RNNs pass everything through one hidden state, one step at a time; attention gives every pair of tokens a direct path and trains in parallel.",
            "The price is O(n²) cost in sequence length, which FlashAttention, sparse attention and state-space models attack.",
        ),
    ),
    BigQuestion(
        "How are large language models trained, and when should I fine-tune instead of using RAG?",
        (_ML + "training_stages", _ML + "fine_tuning", _ML + "tokenization", _ML + "losses", _EMB + "operations", _AG + "rag"),
        (
            "Pretraining: next-token prediction over trillions of tokens produces a knowledgeable base model.",
            "Supervised fine-tuning teaches the assistant format; preference tuning (RLHF or DPO) shapes helpfulness and safety.",
            "Adaptation, cheapest first: prompting, then RAG, then LoRA, then (rarely) a full fine-tune.",
            "Fine-tuning changes behavior; RAG supplies knowledge that changes or must be cited.",
            "Distillation trains a small model to imitate a large one, often the biggest production cost win.",
        ),
    ),
    BigQuestion(
        "What makes serving a model fast and affordable?",
        (_ML + "hardware", _ML + "inference", _ML + "efficient_architectures", _ML + "attention", _AG + "cost", _AG + "context"),
        (
            "Prefill is compute-bound and sets time to first token; decode is memory-bound and sets tokens per second.",
            "The KV cache trades GPU memory for speed; its size is 2 × layers × KV heads × head dimension × bytes, per token.",
            "Memory math: weights = parameters × bytes per parameter (70B at 16-bit is about 140 GB).",
            "Speedups: quantization, continuous batching, speculative decoding, grouped-query attention, prompt caching.",
            "At the system level: route easy work to small models, cache stable prefixes, trim tokens, batch offline work.",
        ),
    ),
    BigQuestion(
        "What is an embedding, and how is an embedding model trained?",
        (_EMB + "word2vec", _EMB + "contrastive", _EMB + "similarity", _ML + "losses"),
        (
            "An embedding is a learned vector where closeness means similar meaning.",
            "word2vec learned one vector per word from co-occurrence; contextual models give each token a vector that depends on its sentence.",
            "Sentence embeddings pool token vectors; models trained for similarity beat plain pooled encoders.",
            "Contrastive training pulls matching pairs together and pushes others apart, using in-batch negatives (InfoNCE).",
            "Hard negatives (right topic, wrong answer) are the biggest driver of retrieval quality.",
            "CLIP applies the same idea across images and text, putting both in one space.",
        ),
    ),
    BigQuestion(
        "How do you search millions of vectors quickly, and what does it cost?",
        (_EMB + "similarity", _EMB + "compression", _EMB + "ann"),
        (
            "On normalized vectors, cosine, dot product and Euclidean distance give the same ranking; use what the model was trained with.",
            "Storage math: vectors × dimensions × 4 bytes (10M × 1536 is about 61 GB) before index overhead.",
            "Exact search is too slow at scale; approximate indexes trade a little recall for a lot of speed.",
            "HNSW: layered graph, long jumps on top, local search at the bottom; M, efConstruction and efSearch are the knobs.",
            "IVF searches only the nearest clusters (nprobe); PQ compresses vectors into codes.",
            "Matryoshka truncation and scalar or binary quantization shrink memory; rescoring the shortlist recovers accuracy.",
        ),
    ),
    BigQuestion(
        "How do you build retrieval that returns the right passages?",
        (_EMB + "retrieval", _EMB + "clustering", _EMB + "operations", _ML + "metrics", _AG + "rag"),
        (
            "Chunk on document structure, with overlap and metadata; chunking often matters more than the model.",
            "Dense search finds meaning; BM25 finds exact IDs and rare terms; hybrid search fuses both with reciprocal rank fusion.",
            "Retrieve wide with a bi-encoder, then rerank the shortlist with a cross-encoder.",
            "Measure retrieval on its own with recall@k, MRR and nDCG on a labeled set before tuning prompts.",
            "Operations: new embedding model means re-embedding everything; version indexes and switch traffic behind an alias.",
        ),
    ),
    BigQuestion(
        "How do you know if a model or an agent is any good?",
        (_ML + "metrics", _ML + "benchmarks", _ML + "losses", _ML + "regularization", _AG + "evals"),
        (
            "Pick metrics by the cost of each error: precision vs. recall; accuracy misleads on imbalanced data.",
            "Keep training, validation and test data apart, and watch for leakage and benchmark contamination.",
            "For agents: a golden set of real tasks, graded by code wherever possible (end state, schema, tests).",
            "For open-ended output: an LLM judge with an explicit rubric, calibrated against human labels.",
            "Track trajectory, cost and latency beside quality; run the suite on every change and feed production failures back in.",
        ),
    ),
    BigQuestion(
        "When should you build an agent, and how does one work?",
        (_AG + "llm", _AG + "orchestration", _AG + "agent_loop", _AG + "tools", _AG + "coding_agents", _AG + "mcp", _AG + "planning"),
        (
            "Use the least autonomy that solves the problem: fixed workflow, then router, then agent loop, then multiple agents.",
            "Tool calling: the model emits a structured request; your code validates it, runs it and returns the result. The model executes nothing.",
            "The loop: think, call a tool, observe, decide, with step limits, token budgets and loop detection.",
            "Tool design is prompt design: few, high-level tools with precise descriptions, validated arguments and actionable errors.",
            "Long tasks fail by compounding error (0.95¹⁰ ≈ 0.60), so plan, verify each step externally, and checkpoint.",
            "MCP standardizes how apps connect to tools, and brings its own risks: tool poisoning, rug pulls, broad permissions.",
        ),
    ),
    BigQuestion(
        "How do you give an AI system the right context and memory?",
        (_AG + "context", _AG + "memory", _AG + "rag"),
        (
            "Context engineering: the smallest set of high-signal content, structured with clear delimiters.",
            "More context is not better: models use the middle of long inputs least reliably, and quality rots as sessions grow.",
            "Put stable content first so prompt caching can reuse it; summarize or drop old turns; compress tool outputs.",
            "Long-term memory is retrieval over the system's own history: episodic, semantic and procedural.",
            "Memory must be isolated per tenant and user, updatable, and deletable on request.",
        ),
    ),
    BigQuestion(
        "How do you make an AI system safe to put in front of real users?",
        (_ML + "alignment", _AG + "guardrails", _AG + "tools", _AG + "deployment", _AG + "observability"),
        (
            "Layer guardrails on inputs, outputs and actions; no single check is reliable alone.",
            "Treat retrieved content, emails and tool outputs as untrusted: no prompt wording fully prevents injection.",
            "Privilege separation: the part that reads untrusted content holds no dangerous tools; actions pass a policy or human check.",
            "Graduate autonomy with evidence: shadow mode, then approval per action, then autonomy for low-risk actions.",
            "Trace every run, keep tamper-evident audit logs, rate-limit actions and keep a kill switch and a one-step rollback.",
        ),
    ),
    BigQuestion(
        "How do you cut cost and latency without hurting quality?",
        (_AG + "cost", _AG + "context", _ML + "inference", _EMB + "clustering"),
        (
            "Measure cost per successful task, not per call.",
            "Route each step to the cheapest model that handles it; this is usually the biggest lever.",
            "Cache: prompt caching for stable prefixes, response and semantic caches for repeated questions.",
            "Trim tokens: tight prompts, compressed tool outputs, only the top reranked chunks.",
            "Run independent tool calls in parallel, stream output, and move offline work to batch APIs.",
            "Enforce per-task and per-tenant budgets with anomaly alerts.",
        ),
    ),
    BigQuestion(
        "Why do AI systems fail in production, and how do you fix them?",
        (_AG + "failures", _ML + "structured_output", _AG + "planning", _AG + "rag", _AG + "evals", _AG + "observability"),
        (
            "Compounding error over long tasks: shorten paths, verify steps, checkpoint.",
            "Bad retrieval behind confident wrong answers: hybrid search, reranking, retrieval evals.",
            "Ambiguous tools, loops and runaway cost: better tool design, budgets, loop detection.",
            "Prompt injection and messy enterprise data: untrusted-content boundaries, permission-aware retrieval, investment in parsing.",
            "No evals and no traces: regressions ship silently; build the loop from production failure to trace to test case to fix.",
        ),
    ),
    BigQuestion(
        "What does it take to pretrain a large model?",
        (_ML + "training_stages", _ML + "tokenization", _ML + "pretraining", _ML + "optimizers", _ML + "hardware"),
        (
            "Most of a web crawl is thrown away: language ID, quality rules and classifiers, and exact and near-duplicate removal.",
            "Sources are mixed by weight, not size; about 20 tokens per parameter is compute-optimal, but models meant for heavy use train far longer.",
            "Adam in mixed precision needs about 16 bytes per parameter before activations, so one GPU can't hold a large model.",
            "Data parallelism shares gradients, ZeRO/FSDP shards the training state, tensor parallelism splits each matrix multiply, and pipeline parallelism splits the layers.",
            "The maths runs in bf16 or fp8 with scaling, while the master weights stay in fp32.",
            "Warmup, gradient clipping, spike rollback and regular checkpoints keep a months-long run alive.",
        ),
    ),
    BigQuestion(
        "How does a model learn from rewards instead of examples?",
        (_ML + "reinforcement", _ML + "training_stages", _ML + "reasoning", _ML + "alignment"),
        (
            "Reinforcement learning samples an action, scores it, and makes high-scoring actions more likely.",
            "A baseline turns rewards into advantages (better or worse than usual), which cuts noise without bias.",
            "PPO reuses each batch for several steps, clips how far the policy moves, and leashes it to a reference with a KL penalty.",
            "GRPO drops the value network by comparing rewards within a group of answers to the same prompt.",
            "A checker as the reward (the right answer, passing tests) is how reasoning models are trained.",
            "The policy optimises the reward you wrote, not the goal you meant; verifiable rewards, a KL leash and held-out checks defend against that.",
        ),
    ),
    BigQuestion(
        "How do reasoning models think, and when is extra thinking worth it?",
        (_ML + "inference", _ML + "reinforcement", _ML + "reasoning", _AG + "planning", _AG + "cost"),
        (
            "Every written token is another forward pass, so a chain of thought buys serial computation, and the text is the model's working memory.",
            "Test-time compute can go into one longer chain, or into many chains with a vote or a verifier picking one answer.",
            "Voting helps only when the right answer is the most common one and the samples' mistakes are independent.",
            "Checking each step catches errors that checking only the final answer misses.",
            "Training with verifiable rewards makes longer, self-checking reasoning emerge.",
            "Thinking is paid for per token and slips compound over long chains, so route easy tasks to little thinking and measure cost per successful task.",
        ),
    ),
    BigQuestion(
        "How do models handle very long contexts?",
        (_ML + "attention", _ML + "positional", _ML + "inference", _ML + "efficient_architectures"),
        (
            "Attention scores every pair of tokens, and the KV cache grows with every token, per layer, per conversation.",
            "Sliding windows and sparse patterns score fewer pairs; stacked layers still carry information far.",
            "Linear attention and state-space models keep a fixed-size summary: linear time and constant memory, but blurrier recall.",
            "Mamba makes the summary selective: each token decides how much to keep and how much to write.",
            "Hybrids keep a few attention layers for exact lookup.",
            "The cache shrinks by sharing key/value heads, caching a small latent, or storing fewer bits.",
        ),
    ),
    BigQuestion(
        "What is going on inside a trained model, and how can we tell?",
        (_ML + "transformer", _EMB + "word2vec", _ML + "interpretability", _ML + "alignment"),
        (
            "Models store features as directions across many neurons, not one feature per neuron.",
            "Probes and the logit lens read what is present; they show correlation, not use.",
            "Activation patching changes one activation and watches the output: the causal test.",
            "Sparse features get packed in superposition, which makes individual neurons respond to several things.",
            "Sparse autoencoders unpack superposition into interpretable features, at the cost of some unexplained activity.",
            "These tools give evidence, not proof; full explanations exist only for narrow behaviours.",
        ),
    ),
    BigQuestion(
        "When is a neural network the wrong tool?",
        (_ML + "classical", _ML + "neural_net", _ML + "regularization", _ML + "metrics"),
        (
            "On tables whose columns each mean something alone, gradient-boosted trees or a random forest are the model to beat.",
            "A tree asks one column at a time whether it's above a threshold, so it needs no feature scaling and handles categories natively.",
            "A single deep tree overfits; forests average many decorrelated trees, and boosting adds small trees fit to the remaining errors.",
            "Neural networks win when meaning lives in arrangements of raw values (images, audio, text), when data is huge, or when a pretrained model can be reused.",
            "Trees can't extrapolate beyond the values they trained on, and impurity importances credit noise, so check importances on held-out data.",
        ),
    ),
    BigQuestion(
        "How do AI models generate images, audio and video?",
        (_GEN + "autoencoders", _GEN + "gans", _GEN + "diffusion", _GEN + "multimodal"),
        (
            "A generator learns a whole distribution, so it can sample new examples; predicting the average gives blur.",
            "An autoencoder squeezes data into a small code and back; a VAE shapes that code so random codes decode to new data.",
            "A GAN trains a generator against a discriminator: sharp, one-pass samples, but unstable training and mode collapse.",
            "Diffusion adds noise on purpose and learns to remove it, generating from pure noise in many small steps.",
            "Flow matching learns straight paths from noise to data, so it needs fewer steps; guidance trades variety for following the prompt.",
            "Real systems denoise an autoencoder's latent with a transformer that reads the prompt; video and audio are the same idea with more tokens.",
        ),
    ),
    BigQuestion(
        "How do AI models see images and hear audio?",
        (_ML + "cnn_rnn", _EMB + "contrastive", _GEN + "multimodal"),
        (
            "Every modality becomes a sequence of vectors a transformer attends over.",
            "Images become patch tokens, (H/P)·(W/P) of them, so cost grows with the square of the resolution.",
            "A small projector connects a vision or audio encoder to a language model; it is trained first, with both models frozen.",
            "Audio becomes a log-mel spectrogram, then about 50 tokens a second.",
            "Video multiplies image tokens by time, so frames are sampled; text stays the densest input.",
        ),
    ),
]


def _lesson(module: str) -> "Lesson":
    return CURRICULUM[_BY_MODULE[module]]


def big_questions_table() -> str:
    """The README's compact map: each big question and its route. Regenerate with `make readme`."""
    rows = ["| Big question | Lessons that answer it, in order |", "|---|---|"]
    for q in BIG_QUESTIONS:
        route = ", ".join(f"[{_lesson(m).title}]({m.replace('.', '/')}.py)" for m in q.route)
        rows.append(f"| {q.question} | {route} |")
    return link_code_references("\n".join(rows) + "\n", '.')


def big_questions_page() -> str:
    """docs/BIG_QUESTIONS.md: every big question with its route and the short version."""
    out = [
        "# Big questions: the map from the top down\n",
        "The lessons build the field from the bottom up. Real conversations about AI systems start from the top, "
        "with questions like these. For each one: the lessons that answer it, in order, and the short version, "
        "the main ideas in the order that builds understanding. Read the short version first, then open the "
        "lessons wherever you want the full story.\n",
        "Generated from `primer/curriculum.py` by `make readme`; edit it there.\n",
    ]
    for n, q in enumerate(BIG_QUESTIONS, 1):
        route = " → ".join(f"[{_lesson(m).title}](../{m.replace('.', '/')}.py)" for m in q.route)
        in_brief = "\n".join(f"{i}. {point}" for i, point in enumerate(q.in_brief, 1))
        out.append(f"\n## {n}. {q.question}\n\n**Route:** {route}\n\n**In brief:**\n\n{in_brief}\n")
    return link_code_references("\n".join(out), 'docs')


def self_test_book() -> str:
    """Every lesson's self-test questions, in reading order, as one markdown page.

    Generated from the lesson docstrings (the single source of truth) into
    docs/SELF_TEST.md by `make readme`.
    """
    import importlib
    import re

    out = [
        "# Self-test: every question in the primer\n",
        "Generated from each lesson's `## Self-test questions` section by `make readme`; "
        "edit the lesson, not this file. Answer each question out loud before reading the answer.\n",
    ]
    for part in PARTS:
        out.append(f"\n## {part.title}\n")
        for i, lesson in lessons_in(part.key):
            try:
                doc = importlib.import_module(lesson.module).__doc__ or ""
            except ModuleNotFoundError:
                continue
            m = re.search(r"^## Self-test questions\s*\n(.*?)(?=^## |\Z)", doc, re.S | re.M)
            if not m:
                continue
            body = re.sub(r"^(#+) ", lambda h: "#" * (len(h.group(1)) + 2) + " ", m.group(1).strip(), flags=re.M)
            link = lesson.module.replace(".", "/") + ".py"
            out.append(f"\n### {i}. {lesson.title}\n\nFrom [`{lesson.module}`](../{link}).\n\n{body}\n")
    return link_code_references("\n".join(out), "docs")


def _render_doc() -> str:
    parts = []
    for part in PARTS:
        parts.append(f"\n## {part.title}\n\n{part.blurb}\n")
        parts += [f"{i}. `{l.module}`: **{l.title}.** {l.outcome}." for i, l in lessons_in(part.key)]
    return __doc__ + "\n".join(parts) + "\n"


__doc__ = _render_doc()


if __name__ == "__main__":
    # `make readme`: rewrite the generated section of README.md in place.
    import re
    import sys
    from pathlib import Path

    readme = Path(__file__).resolve().parent.parent / "README.md"
    text = readme.read_text()
    new, n = re.subn(
        r"(<!-- BEGIN curriculum -->\n).*?(<!-- END curriculum -->)",
        lambda m: m.group(1) + readme_section() + m.group(2),
        text,
        flags=re.S,
    )
    if n != 1:
        sys.exit("README.md needs exactly one <!-- BEGIN curriculum --> ... <!-- END curriculum --> block")
    new, n = re.subn(
        r"(<!-- BEGIN big-questions -->\n).*?(<!-- END big-questions -->)",
        lambda m: m.group(1) + big_questions_table() + m.group(2),
        new,
        flags=re.S,
    )
    if n != 1:
        sys.exit("README.md needs exactly one <!-- BEGIN big-questions --> ... <!-- END big-questions --> block")
    readme.write_text(new)
    (readme.parent / "docs" / "SELF_TEST.md").write_text(self_test_book())
    (readme.parent / "docs" / "BIG_QUESTIONS.md").write_text(big_questions_page())
    print("README.md, docs/SELF_TEST.md and docs/BIG_QUESTIONS.md regenerated from primer/curriculum.py")

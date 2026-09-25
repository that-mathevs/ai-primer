# primer: how modern AI works, built from scratch

Most explanations of AI either stay at the level of analogy or jump straight
to notation that assumes you already know it. This repository does neither.
Every idea is **built in plain Python you can read**, **drawn** as a
diagram, **decoded** symbol by symbol, and **checked** by a test that states
what it proves. Nothing depends on already knowing the jargon.

**Read it online: https://that-mathevs.github.io/ai-primer/** (every lesson,
diagram and annotated paper, with links from each explanation to the code
that implements it and the tests that specify it).

It covers two things:

1. **How the model works inside:** neurons and backpropagation, attention
   and transformers, tokenization, training and inference, and embeddings
   and vector search in depth.
2. **How to build systems people can rely on:** agent loops, tool calling,
   RAG, memory, planning, evals, guardrails, cost, observability and safe
   deployment.

The landmark papers behind each idea come with **annotated interactive
companions**: hover over any term or equation symbol for a plain-English
explanation.

## Start here

**1. Set up** (Python 3.10+; the lessons need only NumPy):

```bash
git clone https://github.com/that-mathevs/ai-primer && cd ai-primer
python -m pip install -e ".[dev,docs]"
make docs            # builds the site; open docs/html/index.html in a browser
```

**2. Pick how you read.** Each lesson works three ways, so use whichever suits you:

- **In the browser** (recommended): [the live site](https://that-mathevs.github.io/ai-primer/),
  or `docs/html/index.html` after `make docs`. It has every lesson in order,
  diagrams and figures, and a plain-English definition when you hover over any
  underlined term. Every page has previous/next links, and every **In code:**
  line links to the function it names, with its source one click away (View
  Source on the page, or its exact lines on GitHub).
- **In your editor**: each lesson is one file, such as `primer/ml/attention.py`.
  The explanation is the docstring at the top and the code follows it.
- **In the terminal**: `python -m primer.ml.attention` runs a lesson as a
  narrated walkthrough with real numbers.

**3. Open lesson 0, then lesson 1.** Lesson 0, `primer.notation`, turns every
symbol you'll meet (Σ, ‖x‖, ∂, ᵀ) into a few lines of Python; skip it if those
already read easily. Lesson 1, `primer.ml.big_picture`, follows one prompt all the
way through a model, and every later lesson zooms into one box of that picture.

**Routes, if you're short on time**

| If you want… | Read these, in order |
|---|---|
| The core ideas in one evening | `primer.ml.big_picture`, `primer.ml.attention`, `primer.ml.embeddings.similarity`, `primer.ml.embeddings.retrieval`, `primer.agents.rag` |
| To build with language models now | `primer.agents.llm`, `primer.agents.agent_loop`, `primer.agents.tools`, `primer.agents.rag`, `primer.agents.evals`, `primer.agents.guardrails` |
| To understand search and embeddings | `primer.ml.embeddings.word2vec`, then the rest of the embeddings part in order, then `primer.agents.rag` |
| To see how a model is trained and served | `primer.ml.neural_net`, `primer.ml.optimizers`, `primer.ml.training_stages`, `primer.ml.inference` |
| The whole thing | Every lesson in the order below, about one per sitting |

After each lesson, test yourself with its questions in
[`docs/SELF_TEST.md`](docs/SELF_TEST.md), and read the lesson's spec
(`make spec` prints every test as a sentence) to see exactly what the code
proves.

**Useful commands**

```bash
python -m primer.ml.attention   # run any lesson's walkthrough
make test                       # run every test
make spec                       # read the whole test suite as a specification
make docs                       # rebuild the site after changing a lesson
```

## Big questions: navigating by topic

The reading order below builds the field from the bottom up. When you need
the top-down view, start from a big question instead: each one lists the
lessons that answer it, in order. [`docs/BIG_QUESTIONS.md`](docs/BIG_QUESTIONS.md)
adds the spine of a complete answer to each question (the points to cover, in
order), which makes it a good way to practise explaining a topic out loud.

<!-- BEGIN big-questions -->
| Big question | Lessons that answer it, in order |
|---|---|
| What happens, step by step, when I send a prompt to a language model? | [Math notation, from zero](primer/notation.py), [The big picture](primer/ml/big_picture.py), [Tokenization](primer/ml/tokenization.py), [Attention](primer/ml/attention.py), [Positional information](primer/ml/positional.py), [The transformer](primer/ml/transformer.py), [Inference](primer/ml/inference.py) |
| How does a neural network actually learn? | [Neural networks](primer/ml/neural_net.py), [Loss functions](primer/ml/losses.py), [Optimizers](primer/ml/optimizers.py), [Training deep networks](primer/ml/deep_nets.py), [Overfitting and regularization](primer/ml/regularization.py) |
| How does attention work, and why did transformers replace RNNs? | [Attention](primer/ml/attention.py), [Positional information](primer/ml/positional.py), [The transformer](primer/ml/transformer.py), [CNNs and RNNs](primer/ml/cnn_rnn.py) |
| How are large language models trained, and when should I fine-tune instead of using RAG? | [Training stages](primer/ml/training_stages.py), [Tokenization](primer/ml/tokenization.py), [Loss functions](primer/ml/losses.py), [Embeddings in production](primer/ml/embeddings/operations.py), [Retrieval-augmented generation](primer/agents/rag.py) |
| What makes serving a model fast and affordable? | [Inference](primer/ml/inference.py), [Attention](primer/ml/attention.py), [Cost and latency](primer/agents/cost.py), [Context engineering](primer/agents/context.py) |
| What is an embedding, and how is an embedding model trained? | [Word embeddings](primer/ml/embeddings/word2vec.py), [Training embedding models](primer/ml/embeddings/contrastive.py), [Similarity](primer/ml/embeddings/similarity.py), [Loss functions](primer/ml/losses.py) |
| How do you search millions of vectors quickly, and what does it cost? | [Similarity](primer/ml/embeddings/similarity.py), [Dimensions and compression](primer/ml/embeddings/compression.py), [Vector indexes](primer/ml/embeddings/ann.py) |
| How do you build retrieval that returns the right passages? | [Retrieval](primer/ml/embeddings/retrieval.py), [Clustering and matching](primer/ml/embeddings/clustering.py), [Embeddings in production](primer/ml/embeddings/operations.py), [Metrics](primer/ml/metrics.py), [Retrieval-augmented generation](primer/agents/rag.py) |
| How do you know if a model or an agent is any good? | [Metrics](primer/ml/metrics.py), [Loss functions](primer/ml/losses.py), [Overfitting and regularization](primer/ml/regularization.py), [Evaluation](primer/agents/evals.py) |
| When should you build an agent, and how does one work? | [Talking to a model](primer/agents/llm.py), [Orchestration](primer/agents/orchestration.py), [The agent loop](primer/agents/agent_loop.py), [Tools](primer/agents/tools.py), [Model Context Protocol](primer/agents/mcp.py), [Planning](primer/agents/planning.py) |
| How do you give an AI system the right context and memory? | [Context engineering](primer/agents/context.py), [Memory](primer/agents/memory.py), [Retrieval-augmented generation](primer/agents/rag.py) |
| How do you make an AI system safe to put in front of real users? | [Guardrails](primer/agents/guardrails.py), [Tools](primer/agents/tools.py), [Safe deployment](primer/agents/deployment.py), [Observability](primer/agents/observability.py) |
| How do you cut cost and latency without hurting quality? | [Cost and latency](primer/agents/cost.py), [Context engineering](primer/agents/context.py), [Inference](primer/ml/inference.py), [Clustering and matching](primer/ml/embeddings/clustering.py) |
| Why do AI systems fail in production, and how do you fix them? | [Why the hard ones fail](primer/agents/failures.py), [Planning](primer/agents/planning.py), [Retrieval-augmented generation](primer/agents/rag.py), [Evaluation](primer/agents/evals.py), [Observability](primer/agents/observability.py) |
<!-- END big-questions -->

## How each lesson works

Every lesson is one Python module that you can read top to bottom, run, and
browse as a web page.

- **Start from zero.** Each concept starts with an everyday picture, then a
  tiny example with numbers you can check by hand, then a diagram, then the
  math and the code, then why it matters in practice.
- **Every formula decoded.** Under each equation: a table of what every
  symbol means, the formula read aloud as a sentence, and the formula worked
  through with real numbers.
- **Diagrams and figures everywhere,** each followed by a *Reading it*
  walkthrough.
- **From scratch.** The lessons use NumPy and the standard library only, so
  you can see every step. PyTorch and scikit-learn appear only in tests, to
  confirm the hand-written versions agree with them.
- **Tests are the specification.** Run `make spec` to read the whole suite
  as a list of plain-English statements about how things behave.
- **Offline and deterministic.** The agent lessons run against a scripted
  model, so you can reproduce every failure mode on purpose. Swap in the real
  Claude adapter (`primer.agents.llm.ClaudeLLM`) when you want to.

## Reading order

Generated from [`primer/curriculum.py`](primer/curriculum.py), so it is always current. Start with lesson 0 if Σ, ‖x‖ or ∂ look unfamiliar; every lesson links back to it the first time it uses a symbol.

<!-- BEGIN curriculum -->
### Before you begin

The notation every formula in this primer uses, decoded as short loops.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 0 | [Math notation, from zero](primer/notation.py) | Every symbol in an ML formula, as a short loop | [page](https://that-mathevs.github.io/ai-primer/primer/notation.html) · [tests](tests/test_notation.py) |

### Part 1: how the model works inside

From a single neuron to a working transformer, and how models are trained and served.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 1 | [The big picture](primer/ml/big_picture.py) | What happens, end to end, when you send a prompt | [page](https://that-mathevs.github.io/ai-primer/primer/ml/big_picture.html) · [tests](tests/test_big_picture.py) |
| 2 | [Neural networks](primer/ml/neural_net.py) | Neurons, activations, the forward pass, backprop by hand | [page](https://that-mathevs.github.io/ai-primer/primer/ml/neural_net.html) · [tests](tests/test_neural_net.py) |
| 3 | [Optimizers](primer/ml/optimizers.py) | SGD, momentum, Adam/AdamW, learning-rate warmup and decay | [page](https://that-mathevs.github.io/ai-primer/primer/ml/optimizers.html) · [tests](tests/test_optimizers.py) |
| 4 | [Training deep networks](primer/ml/deep_nets.py) | Vanishing/exploding gradients, residuals, normalization, initialization | [page](https://that-mathevs.github.io/ai-primer/primer/ml/deep_nets.html) · [tests](tests/test_deep_nets.py) |
| 5 | [Attention](primer/ml/attention.py) | Queries, keys, values, softmax, masking, multi-head, GQA, O(n²) | [page](https://that-mathevs.github.io/ai-primer/primer/ml/attention.html) · [tests](tests/test_attention.py) |
| 6 | [Positional information](primer/ml/positional.py) | Why order must be added, sinusoids and RoPE | [page](https://that-mathevs.github.io/ai-primer/primer/ml/positional.html) · [tests](tests/test_positional.py) |
| 7 | [The transformer](primer/ml/transformer.py) | The block, a tiny GPT, parameter counts, mixture of experts | [page](https://that-mathevs.github.io/ai-primer/primer/ml/transformer.html) · [tests](tests/test_transformer.py) |
| 8 | [Tokenization](primer/ml/tokenization.py) | BPE from scratch, byte-level tokens, why models miscount letters | [page](https://that-mathevs.github.io/ai-primer/primer/ml/tokenization.html) · [tests](tests/test_tokenization.py) |
| 9 | [Training stages](primer/ml/training_stages.py) | Pretraining, SFT, RLHF and DPO, LoRA, fine-tuning vs. RAG | [page](https://that-mathevs.github.io/ai-primer/primer/ml/training_stages.html) · [tests](tests/test_training_stages.py) |
| 10 | [Inference](primer/ml/inference.py) | Prefill vs. decode, the KV cache, sampling, speculative decoding, memory math | [page](https://that-mathevs.github.io/ai-primer/primer/ml/inference.html) · [tests](tests/test_inference.py) |
| 11 | [Loss functions](primer/ml/losses.py) | Cross-entropy, perplexity, MSE/MAE, contrastive losses | [page](https://that-mathevs.github.io/ai-primer/primer/ml/losses.html) · [tests](tests/test_losses.py) |
| 12 | [Metrics](primer/ml/metrics.py) | Precision/recall/F1, ROC-AUC, recall@k, MRR, nDCG, BLEU/ROUGE | [page](https://that-mathevs.github.io/ai-primer/primer/ml/metrics.html) · [tests](tests/test_metrics.py) |
| 13 | [Overfitting and regularization](primer/ml/regularization.py) | Overfitting, early stopping, dropout, L1/L2, leakage | [page](https://that-mathevs.github.io/ai-primer/primer/ml/regularization.html) · [tests](tests/test_regularization.py) |
| 14 | [CNNs and RNNs](primer/ml/cnn_rnn.py) | How convolutions see and recurrent nets remember, and why transformers won | [page](https://that-mathevs.github.io/ai-primer/primer/ml/cnn_rnn.html) · [tests](tests/test_cnn_rnn.py) |

### Embeddings, the centerpiece

Vectors that capture meaning, and the search systems built on them.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 15 | [Word embeddings](primer/ml/embeddings/word2vec.py) | Where embeddings came from, analogies, the "bank" problem | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/word2vec.html) · [tests](tests/test_emb_word2vec.py) |
| 16 | [Similarity](primer/ml/embeddings/similarity.py) | Cosine vs. dot vs. distance, normalization, anisotropy, thresholds | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/similarity.html) · [tests](tests/test_emb_similarity.py) |
| 17 | [Training embedding models](primer/ml/embeddings/contrastive.py) | Contrastive learning, hard negatives, CLIP | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/contrastive.html) · [tests](tests/test_emb_contrastive.py) |
| 18 | [Dimensions and compression](primer/ml/embeddings/compression.py) | Storage math, Matryoshka truncation, int8 and binary quantization | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/compression.html) · [tests](tests/test_emb_compression.py) |
| 19 | [Vector indexes](primer/ml/embeddings/ann.py) | Flat, IVF, PQ and HNSW from scratch, recall vs. latency | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/ann.html) · [tests](tests/test_emb_ann.py) |
| 20 | [Retrieval](primer/ml/embeddings/retrieval.py) | BM25, hybrid search with RRF, rerankers, ColBERT, chunking | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/retrieval.html) · [tests](tests/test_emb_retrieval.py) |
| 21 | [Clustering and matching](primer/ml/embeddings/clustering.py) | k-means, density clustering, dedup, routing, semantic caching | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/clustering.html) · [tests](tests/test_emb_clustering.py) |
| 22 | [Embeddings in production](primer/ml/embeddings/operations.py) | Model migrations, domain mismatch, measuring retrieval on its own | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/operations.html) · [tests](tests/test_emb_operations.py) |

### Part 2: building systems people rely on

Agents, tools, retrieval, memory, evaluation, safety, cost and deployment.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 23 | [Talking to a model](primer/agents/llm.py) | The message format, and what tool calling really is | [page](https://that-mathevs.github.io/ai-primer/primer/agents/llm.html) · [tests](tests/test_agents_llm.py) |
| 24 | [Orchestration](primer/agents/orchestration.py) | Workflows vs. agents, and the named patterns | [page](https://that-mathevs.github.io/ai-primer/primer/agents/orchestration.html) · [tests](tests/test_agents_orchestration.py) |
| 25 | [The agent loop](primer/agents/agent_loop.py) | A production agent loop: budgets, loop detection, recovery | [page](https://that-mathevs.github.io/ai-primer/primer/agents/agent_loop.html) · [tests](tests/test_agents_agent_loop.py) |
| 26 | [Tools](primer/agents/tools.py) | Tool design, validation, idempotency, approvals, least privilege | [page](https://that-mathevs.github.io/ai-primer/primer/agents/tools.html) · [tests](tests/test_agents_tools.py) |
| 27 | [Model Context Protocol](primer/agents/mcp.py) | MCP on the wire, and its security risks | [page](https://that-mathevs.github.io/ai-primer/primer/agents/mcp.html) · [tests](tests/test_agents_mcp.py) |
| 28 | [Retrieval-augmented generation](primer/agents/rag.py) | RAG end to end, with citations and access control | [page](https://that-mathevs.github.io/ai-primer/primer/agents/rag.html) · [tests](tests/test_agents_rag.py) |
| 29 | [Context engineering](primer/agents/context.py) | What goes in the window, compression, cache-friendly layout | [page](https://that-mathevs.github.io/ai-primer/primer/agents/context.html) · [tests](tests/test_agents_context.py) |
| 30 | [Memory](primer/agents/memory.py) | Short- and long-term memory, tenant isolation, forgetting | [page](https://that-mathevs.github.io/ai-primer/primer/agents/memory.html) · [tests](tests/test_agents_memory.py) |
| 31 | [Planning](primer/agents/planning.py) | Plan-and-execute, decomposition, reflection, compounding error | [page](https://that-mathevs.github.io/ai-primer/primer/agents/planning.html) · [tests](tests/test_agents_planning.py) |
| 32 | [Evaluation](primer/agents/evals.py) | Golden sets, graders, LLM-as-judge calibration | [page](https://that-mathevs.github.io/ai-primer/primer/agents/evals.html) · [tests](tests/test_agents_evals.py) |
| 33 | [Guardrails](primer/agents/guardrails.py) | Prompt injection and privilege separation, PII, output checks | [page](https://that-mathevs.github.io/ai-primer/primer/agents/guardrails.html) · [tests](tests/test_agents_guardrails.py) |
| 34 | [Cost and latency](primer/agents/cost.py) | Routing, caching, batching, budgets, cost per successful task | [page](https://that-mathevs.github.io/ai-primer/primer/agents/cost.html) · [tests](tests/test_agents_cost.py) |
| 35 | [Observability](primer/agents/observability.py) | Traces, OpenTelemetry GenAI attributes, the improvement loop | [page](https://that-mathevs.github.io/ai-primer/primer/agents/observability.html) · [tests](tests/test_agents_observability.py) |
| 36 | [Safe deployment](primer/agents/deployment.py) | Shadow mode, graduated autonomy, canaries, kill switches, audit logs | [page](https://that-mathevs.github.io/ai-primer/primer/agents/deployment.html) · [tests](tests/test_agents_deployment.py) |
| 37 | [Why the hard ones fail](primer/agents/failures.py) | The common failure modes, and the fix for each | [page](https://that-mathevs.github.io/ai-primer/primer/agents/failures.html) · [tests](tests/test_agents_failures.py) |

<!-- END curriculum -->

`primer.glossary` defines every term in one place; the HTML site shows those
definitions on hover. [`docs/SELF_TEST.md`](docs/SELF_TEST.md) collects every
lesson's self-test questions and answers, in reading order.

## The papers

`docs/papers/` holds annotated, interactive companions to the papers behind
the lessons, starting with *Attention Is All You Need*. The full list is in
[`docs/papers/CATALOG.md`](docs/papers/CATALOG.md). Companions quote briefly,
redraw figures, and link to the original paper for the full text.

## Contributing

The project's rules (voice, lesson structure, diagrams, test style) are in
[`CLAUDE.md`](CLAUDE.md). They apply to human and AI contributors alike.

## License

MIT. See [`LICENSE`](LICENSE).

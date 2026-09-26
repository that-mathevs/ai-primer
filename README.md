# AI Primer

This is an effort to explain how modern AI works, from the ground up, in a way that is 
accessible to anyone with a basic understanding of programming and math. It is not a course, 
but a reference that you can read in any order, with each lesson building on the previous ones. 
I made this because people kept asking me how modern AI works, and I wanted to have a single place to 
point them to. Despite what most "AI people" say, you don't need to be a PhD in math or computer science 
to understand how AI works.

## 👉 Read it here: [that-mathevs.github.io/ai-primer](https://that-mathevs.github.io/ai-primer/)

**The website is the best place to read this.** Nothing to install: every
lesson in order, with its diagrams and figures drawn, the math typeset, a
plain-English definition when you hover over any term, and a link from each
explanation to the code that implements it and the tests that specify it.
The annotated paper companions live there too. Light and dark themes.

This repository is the source. Clone it to run the lessons yourself (see
[Start here](#start-here)).

Most explanations of AI either stay at the level of analogy or jump straight
to notation that assumes you already know it. This repository does neither.
Every idea is **built in plain Python you can read**, **drawn** as a
diagram, and **decoded** symbol by symbol. Each lesson also comes with
**tests**: small programs that run its code and check it does what the lesson
claims, each named as a plain sentence, such as *given a causal mask, future
tokens receive zero attention* ([the attention lesson's tests](tests/test_attention.py)).
Nothing depends on already knowing the jargon.

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
- **In your editor**: each lesson is one file, such as [`primer/ml/attention.py`](primer/ml/attention.py).
  The explanation is the docstring at the top and the code follows it.
- **In the terminal**: [`python -m primer.ml.attention`](primer/ml/attention.py) runs a lesson as a
  narrated walkthrough with real numbers.

**3. Open lesson 0, then lesson 1.** Lesson 0, [`primer.notation`](primer/notation.py), turns every
symbol you'll meet (Σ, ‖x‖, ∂, ᵀ) into a few lines of Python; skip it if those
already read easily. Lesson 1, [`primer.ml.big_picture`](primer/ml/big_picture.py), follows one prompt all the
way through a model, and every later lesson zooms into one box of that picture.

**Routes, if you're short on time**

| If you want… | Read these, in order |
|---|---|
| The core ideas in one evening | [`primer.ml.big_picture`](primer/ml/big_picture.py), [`primer.ml.attention`](primer/ml/attention.py), [`primer.ml.embeddings.similarity`](primer/ml/embeddings/similarity.py), [`primer.ml.embeddings.retrieval`](primer/ml/embeddings/retrieval.py), [`primer.agents.rag`](primer/agents/rag.py) |
| To build with language models now | [`primer.agents.llm`](primer/agents/llm.py), [`primer.agents.agent_loop`](primer/agents/agent_loop.py), [`primer.agents.tools`](primer/agents/tools.py), [`primer.agents.rag`](primer/agents/rag.py), [`primer.agents.evals`](primer/agents/evals.py), [`primer.agents.guardrails`](primer/agents/guardrails.py) |
| To understand search and embeddings | [`primer.ml.embeddings.word2vec`](primer/ml/embeddings/word2vec.py), then the rest of the embeddings part in order, then [`primer.agents.rag`](primer/agents/rag.py) |
| To see how a model is trained and served | [`primer.ml.neural_net`](primer/ml/neural_net.py), [`primer.ml.optimizers`](primer/ml/optimizers.py), [`primer.ml.training_stages`](primer/ml/training_stages.py), [`primer.ml.inference`](primer/ml/inference.py) |
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
adds the short version of each: the main ideas, in the order that builds
understanding.

<!-- BEGIN big-questions -->
| Big question | Lessons that answer it, in order |
|---|---|
| What happens, step by step, when I send a prompt to a language model? | [Math notation, from zero](primer/notation.py), [The big picture](primer/ml/big_picture.py), [Tokenization](primer/ml/tokenization.py), [Attention](primer/ml/attention.py), [Positional information](primer/ml/positional.py), [The transformer](primer/ml/transformer.py), [Inference](primer/ml/inference.py) |
| How does a neural network actually learn? | [Neural networks](primer/ml/neural_net.py), [Loss functions](primer/ml/losses.py), [Optimizers](primer/ml/optimizers.py), [Training deep networks](primer/ml/deep_nets.py), [Overfitting and regularization](primer/ml/regularization.py) |
| How does attention work, and why did transformers replace RNNs? | [Attention](primer/ml/attention.py), [Positional information](primer/ml/positional.py), [The transformer](primer/ml/transformer.py), [CNNs and RNNs](primer/ml/cnn_rnn.py) |
| How are large language models trained, and when should I fine-tune instead of using RAG? | [Training stages](primer/ml/training_stages.py), [Fine-tuning in practice](primer/ml/fine_tuning.py), [Tokenization](primer/ml/tokenization.py), [Loss functions](primer/ml/losses.py), [Embeddings in production](primer/ml/embeddings/operations.py), [Retrieval-augmented generation](primer/agents/rag.py) |
| What makes serving a model fast and affordable? | [The hardware underneath](primer/ml/hardware.py), [Inference](primer/ml/inference.py), [Long context and efficient architectures](primer/ml/efficient_architectures.py), [Attention](primer/ml/attention.py), [Cost and latency](primer/agents/cost.py), [Context engineering](primer/agents/context.py) |
| What is an embedding, and how is an embedding model trained? | [Word embeddings](primer/ml/embeddings/word2vec.py), [Training embedding models](primer/ml/embeddings/contrastive.py), [Similarity](primer/ml/embeddings/similarity.py), [Loss functions](primer/ml/losses.py) |
| How do you search millions of vectors quickly, and what does it cost? | [Similarity](primer/ml/embeddings/similarity.py), [Dimensions and compression](primer/ml/embeddings/compression.py), [Vector indexes](primer/ml/embeddings/ann.py) |
| How do you build retrieval that returns the right passages? | [Retrieval](primer/ml/embeddings/retrieval.py), [Clustering and matching](primer/ml/embeddings/clustering.py), [Embeddings in production](primer/ml/embeddings/operations.py), [Metrics](primer/ml/metrics.py), [Retrieval-augmented generation](primer/agents/rag.py) |
| How do you know if a model or an agent is any good? | [Metrics](primer/ml/metrics.py), [Reading benchmarks](primer/ml/benchmarks.py), [Loss functions](primer/ml/losses.py), [Overfitting and regularization](primer/ml/regularization.py), [Evaluation](primer/agents/evals.py) |
| When should you build an agent, and how does one work? | [Talking to a model](primer/agents/llm.py), [Orchestration](primer/agents/orchestration.py), [The agent loop](primer/agents/agent_loop.py), [Tools](primer/agents/tools.py), [Coding and computer-use agents](primer/agents/coding_agents.py), [Model Context Protocol](primer/agents/mcp.py), [Planning](primer/agents/planning.py) |
| How do you give an AI system the right context and memory? | [Context engineering](primer/agents/context.py), [Memory](primer/agents/memory.py), [Retrieval-augmented generation](primer/agents/rag.py) |
| How do you make an AI system safe to put in front of real users? | [Alignment and safety](primer/ml/alignment.py), [Guardrails](primer/agents/guardrails.py), [Tools](primer/agents/tools.py), [Safe deployment](primer/agents/deployment.py), [Observability](primer/agents/observability.py) |
| How do you cut cost and latency without hurting quality? | [Cost and latency](primer/agents/cost.py), [Context engineering](primer/agents/context.py), [Inference](primer/ml/inference.py), [Clustering and matching](primer/ml/embeddings/clustering.py) |
| Why do AI systems fail in production, and how do you fix them? | [Why the hard ones fail](primer/agents/failures.py), [Structured output](primer/ml/structured_output.py), [Planning](primer/agents/planning.py), [Retrieval-augmented generation](primer/agents/rag.py), [Evaluation](primer/agents/evals.py), [Observability](primer/agents/observability.py) |
| What does it take to pretrain a large model? | [Training stages](primer/ml/training_stages.py), [Tokenization](primer/ml/tokenization.py), [Pretraining at scale](primer/ml/pretraining.py), [Optimizers](primer/ml/optimizers.py), [The hardware underneath](primer/ml/hardware.py) |
| How does a model learn from rewards instead of examples? | [Reinforcement learning](primer/ml/reinforcement.py), [Training stages](primer/ml/training_stages.py), [Reasoning models](primer/ml/reasoning.py), [Alignment and safety](primer/ml/alignment.py) |
| How do reasoning models think, and when is extra thinking worth it? | [Inference](primer/ml/inference.py), [Reinforcement learning](primer/ml/reinforcement.py), [Reasoning models](primer/ml/reasoning.py), [Planning](primer/agents/planning.py), [Cost and latency](primer/agents/cost.py) |
| How do models handle very long contexts? | [Attention](primer/ml/attention.py), [Positional information](primer/ml/positional.py), [Inference](primer/ml/inference.py), [Long context and efficient architectures](primer/ml/efficient_architectures.py) |
| What is going on inside a trained model, and how can we tell? | [The transformer](primer/ml/transformer.py), [Word embeddings](primer/ml/embeddings/word2vec.py), [Looking inside the model](primer/ml/interpretability.py), [Alignment and safety](primer/ml/alignment.py) |
| When is a neural network the wrong tool? | [Trees and boosting](primer/ml/classical.py), [Neural networks](primer/ml/neural_net.py), [Overfitting and regularization](primer/ml/regularization.py), [Metrics](primer/ml/metrics.py) |
| How do AI models generate images, audio and video? | [Autoencoders and VAEs](primer/ml/generative/autoencoders.py), [GANs](primer/ml/generative/gans.py), [Diffusion and flow matching](primer/ml/generative/diffusion.py), [Multimodal models](primer/ml/generative/multimodal.py) |
| How do AI models see images and hear audio? | [CNNs and RNNs](primer/ml/cnn_rnn.py), [Training embedding models](primer/ml/embeddings/contrastive.py), [Multimodal models](primer/ml/generative/multimodal.py) |
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
  Claude adapter ([`primer.agents.llm.ClaudeLLM`](primer/agents/llm.py)) when you want to.

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
| 10 | [Pretraining at scale](primer/ml/pretraining.py) | Data curation and deduplication, parallelism across GPUs, mixed precision | [page](https://that-mathevs.github.io/ai-primer/primer/ml/pretraining.html) · [tests](tests/test_pretraining.py) |
| 11 | [Fine-tuning in practice](primer/ml/fine_tuning.py) | Preparing data, forgetting old skills, merging models | [page](https://that-mathevs.github.io/ai-primer/primer/ml/fine_tuning.html) · [tests](tests/test_fine_tuning.py) |
| 12 | [Reinforcement learning](primer/ml/reinforcement.py) | Policy gradients from scratch, PPO, GRPO, reward hacking | [page](https://that-mathevs.github.io/ai-primer/primer/ml/reinforcement.html) · [tests](tests/test_reinforcement.py) |
| 13 | [Reasoning models](primer/ml/reasoning.py) | Chain of thought, test-time compute, verifiers, learning to reason with RL | [page](https://that-mathevs.github.io/ai-primer/primer/ml/reasoning.html) · [tests](tests/test_reasoning.py) |
| 14 | [Alignment and safety](primer/ml/alignment.py) | Constitutional AI, red-teaming, sycophancy, refusals | [page](https://that-mathevs.github.io/ai-primer/primer/ml/alignment.html) · [tests](tests/test_alignment.py) |
| 15 | [The hardware underneath](primer/ml/hardware.py) | GPUs, the memory hierarchy, FLOPs vs. bandwidth, number formats | [page](https://that-mathevs.github.io/ai-primer/primer/ml/hardware.html) · [tests](tests/test_hardware.py) |
| 16 | [Inference](primer/ml/inference.py) | Prefill vs. decode, the KV cache, sampling, speculative decoding, memory math | [page](https://that-mathevs.github.io/ai-primer/primer/ml/inference.html) · [tests](tests/test_inference.py) |
| 17 | [Structured output](primer/ml/structured_output.py) | Constrained decoding: grammars and JSON schemas that guarantee valid output | [page](https://that-mathevs.github.io/ai-primer/primer/ml/structured_output.html) · [tests](tests/test_structured_output.py) |
| 18 | [Long context and efficient architectures](primer/ml/efficient_architectures.py) | Sliding-window and sparse attention, state-space models, KV-cache compression | [page](https://that-mathevs.github.io/ai-primer/primer/ml/efficient_architectures.html) · [tests](tests/test_efficient_architectures.py) |
| 19 | [Loss functions](primer/ml/losses.py) | Cross-entropy, perplexity, MSE/MAE, contrastive losses | [page](https://that-mathevs.github.io/ai-primer/primer/ml/losses.html) · [tests](tests/test_losses.py) |
| 20 | [Metrics](primer/ml/metrics.py) | Precision/recall/F1, ROC-AUC, recall@k, MRR, nDCG, BLEU/ROUGE | [page](https://that-mathevs.github.io/ai-primer/primer/ml/metrics.html) · [tests](tests/test_metrics.py) |
| 21 | [Reading benchmarks](primer/ml/benchmarks.py) | What benchmarks measure, contamination, leaderboards and arenas | [page](https://that-mathevs.github.io/ai-primer/primer/ml/benchmarks.html) · [tests](tests/test_benchmarks.py) |
| 22 | [Overfitting and regularization](primer/ml/regularization.py) | Overfitting, early stopping, dropout, L1/L2, leakage | [page](https://that-mathevs.github.io/ai-primer/primer/ml/regularization.html) · [tests](tests/test_regularization.py) |
| 23 | [Trees and boosting](primer/ml/classical.py) | Decision trees, random forests, gradient boosting, and when they still win | [page](https://that-mathevs.github.io/ai-primer/primer/ml/classical.html) · [tests](tests/test_classical.py) |
| 24 | [CNNs and RNNs](primer/ml/cnn_rnn.py) | How convolutions see and recurrent nets remember, and why transformers won | [page](https://that-mathevs.github.io/ai-primer/primer/ml/cnn_rnn.html) · [tests](tests/test_cnn_rnn.py) |
| 25 | [Looking inside the model](primer/ml/interpretability.py) | Probes, the logit lens, activation patching, superposition, sparse autoencoders | [page](https://that-mathevs.github.io/ai-primer/primer/ml/interpretability.html) · [tests](tests/test_interpretability.py) |

### Embeddings, the centerpiece

Vectors that capture meaning, and the search systems built on them.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 26 | [Word embeddings](primer/ml/embeddings/word2vec.py) | Where embeddings came from, analogies, the "bank" problem | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/word2vec.html) · [tests](tests/test_emb_word2vec.py) |
| 27 | [Similarity](primer/ml/embeddings/similarity.py) | Cosine vs. dot vs. distance, normalization, anisotropy, thresholds | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/similarity.html) · [tests](tests/test_emb_similarity.py) |
| 28 | [Training embedding models](primer/ml/embeddings/contrastive.py) | Contrastive learning, hard negatives, CLIP | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/contrastive.html) · [tests](tests/test_emb_contrastive.py) |
| 29 | [Dimensions and compression](primer/ml/embeddings/compression.py) | Storage math, Matryoshka truncation, int8 and binary quantization | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/compression.html) · [tests](tests/test_emb_compression.py) |
| 30 | [Vector indexes](primer/ml/embeddings/ann.py) | Flat, IVF, PQ and HNSW from scratch, recall vs. latency | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/ann.html) · [tests](tests/test_emb_ann.py) |
| 31 | [Retrieval](primer/ml/embeddings/retrieval.py) | BM25, hybrid search with RRF, rerankers, ColBERT, chunking | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/retrieval.html) · [tests](tests/test_emb_retrieval.py) |
| 32 | [Clustering and matching](primer/ml/embeddings/clustering.py) | k-means, density clustering, dedup, routing, semantic caching | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/clustering.html) · [tests](tests/test_emb_clustering.py) |
| 33 | [Embeddings in production](primer/ml/embeddings/operations.py) | Model migrations, domain mismatch, measuring retrieval on its own | [page](https://that-mathevs.github.io/ai-primer/primer/ml/embeddings/operations.html) · [tests](tests/test_emb_operations.py) |

### Generating images, audio and video

Autoencoders, GANs, diffusion, and the multimodal models that connect them to language.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 34 | [Autoencoders and VAEs](primer/ml/generative/autoencoders.py) | Squeezing data into a code and back, and sampling new data from it | [page](https://that-mathevs.github.io/ai-primer/primer/ml/generative/autoencoders.html) · [tests](tests/test_autoencoders.py) |
| 35 | [GANs](primer/ml/generative/gans.py) | A forger against a detective: adversarial training, and why it is unstable | [page](https://that-mathevs.github.io/ai-primer/primer/ml/generative/gans.html) · [tests](tests/test_gans.py) |
| 36 | [Diffusion and flow matching](primer/ml/generative/diffusion.py) | Turning noise into images one small step at a time | [page](https://that-mathevs.github.io/ai-primer/primer/ml/generative/diffusion.html) · [tests](tests/test_diffusion.py) |
| 37 | [Multimodal models](primer/ml/generative/multimodal.py) | Images, audio and video into a language model | [page](https://that-mathevs.github.io/ai-primer/primer/ml/generative/multimodal.html) · [tests](tests/test_multimodal.py) |

### Part 2: building systems people rely on

Agents, tools, retrieval, memory, evaluation, safety, cost and deployment.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 38 | [Talking to a model](primer/agents/llm.py) | The message format, and what tool calling really is | [page](https://that-mathevs.github.io/ai-primer/primer/agents/llm.html) · [tests](tests/test_agents_llm.py) |
| 39 | [Orchestration](primer/agents/orchestration.py) | Workflows vs. agents, and the named patterns | [page](https://that-mathevs.github.io/ai-primer/primer/agents/orchestration.html) · [tests](tests/test_agents_orchestration.py) |
| 40 | [The agent loop](primer/agents/agent_loop.py) | A production agent loop: budgets, loop detection, recovery | [page](https://that-mathevs.github.io/ai-primer/primer/agents/agent_loop.html) · [tests](tests/test_agents_agent_loop.py) |
| 41 | [Tools](primer/agents/tools.py) | Tool design, validation, idempotency, approvals, least privilege | [page](https://that-mathevs.github.io/ai-primer/primer/agents/tools.html) · [tests](tests/test_agents_tools.py) |
| 42 | [Coding and computer-use agents](primer/agents/coding_agents.py) | Edit, run, test, repeat; sandboxes; driving a screen | [page](https://that-mathevs.github.io/ai-primer/primer/agents/coding_agents.html) · [tests](tests/test_agents_coding_agents.py) |
| 43 | [Model Context Protocol](primer/agents/mcp.py) | MCP on the wire, and its security risks | [page](https://that-mathevs.github.io/ai-primer/primer/agents/mcp.html) · [tests](tests/test_agents_mcp.py) |
| 44 | [Retrieval-augmented generation](primer/agents/rag.py) | RAG end to end, with citations and access control | [page](https://that-mathevs.github.io/ai-primer/primer/agents/rag.html) · [tests](tests/test_agents_rag.py) |
| 45 | [Context engineering](primer/agents/context.py) | What goes in the window, compression, cache-friendly layout | [page](https://that-mathevs.github.io/ai-primer/primer/agents/context.html) · [tests](tests/test_agents_context.py) |
| 46 | [Memory](primer/agents/memory.py) | Short- and long-term memory, tenant isolation, forgetting | [page](https://that-mathevs.github.io/ai-primer/primer/agents/memory.html) · [tests](tests/test_agents_memory.py) |
| 47 | [Planning](primer/agents/planning.py) | Plan-and-execute, decomposition, reflection, compounding error | [page](https://that-mathevs.github.io/ai-primer/primer/agents/planning.html) · [tests](tests/test_agents_planning.py) |
| 48 | [Evaluation](primer/agents/evals.py) | Golden sets, graders, LLM-as-judge calibration | [page](https://that-mathevs.github.io/ai-primer/primer/agents/evals.html) · [tests](tests/test_agents_evals.py) |
| 49 | [Guardrails](primer/agents/guardrails.py) | Prompt injection and privilege separation, PII, output checks | [page](https://that-mathevs.github.io/ai-primer/primer/agents/guardrails.html) · [tests](tests/test_agents_guardrails.py) |
| 50 | [Cost and latency](primer/agents/cost.py) | Routing, caching, batching, budgets, cost per successful task | [page](https://that-mathevs.github.io/ai-primer/primer/agents/cost.html) · [tests](tests/test_agents_cost.py) |
| 51 | [Observability](primer/agents/observability.py) | Traces, OpenTelemetry GenAI attributes, the improvement loop | [page](https://that-mathevs.github.io/ai-primer/primer/agents/observability.html) · [tests](tests/test_agents_observability.py) |
| 52 | [Safe deployment](primer/agents/deployment.py) | Shadow mode, graduated autonomy, canaries, kill switches, audit logs | [page](https://that-mathevs.github.io/ai-primer/primer/agents/deployment.html) · [tests](tests/test_agents_deployment.py) |
| 53 | [Why the hard ones fail](primer/agents/failures.py) | The common failure modes, and the fix for each | [page](https://that-mathevs.github.io/ai-primer/primer/agents/failures.html) · [tests](tests/test_agents_failures.py) |

<!-- END curriculum -->

[`primer.glossary`](primer/glossary.py) defines every term in one place; the HTML site shows those
definitions on hover. [`docs/SELF_TEST.md`](docs/SELF_TEST.md) collects every
lesson's self-test questions and answers, in reading order.

## The papers

[`docs/papers/`](docs/papers) holds annotated, interactive companions to the papers behind
the lessons, starting with *Attention Is All You Need*. The full list is in
[`docs/papers/CATALOG.md`](docs/papers/CATALOG.md). Companions quote briefly,
redraw figures, and link to the original paper for the full text.

## Contributing

The project's rules (voice, lesson structure, diagrams, test style) are in
[`CLAUDE.md`](CLAUDE.md). They apply to human and AI contributors alike.

## License

Free for any noncommercial use: read it, learn from it, share it, teach
with it, adapt it. Selling it or using it commercially needs permission.
The code is under the [PolyForm Noncommercial License 1.0.0](LICENSE) and
the teaching content under [CC BY-NC-SA 4.0](LICENSE-CONTENT).

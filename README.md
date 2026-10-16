# AI Primer

**Modern AI from first principles: every concept explained, implemented and tested.**

This is an effort to explain how modern AI works, in a way that is accessible to anyone
with a basic understanding of programming and math. It is not a course, but a reference
that you can read in any order, with each lesson building on the previous ones.
You don't need a PhD in math or computer science to understand how AI works.

Every lesson has two levels, and you choose how deep to go. **Level 1, the
practitioner's guide**, is the level of a good professional book: what the thing is,
when you need it, your options and their trade-offs, what it costs, what breaks, and who
does it in the wild. Most readers stop there, well equipped. **Level 2, how it works from
scratch**, builds the same thing in plain Python, drawn and explained, with every formula
decoded symbol by symbol (that part folds away as Level 3 on the site). Start at the top;
descend as far as you like.

## 👉 Read it here: [rumblr.com](https://rumblr.com/)

**rumblr.com is the best place to read this**: the illustrated edition of every lesson, built from this repository. Nothing to install: every
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

## Where to start

That's a lot of lessons, so pick the path that fits you. Numbers are lesson
numbers; each path skips lessons but never jumps backwards.

<!-- BEGIN paths -->
| If you are… | You'll learn | Lessons, in order |
|---|---|---|
| **Software engineer new to AI** | How a language model works, then how to build on one. | [1](primer/ml/big_picture.py) → [2](primer/ml/neural_net.py) → [6](primer/ml/attention.py) → [8](primer/ml/transformer.py) → [9](primer/ml/tokenization.py) → [20](primer/ml/inference.py) → [25](primer/ml/embeddings/similarity.py) → [29](primer/ml/embeddings/retrieval.py) → [38](primer/agents/llm.py) → [40](primer/agents/agent_loop.py) → [44](primer/agents/rag.py) → [48](primer/agents/evals.py) |
| **AI application engineer** | Agents, retrieval and tools, and keeping them reliable, safe and affordable. | [21](primer/ml/structured_output.py) → [29](primer/ml/embeddings/retrieval.py) → [38](primer/agents/llm.py) → [39](primer/agents/orchestration.py) → [40](primer/agents/agent_loop.py) → [41](primer/agents/tools.py) → [42](primer/agents/coding_agents.py) → [43](primer/agents/mcp.py) → [44](primer/agents/rag.py) → [45](primer/agents/context.py) → [46](primer/agents/memory.py) → [48](primer/agents/evals.py) → [49](primer/agents/guardrails.py) → [50](primer/agents/cost.py) → [51](primer/agents/observability.py) → [52](primer/agents/deployment.py) → [53](primer/agents/failures.py) |
| **ML engineer** | The model itself: training, scaling, serving and looking inside. | [2](primer/ml/neural_net.py) → [3](primer/ml/losses.py) → [4](primer/ml/optimizers.py) → [5](primer/ml/deep_nets.py) → [6](primer/ml/attention.py) → [7](primer/ml/positional.py) → [8](primer/ml/transformer.py) → [10](primer/ml/training_stages.py) → [11](primer/ml/pretraining.py) → [12](primer/ml/fine_tuning.py) → [13](primer/ml/regularization.py) → [14](primer/ml/reinforcement.py) → [17](primer/ml/metrics.py) → [18](primer/ml/benchmarks.py) → [19](primer/ml/hardware.py) → [20](primer/ml/inference.py) → [22](primer/ml/efficient_architectures.py) → [23](primer/ml/interpretability.py) |
| **Engineering manager or architect** | What these systems can do, what they cost, and how they fail. | [1](primer/ml/big_picture.py) → [10](primer/ml/training_stages.py) → [15](primer/ml/reasoning.py) → [16](primer/ml/alignment.py) → [18](primer/ml/benchmarks.py) → [20](primer/ml/inference.py) → [39](primer/agents/orchestration.py) → [44](primer/agents/rag.py) → [48](primer/agents/evals.py) → [50](primer/agents/cost.py) → [52](primer/agents/deployment.py) → [53](primer/agents/failures.py) |
| **Just explain LLMs to me** | The shortest route to understanding what happens when you send a prompt. | [1](primer/ml/big_picture.py) → [6](primer/ml/attention.py) → [8](primer/ml/transformer.py) → [9](primer/ml/tokenization.py) → [10](primer/ml/training_stages.py) → [15](primer/ml/reasoning.py) → [20](primer/ml/inference.py) |
| **Curious about images, audio and video** | How models generate pictures and sound, and how they see and hear. | [2](primer/ml/neural_net.py) → [26](primer/ml/embeddings/contrastive.py) → [33](primer/ml/cnn_rnn.py) → [34](primer/ml/generative/autoencoders.py) → [35](primer/ml/generative/gans.py) → [36](primer/ml/generative/diffusion.py) → [37](primer/ml/generative/multimodal.py) |
<!-- END paths -->

## Start here

**1. Set up** (Python 3.10+; the lessons need only NumPy):

```bash
git clone https://github.com/that-mathevs/ai-primer && cd ai-primer
python -m pip install -e ".[dev,docs]"
make docs            # a plain local preview (docs/html); the published edition is rumblr.com
```

**2. Pick how you read.** Each lesson works three ways, so use whichever suits you:

- **In the browser** (recommended): [rumblr.com](https://rumblr.com/), built from this
  repository, or `docs/html/index.html` after `make docs` for a plain local preview. It has every lesson in order,
  diagrams and figures, and a plain-English definition when you hover over any
  underlined term. Every page has previous/next links, and every **In code:**
  line links to the function it names, with its source one click away (View
  Source on the page, or its exact lines on GitHub).
- **In your editor**: each lesson is one file, such as [`primer/ml/attention.py`](primer/ml/attention.py).
  The explanation is the docstring at the top and the code follows it.
- **In the terminal**: [`python -m primer.ml.attention`](primer/ml/attention.py) runs a lesson as a
  narrated walkthrough with real numbers.

**3. Open lesson 1.** [`primer.ml.big_picture`](primer/ml/big_picture.py) follows one prompt all the
way through a model, and every later lesson zooms into one box of that picture.
When a formula stops you, open the reference, [`primer.notation`](primer/notation.py): it turns every
symbol you'll meet (Σ, ‖x‖, ∂, ᵀ) into a few lines of Python. It is not a first lesson.

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

Generated from [`primer/curriculum.py`](primer/curriculum.py), so it is always current. Six parts, each with what it teaches and what you can do afterwards. Lesson 0 is a reference: open it when Σ, ‖x‖ or ∂ stop you.

<!-- BEGIN curriculum -->
### Reference: reading the symbols

Every symbol an AI formula uses (Σ, ‖x‖, ∂, ᵀ), decoded as a short loop with numbers you can check by hand. It is not the first lesson: open it when a formula stops you, from the link beside any formula.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 0 | [Math notation, from zero](primer/notation.py) | Every symbol in an ML formula, as a short loop | [page](https://rumblr.com/learn/notation) · [tests](tests/test_notation.py) |

### Part 1: how a language model works

The machinery inside a language model: a neural network, the loss and the optimizer that train it, how text becomes tokens, and the attention and transformer blocks that turn them into a reply. If you read one lesson here besides the first, read Attention.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 1 | [The big picture](primer/ml/big_picture.py) | What happens, end to end, when you send a prompt | [page](https://rumblr.com/learn/big-picture) · [tests](tests/test_big_picture.py) |
| 2 | [Neural networks](primer/ml/neural_net.py) | How a network of simple units learns from its mistakes, built and trained by hand | [page](https://rumblr.com/learn/neural-net) · [tests](tests/test_neural_net.py) |
| 3 | [Loss functions](primer/ml/losses.py) | The one number training pushes down, and why choosing it is choosing the behaviour | [page](https://rumblr.com/learn/losses) · [tests](tests/test_losses.py) |
| 4 | [Optimizers](primer/ml/optimizers.py) | How a step is taken down the loss, and why the learning rate matters more than the rule | [page](https://rumblr.com/learn/optimizers) · [tests](tests/test_optimizers.py) |
| 5 | [Training deep networks](primer/ml/deep_nets.py) | Why deep stacks stop learning, and the fixes inside every transformer block | [page](https://rumblr.com/learn/deep-nets) · [tests](tests/test_deep_nets.py) |
| 6 | [Attention](primer/ml/attention.py) | How each word looks at the others, and why the cost grows with the square of the length | [page](https://rumblr.com/learn/attention) · [tests](tests/test_attention.py) |
| 7 | [Positional information](primer/ml/positional.py) | Why order must be added, sinusoids and RoPE | [page](https://rumblr.com/learn/positional) · [tests](tests/test_positional.py) |
| 8 | [The transformer](primer/ml/transformer.py) | The block every language model stacks, and how to turn a config file into gigabytes | [page](https://rumblr.com/learn/transformer) · [tests](tests/test_transformer.py) |
| 9 | [Tokenization](primer/ml/tokenization.py) | How text becomes the pieces a model reads, and why models miscount letters | [page](https://rumblr.com/learn/tokenization) · [tests](tests/test_tokenization.py) |

### Part 2: how it is trained and judged

How a language model gets its weights: pretraining on a large share of the web, then fine-tuning, reward training, reasoning and alignment. Along the way it covers how to judge the result: overfitting, metrics and benchmark scores. In a hurry, read Training stages and Reading benchmarks.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 10 | [Training stages](primer/ml/training_stages.py) | The stages that turn a random network into an assistant, and when to fine-tune instead of prompt | [page](https://rumblr.com/learn/training-stages) · [tests](tests/test_training_stages.py) |
| 11 | [Pretraining at scale](primer/ml/pretraining.py) | What it takes to train on the internet: the data belt, parallelism across GPUs, mixed precision | [page](https://rumblr.com/learn/pretraining) · [tests](tests/test_pretraining.py) |
| 12 | [Fine-tuning in practice](primer/ml/fine_tuning.py) | What a fine-tune costs and risks, and how to prepare data so the model keeps its old skills | [page](https://rumblr.com/learn/fine-tuning) · [tests](tests/test_fine_tuning.py) |
| 13 | [Overfitting and regularization](primer/ml/regularization.py) | Why a model memorises instead of learning, and the checks that catch it | [page](https://rumblr.com/learn/regularization) · [tests](tests/test_regularization.py) |
| 14 | [Reinforcement learning](primer/ml/reinforcement.py) | How a model learns from a score instead of an answer, and how it games the score | [page](https://rumblr.com/learn/reinforcement) · [tests](tests/test_reinforcement.py) |
| 15 | [Reasoning models](primer/ml/reasoning.py) | Why letting a model think before it answers helps, and when the extra tokens are worth it | [page](https://rumblr.com/learn/reasoning) · [tests](tests/test_reasoning.py) |
| 16 | [Alignment and safety](primer/ml/alignment.py) | How a model is taught to behave, and why a score that keeps rising can hide a worse model | [page](https://rumblr.com/learn/alignment) · [tests](tests/test_alignment.py) |
| 17 | [Metrics](primer/ml/metrics.py) | The numbers that decide whether a model ships, and the one that fools you on rare events | [page](https://rumblr.com/learn/metrics) · [tests](tests/test_metrics.py) |
| 18 | [Reading benchmarks](primer/ml/benchmarks.py) | What a benchmark score can and cannot tell you, and how wide its error bar really is | [page](https://rumblr.com/learn/benchmarks) · [tests](tests/test_benchmarks.py) |

### Part 3: how it is run

What it takes, in time and memory, to turn a trained model into answers: the chip, the cache, the sampling step and the designs that make long inputs affordable. Latency and GPU bills follow from arithmetic you can do by hand. The last lesson looks inside a trained model at what it learned.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 19 | [The hardware underneath](primer/ml/hardware.py) | Why a GPU is fast at some work and slow at the rest, and how to read a spec sheet | [page](https://rumblr.com/learn/hardware) · [tests](tests/test_hardware.py) |
| 20 | [Inference](primer/ml/inference.py) | Why the first token is slow and the rest are cheap, and what memory a conversation costs | [page](https://rumblr.com/learn/inference) · [tests](tests/test_inference.py) |
| 21 | [Structured output](primer/ml/structured_output.py) | How to guarantee valid JSON by controlling which tokens a model may pick | [page](https://rumblr.com/learn/structured-output) · [tests](tests/test_structured_output.py) |
| 22 | [Long context and efficient architectures](primer/ml/efficient_architectures.py) | How a model handles a million tokens without a million-squared attention | [page](https://rumblr.com/learn/efficient-architectures) · [tests](tests/test_efficient_architectures.py) |
| 23 | [Looking inside the model](primer/ml/interpretability.py) | How to read the numbers inside a trained model and find what a layer represents | [page](https://rumblr.com/learn/interpretability) · [tests](tests/test_interpretability.py) |

### Part 4: embeddings and search

Embeddings are lists of numbers that place similar meanings close together. This part builds them, then the search systems that use them to find, among millions of passages, the few that answer a question. The retrieval systems in Part 6 are built on it.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 24 | [Word embeddings](primer/ml/embeddings/word2vec.py) | How a word becomes a vector that captures its meaning, and what that vector cannot capture | [page](https://rumblr.com/learn/word2vec) · [tests](tests/test_emb_word2vec.py) |
| 25 | [Similarity](primer/ml/embeddings/similarity.py) | How to compare two embeddings, and why the wrong metric ranks silently wrong | [page](https://rumblr.com/learn/similarity) · [tests](tests/test_emb_similarity.py) |
| 26 | [Training embedding models](primer/ml/embeddings/contrastive.py) | How an embedding model is taught what similar means, from pairs of texts that belong together | [page](https://rumblr.com/learn/contrastive) · [tests](tests/test_emb_contrastive.py) |
| 27 | [Dimensions and compression](primer/ml/embeddings/compression.py) | How to store millions of embeddings in a fraction of the memory without losing your neighbours | [page](https://rumblr.com/learn/compression) · [tests](tests/test_emb_compression.py) |
| 28 | [Vector indexes](primer/ml/embeddings/ann.py) | How an index finds the nearest vectors among millions without checking every one | [page](https://rumblr.com/learn/ann) · [tests](tests/test_emb_ann.py) |
| 29 | [Retrieval](primer/ml/embeddings/retrieval.py) | The whole search pipeline: keywords, embeddings, hybrid ranking, rerankers and chunking | [page](https://rumblr.com/learn/retrieval) · [tests](tests/test_emb_retrieval.py) |
| 30 | [Clustering and matching](primer/ml/embeddings/clustering.py) | How embeddings group a pile of unlabelled text, find duplicates and route requests | [page](https://rumblr.com/learn/clustering) · [tests](tests/test_emb_clustering.py) |
| 31 | [Embeddings in production](primer/ml/embeddings/operations.py) | What breaks when an embedding index serves real traffic, and how to measure retrieval on its own | [page](https://rumblr.com/learn/operations) · [tests](tests/test_emb_operations.py) |

### Part 5: other kinds of model, and generating media

Models that are not transformers, and models that make pictures and sound: trees for tables of data, convolutions for images, and the autoencoders, GANs and diffusion models behind generated images, audio and video. It ends with models that see and hear. Skip it if your work is text on a hosted model.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 32 | [Trees and boosting](primer/ml/classical.py) | How a tree learns a table of data, and why boosted trees still beat neural networks there | [page](https://rumblr.com/learn/classical) · [tests](tests/test_classical.py) |
| 33 | [CNNs and RNNs](primer/ml/cnn_rnn.py) | How convolutions see and recurrent nets remember, and why transformers won | [page](https://rumblr.com/learn/cnn-rnn) · [tests](tests/test_cnn_rnn.py) |
| 34 | [Autoencoders and VAEs](primer/ml/generative/autoencoders.py) | Squeezing data into a code and back, and sampling new data from it | [page](https://rumblr.com/learn/autoencoders) · [tests](tests/test_autoencoders.py) |
| 35 | [GANs](primer/ml/generative/gans.py) | A forger against a detective: adversarial training, and why it is unstable | [page](https://rumblr.com/learn/gans) · [tests](tests/test_gans.py) |
| 36 | [Diffusion and flow matching](primer/ml/generative/diffusion.py) | Turning noise into images one small step at a time | [page](https://rumblr.com/learn/diffusion) · [tests](tests/test_diffusion.py) |
| 37 | [Multimodal models](primer/ml/generative/multimodal.py) | Images, audio and video into a language model | [page](https://rumblr.com/learn/multimodal) · [tests](tests/test_multimodal.py) |

### Part 6: building systems people rely on

How to build on a language model: the message exchange, the agent loop, tools, retrieval, context and memory. Then how to know it works and keep it safe: evaluation, guardrails, cost, observability and deployment. The last lesson is about why hard tasks still fail.

| # | Lesson | What you'll be able to explain | Read |
|---|---|---|---|
| 38 | [Talking to a model](primer/agents/llm.py) | The message format, and what tool calling really is | [page](https://rumblr.com/learn/llm) · [tests](tests/test_agents_llm.py) |
| 39 | [Orchestration](primer/agents/orchestration.py) | When to script the steps and when to let the model choose them, and the named patterns in between | [page](https://rumblr.com/learn/orchestration) · [tests](tests/test_agents_orchestration.py) |
| 40 | [The agent loop](primer/agents/agent_loop.py) | The loop that runs a model's errands, with the limits that let it stop | [page](https://rumblr.com/learn/agent-loop) · [tests](tests/test_agents_agent_loop.py) |
| 41 | [Tools](primer/agents/tools.py) | How to design a tool so a wrong call is caught before it costs money | [page](https://rumblr.com/learn/tools) · [tests](tests/test_agents_tools.py) |
| 42 | [Coding and computer-use agents](primer/agents/coding_agents.py) | Why agents became dependable on code first, and how one drives a repository or a screen | [page](https://rumblr.com/learn/coding-agents) · [tests](tests/test_agents_coding_agents.py) |
| 43 | [Model Context Protocol](primer/agents/mcp.py) | How one tool can serve every application, what goes over the wire, and the risks of a stranger's tools | [page](https://rumblr.com/learn/mcp) · [tests](tests/test_agents_mcp.py) |
| 44 | [Retrieval-augmented generation](primer/agents/rag.py) | How a model answers from your documents, with citations and access control | [page](https://rumblr.com/learn/rag) · [tests](tests/test_agents_rag.py) |
| 45 | [Context engineering](primer/agents/context.py) | What goes in the prompt when code assembles it, and why an agent gets worse as a session runs | [page](https://rumblr.com/learn/context) · [tests](tests/test_agents_context.py) |
| 46 | [Memory](primer/agents/memory.py) | How an agent carries what it learned across conversations, and keeps one user's memory from another's | [page](https://rumblr.com/learn/memory) · [tests](tests/test_agents_memory.py) |
| 47 | [Planning](primer/agents/planning.py) | Why a twenty-step task fails when a three-step demo worked, and how a plan survives it | [page](https://rumblr.com/learn/planning) · [tests](tests/test_agents_planning.py) |
| 48 | [Evaluation](primer/agents/evals.py) | How to know a change helped before a customer finds out that it did not | [page](https://rumblr.com/learn/evals) · [tests](tests/test_agents_evals.py) |
| 49 | [Guardrails](primer/agents/guardrails.py) | How to keep text a model reads from becoming commands it follows, and check what it sends out | [page](https://rumblr.com/learn/guardrails) · [tests](tests/test_agents_guardrails.py) |
| 50 | [Cost and latency](primer/agents/cost.py) | The one number that decides cost, and the levers that move it | [page](https://rumblr.com/learn/cost) · [tests](tests/test_agents_cost.py) |
| 51 | [Observability](primer/agents/observability.py) | How to find out what an agent did yesterday, and turn that record into the next improvement | [page](https://rumblr.com/learn/observability) · [tests](tests/test_agents_observability.py) |
| 52 | [Safe deployment](primer/agents/deployment.py) | How an agent earns the right to act, one step of trust at a time | [page](https://rumblr.com/learn/deployment) · [tests](tests/test_agents_deployment.py) |
| 53 | [Why the hard ones fail](primer/agents/failures.py) | The common failure modes, and the fix for each | [page](https://rumblr.com/learn/failures) · [tests](tests/test_agents_failures.py) |

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

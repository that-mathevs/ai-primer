# Big questions: the map from the top down

The lessons build the field from the bottom up. Real conversations about AI systems start from the top, with questions like these. For each one: the lessons that answer it, in order, and the short version, the main ideas in the order that builds understanding. Read the short version first, then open the lessons wherever you want the full story.

Generated from [`primer/curriculum.py`](../primer/curriculum.py) by `make readme`; edit it there.


## 1. What happens, step by step, when I send a prompt to a language model?

**Route:** [Math notation, from zero](../primer/notation.py) → [The big picture](../primer/ml/big_picture.py) → [Tokenization](../primer/ml/tokenization.py) → [Attention](../primer/ml/attention.py) → [Positional information](../primer/ml/positional.py) → [The transformer](../primer/ml/transformer.py) → [Inference](../primer/ml/inference.py)

**In brief:**

1. Tokenizer: text becomes subword IDs; cost and context limits are counted in tokens.
2. Embedding lookup turns each ID into a vector; position information is mixed in.
3. Dozens of transformer blocks: attention mixes information across tokens, the feed-forward layer processes each token.
4. The last position's vector becomes a score for every vocabulary token; softmax turns scores into probabilities.
5. Sampling (temperature, top-p) picks one token, which is appended; the loop repeats until a stop token.
6. Prefill processes the prompt in parallel; decode generates one token at a time, made cheap by the KV cache.


## 2. How does a neural network actually learn?

**Route:** [Neural networks](../primer/ml/neural_net.py) → [Loss functions](../primer/ml/losses.py) → [Optimizers](../primer/ml/optimizers.py) → [Training deep networks](../primer/ml/deep_nets.py) → [Overfitting and regularization](../primer/ml/regularization.py)

**In brief:**

1. A forward pass makes a prediction; a loss turns 'how wrong' into one number.
2. Backpropagation applies the chain rule to find every weight's gradient.
3. An optimizer (SGD, Adam/AdamW) steps each weight against its gradient; the learning rate sets the step size.
4. Depth brings vanishing and exploding gradients; residual connections, normalization and good initialization fix them.
5. Watch validation loss: when it rises while training loss falls, the model is overfitting; regularize or stop early.


## 3. How does attention work, and why did transformers replace RNNs?

**Route:** [Attention](../primer/ml/attention.py) → [Positional information](../primer/ml/positional.py) → [The transformer](../primer/ml/transformer.py) → [CNNs and RNNs](../primer/ml/cnn_rnn.py)

**In brief:**

1. Each token forms a query, key and value; query-key dot products score relevance; softmax turns scores into weights; the output blends values.
2. Scores are divided by the square root of d_k so softmax doesn't saturate and gradients keep flowing.
3. A causal mask hides future tokens, which makes next-token training honest and generation cacheable.
4. Multi-head attention runs several attentions in parallel; grouped-query attention shares keys and values to shrink the KV cache.
5. RNNs pass everything through one hidden state, one step at a time; attention gives every pair of tokens a direct path and trains in parallel.
6. The price is O(n²) cost in sequence length, which FlashAttention, sparse attention and state-space models attack.


## 4. How are large language models trained, and when should I fine-tune instead of using RAG?

**Route:** [Training stages](../primer/ml/training_stages.py) → [Fine-tuning in practice](../primer/ml/fine_tuning.py) → [Tokenization](../primer/ml/tokenization.py) → [Loss functions](../primer/ml/losses.py) → [Embeddings in production](../primer/ml/embeddings/operations.py) → [Retrieval-augmented generation](../primer/agents/rag.py)

**In brief:**

1. Pretraining: next-token prediction over trillions of tokens produces a knowledgeable base model.
2. Supervised fine-tuning teaches the assistant format; preference tuning (RLHF or DPO) shapes helpfulness and safety.
3. Adaptation, cheapest first: prompting, then RAG, then LoRA, then (rarely) a full fine-tune.
4. Fine-tuning changes behavior; RAG supplies knowledge that changes or must be cited.
5. Distillation trains a small model to imitate a large one, often the biggest production cost win.


## 5. What makes serving a model fast and affordable?

**Route:** [The hardware underneath](../primer/ml/hardware.py) → [Inference](../primer/ml/inference.py) → [Long context and efficient architectures](../primer/ml/efficient_architectures.py) → [Attention](../primer/ml/attention.py) → [Cost and latency](../primer/agents/cost.py) → [Context engineering](../primer/agents/context.py)

**In brief:**

1. Prefill is compute-bound and sets time to first token; decode is memory-bound and sets tokens per second.
2. The KV cache trades GPU memory for speed; its size is 2 × layers × KV heads × head dimension × bytes, per token.
3. Memory math: weights = parameters × bytes per parameter (70B at 16-bit is about 140 GB).
4. Speedups: quantization, continuous batching, speculative decoding, grouped-query attention, prompt caching.
5. At the system level: route easy work to small models, cache stable prefixes, trim tokens, batch offline work.


## 6. What is an embedding, and how is an embedding model trained?

**Route:** [Word embeddings](../primer/ml/embeddings/word2vec.py) → [Training embedding models](../primer/ml/embeddings/contrastive.py) → [Similarity](../primer/ml/embeddings/similarity.py) → [Loss functions](../primer/ml/losses.py)

**In brief:**

1. An embedding is a learned vector where closeness means similar meaning.
2. word2vec learned one vector per word from co-occurrence; contextual models give each token a vector that depends on its sentence.
3. Sentence embeddings pool token vectors; models trained for similarity beat plain pooled encoders.
4. Contrastive training pulls matching pairs together and pushes others apart, using in-batch negatives (InfoNCE).
5. Hard negatives (right topic, wrong answer) are the biggest driver of retrieval quality.
6. CLIP applies the same idea across images and text, putting both in one space.


## 7. How do you search millions of vectors quickly, and what does it cost?

**Route:** [Similarity](../primer/ml/embeddings/similarity.py) → [Dimensions and compression](../primer/ml/embeddings/compression.py) → [Vector indexes](../primer/ml/embeddings/ann.py)

**In brief:**

1. On normalized vectors, cosine, dot product and Euclidean distance give the same ranking; use what the model was trained with.
2. Storage math: vectors × dimensions × 4 bytes (10M × 1536 is about 61 GB) before index overhead.
3. Exact search is too slow at scale; approximate indexes trade a little recall for a lot of speed.
4. HNSW: layered graph, long jumps on top, local search at the bottom; M, efConstruction and efSearch are the knobs.
5. IVF searches only the nearest clusters (nprobe); PQ compresses vectors into codes.
6. Matryoshka truncation and scalar or binary quantization shrink memory; rescoring the shortlist recovers accuracy.


## 8. How do you build retrieval that returns the right passages?

**Route:** [Retrieval](../primer/ml/embeddings/retrieval.py) → [Clustering and matching](../primer/ml/embeddings/clustering.py) → [Embeddings in production](../primer/ml/embeddings/operations.py) → [Metrics](../primer/ml/metrics.py) → [Retrieval-augmented generation](../primer/agents/rag.py)

**In brief:**

1. Chunk on document structure, with overlap and metadata; chunking often matters more than the model.
2. Dense search finds meaning; BM25 finds exact IDs and rare terms; hybrid search fuses both with reciprocal rank fusion.
3. Retrieve wide with a bi-encoder, then rerank the shortlist with a cross-encoder.
4. Measure retrieval on its own with recall@k, MRR and nDCG on a labeled set before tuning prompts.
5. Operations: new embedding model means re-embedding everything; version indexes and switch traffic behind an alias.


## 9. How do you know if a model or an agent is any good?

**Route:** [Metrics](../primer/ml/metrics.py) → [Reading benchmarks](../primer/ml/benchmarks.py) → [Loss functions](../primer/ml/losses.py) → [Overfitting and regularization](../primer/ml/regularization.py) → [Evaluation](../primer/agents/evals.py)

**In brief:**

1. Pick metrics by the cost of each error: precision vs. recall; accuracy misleads on imbalanced data.
2. Keep training, validation and test data apart, and watch for leakage and benchmark contamination.
3. For agents: a golden set of real tasks, graded by code wherever possible (end state, schema, tests).
4. For open-ended output: an LLM judge with an explicit rubric, calibrated against human labels.
5. Track trajectory, cost and latency beside quality; run the suite on every change and feed production failures back in.


## 10. When should you build an agent, and how does one work?

**Route:** [Talking to a model](../primer/agents/llm.py) → [Orchestration](../primer/agents/orchestration.py) → [The agent loop](../primer/agents/agent_loop.py) → [Tools](../primer/agents/tools.py) → [Coding and computer-use agents](../primer/agents/coding_agents.py) → [Model Context Protocol](../primer/agents/mcp.py) → [Planning](../primer/agents/planning.py)

**In brief:**

1. Use the least autonomy that solves the problem: fixed workflow, then router, then agent loop, then multiple agents.
2. Tool calling: the model emits a structured request; your code validates it, runs it and returns the result. The model executes nothing.
3. The loop: think, call a tool, observe, decide, with step limits, token budgets and loop detection.
4. Tool design is prompt design: few, high-level tools with precise descriptions, validated arguments and actionable errors.
5. Long tasks fail by compounding error (0.95¹⁰ ≈ 0.60), so plan, verify each step externally, and checkpoint.
6. MCP standardizes how apps connect to tools, and brings its own risks: tool poisoning, rug pulls, broad permissions.


## 11. How do you give an AI system the right context and memory?

**Route:** [Context engineering](../primer/agents/context.py) → [Memory](../primer/agents/memory.py) → [Retrieval-augmented generation](../primer/agents/rag.py)

**In brief:**

1. Context engineering: the smallest set of high-signal content, structured with clear delimiters.
2. More context is not better: models use the middle of long inputs least reliably, and quality rots as sessions grow.
3. Put stable content first so prompt caching can reuse it; summarize or drop old turns; compress tool outputs.
4. Long-term memory is retrieval over the system's own history: episodic, semantic and procedural.
5. Memory must be isolated per tenant and user, updatable, and deletable on request.


## 12. How do you make an AI system safe to put in front of real users?

**Route:** [Alignment and safety](../primer/ml/alignment.py) → [Guardrails](../primer/agents/guardrails.py) → [Tools](../primer/agents/tools.py) → [Safe deployment](../primer/agents/deployment.py) → [Observability](../primer/agents/observability.py)

**In brief:**

1. Layer guardrails on inputs, outputs and actions; no single check is reliable alone.
2. Treat retrieved content, emails and tool outputs as untrusted: no prompt wording fully prevents injection.
3. Privilege separation: the part that reads untrusted content holds no dangerous tools; actions pass a policy or human check.
4. Graduate autonomy with evidence: shadow mode, then approval per action, then autonomy for low-risk actions.
5. Trace every run, keep tamper-evident audit logs, rate-limit actions and keep a kill switch and a one-step rollback.


## 13. How do you cut cost and latency without hurting quality?

**Route:** [Cost and latency](../primer/agents/cost.py) → [Context engineering](../primer/agents/context.py) → [Inference](../primer/ml/inference.py) → [Clustering and matching](../primer/ml/embeddings/clustering.py)

**In brief:**

1. Measure cost per successful task, not per call.
2. Route each step to the cheapest model that handles it; this is usually the biggest lever.
3. Cache: prompt caching for stable prefixes, response and semantic caches for repeated questions.
4. Trim tokens: tight prompts, compressed tool outputs, only the top reranked chunks.
5. Run independent tool calls in parallel, stream output, and move offline work to batch APIs.
6. Enforce per-task and per-tenant budgets with anomaly alerts.


## 14. Why do AI systems fail in production, and how do you fix them?

**Route:** [Why the hard ones fail](../primer/agents/failures.py) → [Structured output](../primer/ml/structured_output.py) → [Planning](../primer/agents/planning.py) → [Retrieval-augmented generation](../primer/agents/rag.py) → [Evaluation](../primer/agents/evals.py) → [Observability](../primer/agents/observability.py)

**In brief:**

1. Compounding error over long tasks: shorten paths, verify steps, checkpoint.
2. Bad retrieval behind confident wrong answers: hybrid search, reranking, retrieval evals.
3. Ambiguous tools, loops and runaway cost: better tool design, budgets, loop detection.
4. Prompt injection and messy enterprise data: untrusted-content boundaries, permission-aware retrieval, investment in parsing.
5. No evals and no traces: regressions ship silently; build the loop from production failure to trace to test case to fix.


## 15. What does it take to pretrain a large model?

**Route:** [Training stages](../primer/ml/training_stages.py) → [Tokenization](../primer/ml/tokenization.py) → [Pretraining at scale](../primer/ml/pretraining.py) → [Optimizers](../primer/ml/optimizers.py) → [The hardware underneath](../primer/ml/hardware.py)

**In brief:**

1. Most of a web crawl is thrown away: language ID, quality rules and classifiers, and exact and near-duplicate removal.
2. Sources are mixed by weight, not size; about 20 tokens per parameter is compute-optimal, but models meant for heavy use train far longer.
3. Adam in mixed precision needs about 16 bytes per parameter before activations, so one GPU can't hold a large model.
4. Data parallelism shares gradients, ZeRO/FSDP shards the training state, tensor parallelism splits each matrix multiply, and pipeline parallelism splits the layers.
5. The maths runs in bf16 or fp8 with scaling, while the master weights stay in fp32.
6. Warmup, gradient clipping, spike rollback and regular checkpoints keep a months-long run alive.


## 16. How does a model learn from rewards instead of examples?

**Route:** [Reinforcement learning](../primer/ml/reinforcement.py) → [Training stages](../primer/ml/training_stages.py) → [Reasoning models](../primer/ml/reasoning.py) → [Alignment and safety](../primer/ml/alignment.py)

**In brief:**

1. Reinforcement learning samples an action, scores it, and makes high-scoring actions more likely.
2. A baseline turns rewards into advantages (better or worse than usual), which cuts noise without bias.
3. PPO reuses each batch for several steps, clips how far the policy moves, and leashes it to a reference with a KL penalty.
4. GRPO drops the value network by comparing rewards within a group of answers to the same prompt.
5. A checker as the reward (the right answer, passing tests) is how reasoning models are trained.
6. The policy optimises the reward you wrote, not the goal you meant; verifiable rewards, a KL leash and held-out checks defend against that.


## 17. How do reasoning models think, and when is extra thinking worth it?

**Route:** [Inference](../primer/ml/inference.py) → [Reinforcement learning](../primer/ml/reinforcement.py) → [Reasoning models](../primer/ml/reasoning.py) → [Planning](../primer/agents/planning.py) → [Cost and latency](../primer/agents/cost.py)

**In brief:**

1. Every written token is another forward pass, so a chain of thought buys serial computation, and the text is the model's working memory.
2. Test-time compute can go into one longer chain, or into many chains with a vote or a verifier picking one answer.
3. Voting helps only when the right answer is the most common one and the samples' mistakes are independent.
4. Checking each step catches errors that checking only the final answer misses.
5. Training with verifiable rewards makes longer, self-checking reasoning emerge.
6. Thinking is paid for per token and slips compound over long chains, so route easy tasks to little thinking and measure cost per successful task.


## 18. How do models handle very long contexts?

**Route:** [Attention](../primer/ml/attention.py) → [Positional information](../primer/ml/positional.py) → [Inference](../primer/ml/inference.py) → [Long context and efficient architectures](../primer/ml/efficient_architectures.py)

**In brief:**

1. Attention scores every pair of tokens, and the KV cache grows with every token, per layer, per conversation.
2. Sliding windows and sparse patterns score fewer pairs; stacked layers still carry information far.
3. Linear attention and state-space models keep a fixed-size summary: linear time and constant memory, but blurrier recall.
4. Mamba makes the summary selective: each token decides how much to keep and how much to write.
5. Hybrids keep a few attention layers for exact lookup.
6. The cache shrinks by sharing key/value heads, caching a small latent, or storing fewer bits.


## 19. What is going on inside a trained model, and how can we tell?

**Route:** [The transformer](../primer/ml/transformer.py) → [Word embeddings](../primer/ml/embeddings/word2vec.py) → [Looking inside the model](../primer/ml/interpretability.py) → [Alignment and safety](../primer/ml/alignment.py)

**In brief:**

1. Models store features as directions across many neurons, not one feature per neuron.
2. Probes and the logit lens read what is present; they show correlation, not use.
3. Activation patching changes one activation and watches the output: the causal test.
4. Sparse features get packed in superposition, which makes individual neurons respond to several things.
5. Sparse autoencoders unpack superposition into interpretable features, at the cost of some unexplained activity.
6. These tools give evidence, not proof; full explanations exist only for narrow behaviours.


## 20. When is a neural network the wrong tool?

**Route:** [Trees and boosting](../primer/ml/classical.py) → [Neural networks](../primer/ml/neural_net.py) → [Overfitting and regularization](../primer/ml/regularization.py) → [Metrics](../primer/ml/metrics.py)

**In brief:**

1. On tables whose columns each mean something alone, gradient-boosted trees or a random forest are the model to beat.
2. A tree asks one column at a time whether it's above a threshold, so it needs no feature scaling and handles categories natively.
3. A single deep tree overfits; forests average many decorrelated trees, and boosting adds small trees fit to the remaining errors.
4. Neural networks win when meaning lives in arrangements of raw values (images, audio, text), when data is huge, or when a pretrained model can be reused.
5. Trees can't extrapolate beyond the values they trained on, and impurity importances credit noise, so check importances on held-out data.


## 21. How do AI models generate images, audio and video?

**Route:** [Autoencoders and VAEs](../primer/ml/generative/autoencoders.py) → [GANs](../primer/ml/generative/gans.py) → [Diffusion and flow matching](../primer/ml/generative/diffusion.py) → [Multimodal models](../primer/ml/generative/multimodal.py)

**In brief:**

1. A generator learns a whole distribution, so it can sample new examples; predicting the average gives blur.
2. An autoencoder squeezes data into a small code and back; a VAE shapes that code so random codes decode to new data.
3. A GAN trains a generator against a discriminator: sharp, one-pass samples, but unstable training and mode collapse.
4. Diffusion adds noise on purpose and learns to remove it, generating from pure noise in many small steps.
5. Flow matching learns straight paths from noise to data, so it needs fewer steps; guidance trades variety for following the prompt.
6. Real systems denoise an autoencoder's latent with a transformer that reads the prompt; video and audio are the same idea with more tokens.


## 22. How do AI models see images and hear audio?

**Route:** [CNNs and RNNs](../primer/ml/cnn_rnn.py) → [Training embedding models](../primer/ml/embeddings/contrastive.py) → [Multimodal models](../primer/ml/generative/multimodal.py)

**In brief:**

1. Every modality becomes a sequence of vectors a transformer attends over.
2. Images become patch tokens, (H/P)·(W/P) of them, so cost grows with the square of the resolution.
3. A small projector connects a vision or audio encoder to a language model; it is trained first, with both models frozen.
4. Audio becomes a log-mel spectrogram, then about 50 tokens a second.
5. Video multiplies image tokens by time, so frames are sampled; text stays the densest input.

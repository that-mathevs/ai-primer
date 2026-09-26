window.PRIMER_GLOSSARY = {
 "acceptance rate": {
  "def": "How often the big model keeps a drafted token in speculative decoding.",
  "lesson": "primer/ml/inference.html",
  "term": "acceptance rate"
 },
 "accuracy": {
  "def": "The share of all predictions that were correct. Misleading when one class is rare.",
  "lesson": "primer/ml/metrics.html",
  "term": "accuracy"
 },
 "acl": {
  "def": "Access-control list: the users or groups allowed to read a document.",
  "lesson": "primer/agents/rag.html",
  "term": "ACL"
 },
 "activation checkpointing": {
  "def": "Keeping only each layer's input and recomputing the rest during the backward pass.",
  "lesson": "primer/ml/pretraining.html",
  "term": "activation checkpointing"
 },
 "activation function": {
  "def": "The nonlinear function inside a neuron, such as ReLU or GELU. Without it, stacked layers collapse into one.",
  "lesson": "primer/ml/neural_net.html",
  "term": "activation function"
 },
 "activation patching": {
  "def": "Copying one activation from a clean run into a corrupted run to measure how much of the right answer returns; also called causal tracing.",
  "lesson": "primer/ml/interpretability.html",
  "term": "activation patching"
 },
 "adagrad": {
  "def": "An optimizer that divides each weight's step by the square root of the sum of all its past squared gradients, so steps only ever shrink.",
  "lesson": "primer/ml/optimizers.html",
  "term": "adagrad"
 },
 "adam": {
  "def": "An optimizer that adapts the step size for each weight using running averages of its gradients.",
  "lesson": "primer/ml/optimizers.html",
  "term": "Adam"
 },
 "adamw": {
  "def": "Adam with weight decay applied separately from the gradient step. The default optimizer for transformers.",
  "lesson": "primer/ml/optimizers.html",
  "term": "AdamW"
 },
 "adapter": {
  "def": "A small trainable module added beside frozen weights to specialise a model.",
  "lesson": "primer/ml/training_stages.html",
  "term": "adapter"
 },
 "advantage": {
  "def": "How much better an action did than typical: reward minus baseline.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "advantage"
 },
 "agent": {
  "def": "A system where a language model decides which tools to call, looks at the results and decides the next step, in a loop until done.",
  "lesson": "primer/agents/agent_loop.html",
  "term": "agent"
 },
 "agent loop": {
  "def": "The cycle an agent repeats: think, call a tool, observe the result, decide what next.",
  "lesson": "primer/agents/agent_loop.html",
  "term": "agent loop"
 },
 "agentic rag": {
  "def": "RAG where the model decides whether, what and how often to search.",
  "lesson": "primer/agents/rag.html",
  "term": "agentic RAG"
 },
 "alibi": {
  "def": "A position scheme that adds no position vectors and instead subtracts a penalty proportional to distance from each attention score.",
  "lesson": "primer/ml/positional.html",
  "term": "alibi"
 },
 "alignment": {
  "def": "Making a model's behaviour match what we want (helpful, honest, harmless), not just the measurements we optimize.",
  "lesson": "primer/ml/alignment.html",
  "term": "alignment"
 },
 "alignment tax": {
  "def": "Capability lost as a side effect of training a model to be helpful, honest and harmless.",
  "lesson": "primer/ml/training_stages.html",
  "term": "alignment tax"
 },
 "all-reduce": {
  "def": "Summing a value across GPUs so every GPU ends up holding the total.",
  "lesson": "primer/ml/hardware.html",
  "term": "all-reduce"
 },
 "anisotropy": {
  "def": "When a model's vectors all crowd into a narrow cone, so even unrelated texts score as fairly similar.",
  "lesson": "primer/ml/embeddings/similarity.html",
  "term": "anisotropy"
 },
 "ann": {
  "def": "Approximate nearest neighbor search: trade a little recall for a lot of speed.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "ANN"
 },
 "anomaly alert": {
  "def": "An alert when a metric jumps far outside its usual range.",
  "lesson": "primer/agents/cost.html",
  "term": "anomaly alert"
 },
 "anomaly detection": {
  "def": "Flagging items far from every known group.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "anomaly detection"
 },
 "approximate nearest neighbor": {
  "def": "Finding vectors close to a query quickly by searching only part of the collection, accepting a small chance of missing the true closest.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "approximate nearest neighbor"
 },
 "arena": {
  "def": "Ratings built from people voting between two anonymous answers.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "arena"
 },
 "argmax": {
  "def": "The position of the largest value in a list, rather than the value itself.",
  "lesson": "primer/notation.html",
  "term": "argmax"
 },
 "arithmetic intensity": {
  "def": "Operations performed per byte read from memory.",
  "lesson": "primer/ml/inference.html",
  "term": "arithmetic intensity"
 },
 "asymmetric distance computation": {
  "def": "Scoring compressed vectors against an uncompressed query by adding up precomputed table lookups.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "asymmetric distance computation"
 },
 "attack success rate": {
  "def": "The share of red-team attempts that get past a safety check.",
  "lesson": "primer/ml/alignment.html",
  "term": "attack success rate"
 },
 "attention": {
  "def": "The mechanism that lets each token look at every other token, score how relevant each one is, and blend in information from the relevant ones.",
  "lesson": "primer/ml/attention.html",
  "term": "attention"
 },
 "attention sink": {
  "def": "Early tokens that trained models park spare attention on; dropping them breaks streaming generation.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "attention sink"
 },
 "audit log": {
  "def": "A record of which actor took which action, for whom, with what inputs.",
  "lesson": "primer/agents/deployment.html",
  "term": "audit log"
 },
 "audit trail": {
  "def": "A tamper-evident log of which agent did what, for whom and with what authority.",
  "lesson": "primer/agents/deployment.html",
  "term": "audit trail"
 },
 "auto-regressive": {
  "def": "Generating one token at a time, each predicted from everything before it and then fed back in.",
  "lesson": "primer/ml/big_picture.html",
  "term": "auto-regressive"
 },
 "autoencoder": {
  "def": "A network that squeezes its input into a small code and rebuilds the input from it, trained only to make the rebuild match.",
  "lesson": "primer/ml/generative/autoencoders.html",
  "term": "autoencoder"
 },
 "autoregressive generation": {
  "def": "Producing text one token at a time, where each new token is predicted from all the tokens before it and then appended.",
  "lesson": "primer/ml/big_picture.html",
  "term": "autoregressive generation"
 },
 "backfill": {
  "def": "Re-processing existing records into a new system, such as re-embedding a corpus with a new model.",
  "lesson": "primer/ml/embeddings/operations.html",
  "term": "backfill"
 },
 "backpropagation": {
  "def": "The chain rule run backwards through the network to find, for every weight, how much nudging it would change the loss.",
  "lesson": "primer/ml/neural_net.html",
  "term": "backpropagation"
 },
 "backpropagation through time": {
  "def": "Backpropagation applied to an RNN unrolled across its time steps.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "backpropagation through time"
 },
 "bag of words": {
  "def": "Representing text by counting how many times each word appears, ignoring order.",
  "lesson": "primer/ml/embeddings/contrastive.html",
  "term": "bag of words"
 },
 "bagging": {
  "def": "Bootstrap aggregating: train one model per bootstrap sample and average them to cut variance.",
  "lesson": "primer/ml/classical.html",
  "term": "bagging"
 },
 "base model": {
  "def": "A model after pretraining only: knowledgeable, but it continues text rather than following instructions.",
  "lesson": "primer/ml/training_stages.html",
  "term": "base model"
 },
 "baseline": {
  "def": "A typical reward subtracted before updating; it reduces noise without changing the average gradient.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "baseline"
 },
 "batch": {
  "def": "The group of examples processed before one weight update.",
  "lesson": "primer/ml/neural_net.html",
  "term": "batch"
 },
 "batch api": {
  "def": "A provider interface that processes many requests asynchronously at a discount.",
  "lesson": "primer/agents/cost.html",
  "term": "batch API"
 },
 "batch normalization": {
  "def": "Rescaling each feature using statistics from the current batch of examples. Common in image models.",
  "lesson": "primer/ml/deep_nets.html",
  "term": "batch normalization"
 },
 "beam search": {
  "def": "Decoding that keeps the few most probable partial outputs at each step instead of just one, then picks the best finished one.",
  "lesson": "primer/ml/inference.html",
  "term": "beam search"
 },
 "benchmark": {
  "def": "Fixed questions plus a scoring rule, averaged into one comparable number.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "benchmark"
 },
 "benchmark contamination": {
  "def": "When test questions appeared in a model's training data, inflating its scores.",
  "lesson": "primer/ml/regularization.html",
  "term": "benchmark contamination"
 },
 "bertscore": {
  "def": "Compares generated and reference text by embedding similarity instead of exact words.",
  "lesson": "primer/ml/metrics.html",
  "term": "BERTScore"
 },
 "best-of-n": {
  "def": "Sample n candidates and keep the one a verifier scores highest.",
  "lesson": "primer/ml/reasoning.html",
  "term": "best-of-n"
 },
 "bf16": {
  "def": "A 16-bit float with fp32's 8 exponent bits and 7 mantissa bits: the same range, less precision.",
  "lesson": "primer/ml/hardware.html",
  "term": "bf16"
 },
 "bi-encoder": {
  "def": "Embeds the query and each document separately, so document vectors can be computed once and searched fast.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "bi-encoder"
 },
 "bias": {
  "def": "A learned number added to a neuron's weighted sum, letting it shift its output up or down.",
  "lesson": "primer/ml/neural_net.html",
  "term": "bias"
 },
 "bias correction": {
  "def": "Adam's rescaling of its running averages early in training, since they start at zero and would otherwise be too small.",
  "lesson": "primer/ml/optimizers.html",
  "term": "bias correction"
 },
 "bias-variance trade-off": {
  "def": "Splitting a model's error into systematic error from being too simple (bias) and sensitivity to the training sample from being too flexible (variance).",
  "lesson": "primer/ml/regularization.html",
  "term": "bias-variance trade-off"
 },
 "big-o": {
  "def": "A way to say how cost grows with input size, ignoring constant factors. O(n²) means doubling n quadruples the cost.",
  "lesson": "primer/notation.html",
  "term": "big-O"
 },
 "binary cross-entropy": {
  "def": "Cross-entropy for yes/no predictions: −[y·ln p + (1 − y)·ln(1 − p)].",
  "lesson": "primer/ml/losses.html",
  "term": "binary cross-entropy"
 },
 "binary quantization": {
  "def": "Storing only the sign of each number, as a single bit.",
  "lesson": "primer/ml/embeddings/compression.html",
  "term": "binary quantization"
 },
 "blast radius": {
  "def": "How much damage a failure can do before it is noticed and stopped.",
  "lesson": "primer/agents/deployment.html",
  "term": "blast radius"
 },
 "bleu": {
  "def": "A score that counts overlapping word sequences with a reference text. Cheap, but blind to paraphrase.",
  "lesson": "primer/ml/metrics.html",
  "term": "BLEU"
 },
 "block table": {
  "def": "A per-request map from logical KV-cache blocks to physical blocks in GPU memory.",
  "lesson": "primer/ml/inference.html",
  "term": "block table"
 },
 "blue/green deployment": {
  "def": "Building the new version beside the old one, switching traffic once it proves itself, and keeping the old one for rollback.",
  "lesson": "primer/ml/embeddings/operations.html",
  "term": "blue/green deployment"
 },
 "bm25": {
  "def": "The classic keyword-search score: rewards documents containing the query's rarer words, with diminishing returns for repeats.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "BM25"
 },
 "bootstrap": {
  "def": "Measuring uncertainty by rescoring many with-replacement resamples of your own data.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "bootstrap"
 },
 "bootstrap sample": {
  "def": "A resample of the training rows drawn with replacement, so some rows repeat and about 37% are left out.",
  "lesson": "primer/ml/classical.html",
  "term": "bootstrap sample"
 },
 "bottleneck": {
  "def": "The narrow middle of an autoencoder: too small to copy through, it forces the network to keep only what matters.",
  "lesson": "primer/ml/generative/autoencoders.html",
  "term": "bottleneck"
 },
 "bpe": {
  "def": "Byte pair encoding: build a vocabulary by repeatedly merging the most frequent adjacent pair of symbols.",
  "lesson": "primer/ml/tokenization.html",
  "term": "BPE"
 },
 "bradley-terry model": {
  "def": "The rule that the probability A beats B is the sigmoid of their score difference.",
  "lesson": "primer/ml/training_stages.html",
  "term": "Bradley-Terry model"
 },
 "byte pair encoding": {
  "def": "A tokenizer-building method that starts from characters and repeatedly merges the most frequent adjacent pair into a new token.",
  "lesson": "primer/ml/tokenization.html",
  "term": "byte pair encoding"
 },
 "byte-level bpe": {
  "def": "Byte pair encoding run on the raw bytes of UTF-8 text, so any string can be tokenized and there is never an unknown token.",
  "lesson": "primer/ml/tokenization.html",
  "term": "byte-level BPE"
 },
 "canary release": {
  "def": "Sending a small share of traffic to a new version first and watching its metrics before rolling out further.",
  "lesson": "primer/agents/deployment.html",
  "term": "canary release"
 },
 "canary string": {
  "def": "A unique marker in benchmark files so trainers can filter out copies.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "canary string"
 },
 "capability negotiation": {
  "def": "The start-of-session exchange where client and server announce which optional features they support.",
  "lesson": "primer/agents/mcp.html",
  "term": "capability negotiation"
 },
 "catastrophic forgetting": {
  "def": "When training on a new task alone erodes or erases skills a model already had.",
  "lesson": "primer/ml/fine_tuning.html",
  "term": "catastrophic forgetting"
 },
 "causal mask": {
  "def": "Hides future tokens from each token during attention, so a model predicting the next word can't peek at it.",
  "lesson": "primer/ml/attention.html",
  "term": "causal mask"
 },
 "cbow": {
  "def": "Continuous bag of words: the word2vec model that averages the surrounding words' vectors to predict the middle word.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "cbow"
 },
 "cell state": {
  "def": "An LSTM's notebook, edited by adding rather than overwriting, so information survives many steps.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "cell state"
 },
 "centroid": {
  "def": "The average position of a group of vectors, used as the group's representative point.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "centroid"
 },
 "chain of thought": {
  "def": "Intermediate reasoning steps a model writes before its final answer, each one readable by the next forward pass.",
  "lesson": "primer/ml/reasoning.html",
  "term": "chain of thought"
 },
 "chain rule": {
  "def": "To get the slope through a chain of functions, multiply the slopes of each link.",
  "lesson": "primer/notation.html",
  "term": "chain rule"
 },
 "chat template": {
  "def": "The exact role markers and layout a model family uses to turn messages into training text.",
  "lesson": "primer/ml/fine_tuning.html",
  "term": "chat template"
 },
 "checkpoint": {
  "def": "Saved progress after a completed step, so a failure resumes from there instead of from the start.",
  "lesson": "primer/agents/planning.html",
  "term": "checkpoint"
 },
 "checkpoint averaging": {
  "def": "Averaging the weights saved at the last few points of training, which usually gives a slightly better model.",
  "lesson": "primer/ml/training_stages.html",
  "term": "checkpoint averaging"
 },
 "chinchilla scaling": {
  "def": "The compute-optimal rule of about 20 training tokens per model parameter.",
  "lesson": "primer/ml/pretraining.html",
  "term": "chinchilla scaling"
 },
 "chunking": {
  "def": "Splitting documents into passages before embedding them, so retrieval can return just the relevant part.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "chunking"
 },
 "circuit": {
  "def": "A chain of components that together compute one behaviour.",
  "lesson": "primer/ml/interpretability.html",
  "term": "circuit"
 },
 "circuit breaker": {
  "def": "A wrapper that stops calling a failing dependency for a cool-down period, failing fast instead.",
  "lesson": "primer/agents/failures.html",
  "term": "circuit breaker"
 },
 "classifier-free guidance": {
  "def": "Mixing a model's guesses with and without the prompt, and pushing past the prompted one to follow it more closely.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "classifier-free guidance"
 },
 "clip": {
  "def": "A model that trains an image encoder and a text encoder together so pictures and their captions land near each other in one vector space.",
  "lesson": "primer/ml/embeddings/contrastive.html",
  "term": "clip"
 },
 "clustering": {
  "def": "Grouping items so similar ones end up together, without being told the groups.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "clustering"
 },
 "co-adaptation": {
  "def": "When neurons learn to depend on each other's exact behaviour, each correcting the others' quirks; it fits the training data but breaks on new data.",
  "lesson": "primer/ml/regularization.html",
  "term": "co-adaptation"
 },
 "code grader": {
  "def": "An automatic check written in code, such as exact match, schema or database state, that scores an agent's output.",
  "lesson": "primer/agents/evals.html",
  "term": "code grader"
 },
 "codebook": {
  "def": "A small catalogue of representative vectors; each vector (or piece of one) is stored as the number of its nearest entry, as in product quantization or tokenizers for images and audio.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "codebook"
 },
 "cohen's kappa": {
  "def": "Agreement between two raters after subtracting the agreement expected by chance.",
  "lesson": "primer/ml/metrics.html",
  "term": "Cohen's kappa"
 },
 "colbert": {
  "def": "A retrieval model that keeps one vector per token and matches each query token to its best document token.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "ColBERT"
 },
 "collector": {
  "def": "In OpenTelemetry, the service that receives spans from apps and forwards them to storage backends.",
  "lesson": "primer/agents/observability.html",
  "term": "collector"
 },
 "compounding error": {
  "def": "Small per-step failure rates multiplying over many steps: ten steps at 95% succeed only about 60% of the time.",
  "lesson": "primer/agents/planning.html",
  "term": "compounding error"
 },
 "compute-bound": {
  "def": "Limited by arithmetic throughput, as in prefill.",
  "lesson": "primer/ml/inference.html",
  "term": "compute-bound"
 },
 "computer use": {
  "def": "An agent operating a graphical interface from screenshots, sending clicks and keystrokes.",
  "lesson": "primer/agents/coding_agents.html",
  "term": "computer use"
 },
 "confidence interval": {
  "def": "A range built so that 95 in 100 such ranges contain the true value.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "confidence interval"
 },
 "confused deputy": {
  "def": "A program with broad authority tricked into using it for someone who lacks that authority.",
  "lesson": "primer/agents/mcp.html",
  "term": "confused deputy"
 },
 "confusion matrix": {
  "def": "The four counts behind every classification metric: hits, false alarms, misses and correct passes.",
  "lesson": "primer/ml/metrics.html",
  "term": "confusion matrix"
 },
 "constitutional ai": {
  "def": "Training against a written list of principles: the model critiques and revises its answers, and AI-labelled preferences train the reward model.",
  "lesson": "primer/ml/alignment.html",
  "term": "Constitutional AI"
 },
 "constrained decoding": {
  "def": "Before each token is drawn, setting every token that can't lead to a valid answer to probability zero.",
  "lesson": "primer/ml/structured_output.html",
  "term": "constrained decoding"
 },
 "contamination": {
  "def": "Test questions and answers leaking into a model's training data.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "contamination"
 },
 "content-addressed": {
  "def": "Named by a hash of its own content, so identical content always gets the identical name.",
  "lesson": "primer/agents/deployment.html",
  "term": "content-addressed"
 },
 "context engineering": {
  "def": "Deciding exactly what goes into the model's context window on each call: instructions, facts, tool results, memory.",
  "lesson": "primer/agents/context.html",
  "term": "context engineering"
 },
 "context rot": {
  "def": "Quality dropping as a long session fills the context with stale or irrelevant material.",
  "lesson": "primer/agents/context.html",
  "term": "context rot"
 },
 "context window": {
  "def": "The maximum number of tokens a model can take in at once.",
  "lesson": "primer/ml/inference.html",
  "term": "context window"
 },
 "contextual embedding": {
  "def": "A vector computed for a word within its sentence, so the same word gets different vectors in different contexts.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "contextual embedding"
 },
 "contextual retrieval": {
  "def": "Indexing each chunk together with a line saying where in its document it comes from.",
  "lesson": "primer/agents/rag.html",
  "term": "contextual retrieval"
 },
 "continuous batching": {
  "def": "Refilling a finished request's slot in the batch immediately with the next request.",
  "lesson": "primer/ml/inference.html",
  "term": "continuous batching"
 },
 "continuous integration": {
  "def": "The automated checks that run on every proposed change before it can merge.",
  "lesson": "primer/agents/evals.html",
  "term": "continuous integration"
 },
 "contract test": {
  "def": "An automated check that an external API still accepts and returns what your code expects.",
  "lesson": "primer/agents/failures.html",
  "term": "contract test"
 },
 "contrastive learning": {
  "def": "Training that pulls matching pairs together and pushes non-matching pairs apart.",
  "lesson": "primer/ml/embeddings/contrastive.html",
  "term": "contrastive learning"
 },
 "contrastive loss": {
  "def": "A loss that pulls matching pairs together in vector space and pushes non-matching pairs apart.",
  "lesson": "primer/ml/losses.html",
  "term": "contrastive loss"
 },
 "control task": {
  "def": "A probe trained on random labels, showing how much a probe fits with nothing real to find.",
  "lesson": "primer/ml/interpretability.html",
  "term": "control task"
 },
 "convex": {
  "def": "Bowl-shaped everywhere with a single bottom, so any downhill path reaches the same minimum; neural network losses are not convex.",
  "lesson": "primer/notation.html",
  "term": "convex"
 },
 "convolution": {
  "def": "Sliding a small grid of learned weights across an image and computing a weighted sum at every position, to find where a pattern appears.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "convolution"
 },
 "cooldown": {
  "def": "A quiet period after an action during which the same action is not repeated.",
  "lesson": null,
  "term": "cooldown"
 },
 "copy-on-write": {
  "def": "Sharing one copy of data and making a private copy only when someone needs to change it.",
  "lesson": "primer/ml/inference.html",
  "term": "copy-on-write"
 },
 "cosine decay": {
  "def": "Lowering the learning rate from its peak to a floor along half a cosine wave.",
  "lesson": "primer/ml/optimizers.html",
  "term": "cosine decay"
 },
 "cosine similarity": {
  "def": "How closely two vectors point in the same direction, from −1 (opposite) to 1 (identical), ignoring their lengths.",
  "lesson": "primer/ml/embeddings/similarity.html",
  "term": "cosine similarity"
 },
 "cost per resolved task": {
  "def": "Everything spent on all attempts, divided by the tasks actually resolved.",
  "lesson": "primer/agents/coding_agents.html",
  "term": "cost per resolved task"
 },
 "cost per successful task": {
  "def": "Total spend divided by the number of tasks that succeeded, counting retries and cleanup.",
  "lesson": "primer/agents/cost.html",
  "term": "cost per successful task"
 },
 "credit assignment": {
  "def": "Working out which of many earlier actions deserves the blame or credit for how an episode ended.",
  "lesson": "primer/agents/planning.html",
  "term": "credit assignment"
 },
 "critic": {
  "def": "A Wasserstein GAN's discriminator, which outputs an unbounded score instead of a probability.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "critic"
 },
 "cross-attention": {
  "def": "Attention whose queries come from one sequence (image patches) and whose keys and values come from another (text tokens).",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "cross-attention"
 },
 "cross-encoder": {
  "def": "Reads the query and one document together and outputs a relevance score. Accurate but slow, so it is used to rerank a shortlist.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "cross-encoder"
 },
 "cross-entropy": {
  "def": "The standard loss for predicting categories: minus the log of the probability given to the right answer.",
  "lesson": "primer/ml/losses.html",
  "term": "cross-entropy"
 },
 "cross-validation": {
  "def": "Rotating which slice of the data is held out, training once per slice, and averaging the scores.",
  "lesson": "primer/ml/regularization.html",
  "term": "cross-validation"
 },
 "curse of dimensionality": {
  "def": "In very high dimensions, random points are all nearly the same distance apart, which makes exact nearest-neighbor search slow and fragile.",
  "lesson": "primer/ml/embeddings/similarity.html",
  "term": "curse of dimensionality"
 },
 "dark knowledge": {
  "def": "How a teacher model ranks the wrong answers, revealed by its soft targets.",
  "lesson": "primer/ml/training_stages.html",
  "term": "dark knowledge"
 },
 "data leakage": {
  "def": "When information from the test set, or from the future, sneaks into training and inflates offline scores.",
  "lesson": "primer/ml/regularization.html",
  "term": "data leakage"
 },
 "data mixture": {
  "def": "The share of training tokens drawn from each data source.",
  "lesson": "primer/ml/pretraining.html",
  "term": "data mixture"
 },
 "data parallelism": {
  "def": "Every GPU holds the whole model and trains on a different slice of the batch.",
  "lesson": "primer/ml/pretraining.html",
  "term": "data parallelism"
 },
 "dbscan": {
  "def": "Density-based clustering that grows clusters from points with enough close neighbours and labels the rest as noise.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "DBSCAN"
 },
 "ddim": {
  "def": "A deterministic diffusion sampler that predicts the clean result and jumps to a much less noisy step, so it needs far fewer steps.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "DDIM"
 },
 "ddpm": {
  "def": "The original diffusion sampler: remove the guessed noise in many small steps, adding a small fresh wobble each time.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "DDPM"
 },
 "debounce": {
  "def": "Waiting until a signal has settled before acting on it, so a burst of changes triggers one action.",
  "lesson": null,
  "term": "debounce"
 },
 "decision tree": {
  "def": "A flowchart of yes/no questions about one feature at a time, learned from examples, ending in leaves that give the answer.",
  "lesson": "primer/ml/classical.html",
  "term": "decision tree"
 },
 "decode": {
  "def": "The second phase of generation: output tokens are produced one at a time. It sets the tokens per second.",
  "lesson": "primer/ml/inference.html",
  "term": "decode"
 },
 "decoder": {
  "def": "A transformer that generates text left to right, each token seeing only the ones before it. GPT, Claude and Llama are decoders.",
  "lesson": "primer/ml/transformer.html",
  "term": "decoder"
 },
 "decomposition": {
  "def": "Splitting a big goal into small subtasks, each with a checkable definition of done.",
  "lesson": "primer/agents/planning.html",
  "term": "decomposition"
 },
 "deduplication": {
  "def": "Removing repeated copies of documents from training data.",
  "lesson": "primer/ml/pretraining.html",
  "term": "deduplication"
 },
 "degradation problem": {
  "def": "Deeper plain networks reaching higher error even on their training data: an optimization failure, not overfitting.",
  "lesson": "primer/ml/deep_nets.html",
  "term": "degradation problem"
 },
 "denoiser": {
  "def": "The network in a diffusion model that looks at a noisy input and its step, and guesses the noise in it.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "denoiser"
 },
 "denoising score matching": {
  "def": "Learning the score by training a network to guess the noise added to real examples.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "denoising score matching"
 },
 "dense retrieval": {
  "def": "Search by comparing embedding vectors, so matches are by meaning rather than exact words.",
  "lesson": "primer/agents/rag.html",
  "term": "dense retrieval"
 },
 "derivative": {
  "def": "The slope of a function at a point: how much the output changes per tiny nudge of the input.",
  "lesson": "primer/notation.html",
  "term": "derivative"
 },
 "diffusion model": {
  "def": "A generator that learns to turn random noise into data by removing a little noise at a time.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "diffusion model"
 },
 "diffusion transformer": {
  "def": "A transformer used as the denoiser, reading patches of the latent as tokens (DiT).",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "diffusion transformer"
 },
 "dimension": {
  "def": "One of the numbers in a vector: a 768-dimension embedding is a list of 768 numbers.",
  "lesson": "primer/ml/embeddings/similarity.html",
  "term": "dimension"
 },
 "discretization": {
  "def": "Turning a continuous rate of change into one step's keep and write factors.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "discretization"
 },
 "discriminator": {
  "def": "The network in a GAN that outputs the probability that a sample is real rather than generated.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "discriminator"
 },
 "distillation": {
  "def": "Training a small model to imitate a large one's outputs, often the biggest cost saving in production.",
  "lesson": "primer/ml/training_stages.html",
  "term": "distillation"
 },
 "diversity heuristic": {
  "def": "HNSW's rule of keeping a neighbour only if it is closer to the new node than to any neighbour already kept, so links fan out in different directions.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "diversity heuristic"
 },
 "domain mismatch": {
  "def": "A general model underperforming on specialised language it never saw, such as company jargon.",
  "lesson": "primer/ml/embeddings/operations.html",
  "term": "domain mismatch"
 },
 "dot product": {
  "def": "Multiply two lists position by position and add up the products. It is large when two vectors point the same way.",
  "lesson": "primer/notation.html",
  "term": "dot product"
 },
 "double descent": {
  "def": "The observation that past a point, even larger models generalize better again.",
  "lesson": "primer/ml/regularization.html",
  "term": "double descent"
 },
 "double quantization": {
  "def": "Compressing the per-block scale factors of a quantized model as well, saving about 0.37 bits per parameter.",
  "lesson": "primer/ml/inference.html",
  "term": "double quantization"
 },
 "dpo": {
  "def": "Direct preference optimization: tuning a model straight from pairs of preferred and rejected answers, without a separate reward model.",
  "lesson": "primer/ml/training_stages.html",
  "term": "DPO"
 },
 "draft model": {
  "def": "The small, fast model that proposes tokens in speculative decoding.",
  "lesson": "primer/ml/inference.html",
  "term": "draft model"
 },
 "drift": {
  "def": "A metric creeping away from its usual level over time or after a deploy.",
  "lesson": "primer/agents/evals.html",
  "term": "drift"
 },
 "dropout": {
  "def": "Randomly switching off neurons during training so the network can't rely on any single path.",
  "lesson": "primer/ml/regularization.html",
  "term": "dropout"
 },
 "dry run": {
  "def": "Running every check for an action and describing it without actually doing it.",
  "lesson": "primer/agents/tools.html",
  "term": "dry run"
 },
 "dual write": {
  "def": "Writing every new record to both the old and the new system during a migration.",
  "lesson": "primer/ml/embeddings/operations.html",
  "term": "dual write"
 },
 "durable execution": {
  "def": "Running a workflow so its progress survives crashes, by storing each completed step's result.",
  "lesson": "primer/agents/orchestration.html",
  "term": "durable execution"
 },
 "dying relu": {
  "def": "A ReLU neuron whose input stays negative, so it outputs zero and receives zero gradient forever.",
  "lesson": "primer/ml/neural_net.html",
  "term": "dying relu"
 },
 "dynamic tool loading": {
  "def": "Sending the model only the few tool definitions most relevant to the current request.",
  "lesson": "primer/agents/tools.html",
  "term": "dynamic tool loading"
 },
 "early stopping": {
  "def": "Stopping training when the score on held-out data stops improving.",
  "lesson": "primer/ml/regularization.html",
  "term": "early stopping"
 },
 "edit-run-test loop": {
  "def": "An agent loop that changes code, runs the tests, and repeats until they pass or a budget runs out.",
  "lesson": "primer/agents/coding_agents.html",
  "term": "edit-run-test loop"
 },
 "efconstruction": {
  "def": "HNSW's beam width while building the graph; higher means a better graph and a slower build.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "efConstruction"
 },
 "efsearch": {
  "def": "HNSW's query-time beam width; raising it trades speed for recall.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "efsearch"
 },
 "elbow method": {
  "def": "Choosing the number of clusters where adding more stops reducing inertia much.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "elbow method"
 },
 "elo rating": {
  "def": "A rating scale where a 400-point gap means ten-to-one odds of winning.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "Elo rating"
 },
 "embedding": {
  "def": "A learned vector for a piece of content, arranged so that similar meanings end up close together.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "embedding"
 },
 "encoder": {
  "def": "A transformer that reads the whole input at once, with every token seeing every other. Used for embeddings and classification.",
  "lesson": "primer/ml/transformer.html",
  "term": "encoder"
 },
 "encoder-decoder": {
  "def": "A transformer where an encoder reads the whole input and a decoder writes the output one token at a time while attending to the encoder.",
  "lesson": "primer/ml/transformer.html",
  "term": "encoder-decoder"
 },
 "encoder-decoder attention": {
  "def": "Attention where the decoder's queries look at the encoder's keys and values, so each output word can consult the whole input.",
  "lesson": "primer/ml/transformer.html",
  "term": "encoder-decoder attention"
 },
 "ensemble": {
  "def": "A model that combines the predictions of many models, such as trees.",
  "lesson": "primer/ml/classical.html",
  "term": "ensemble"
 },
 "entailment": {
  "def": "Whether one text logically follows from another, checked by a model trained for it.",
  "lesson": "primer/agents/guardrails.html",
  "term": "entailment"
 },
 "entropy": {
  "def": "The average surprise of a distribution: how many yes/no questions, on average, it takes to learn an outcome; 0 means certain.",
  "lesson": "primer/ml/classical.html",
  "term": "entropy"
 },
 "episodic memory": {
  "def": "Memory of what happened, such as 'last week the user rejected this vendor'.",
  "lesson": "primer/agents/memory.html",
  "term": "episodic memory"
 },
 "epoch": {
  "def": "One full pass over the training data.",
  "lesson": "primer/ml/neural_net.html",
  "term": "epoch"
 },
 "erp": {
  "def": "Enterprise resource planning: a company's finance and operations system of record.",
  "lesson": "primer/agents/deployment.html",
  "term": "erp"
 },
 "escaping": {
  "def": "Replacing characters like < and > so data can't pose as markup or instructions.",
  "lesson": "primer/agents/context.html",
  "term": "escaping"
 },
 "euclidean distance": {
  "def": "The straight-line distance between two vectors.",
  "lesson": "primer/ml/embeddings/similarity.html",
  "term": "Euclidean distance"
 },
 "euler step": {
  "def": "Moving in a straight line for a short time at the current velocity, then looking again.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "Euler step"
 },
 "eval": {
  "def": "A repeatable test of a model or agent's behavior on a fixed set of tasks with known good outcomes.",
  "lesson": "primer/agents/evals.html",
  "term": "eval"
 },
 "evaluator-optimizer": {
  "def": "A generator drafts and an evaluator critiques, repeated until the draft passes or the rounds run out.",
  "lesson": "primer/agents/orchestration.html",
  "term": "evaluator-optimizer"
 },
 "evidence lower bound": {
  "def": "A quantity that never exceeds the log-probability a model gives the data; the VAE loss is its negative.",
  "lesson": "primer/ml/generative/autoencoders.html",
  "term": "evidence lower bound"
 },
 "expectation": {
  "def": "The average of a quantity over many random draws, written E.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "expectation"
 },
 "expected value": {
  "def": "The average you would get over many tries.",
  "lesson": "primer/notation.html",
  "term": "expected value"
 },
 "exploding gradient": {
  "def": "When gradients grow huge as they pass back through many layers, so training diverges.",
  "lesson": "primer/ml/deep_nets.html",
  "term": "exploding gradient"
 },
 "exploration": {
  "def": "Trying actions you are unsure of instead of repeating the best one so far.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "exploration"
 },
 "exponent": {
  "def": "The bits of a float that pick the power of two, which sets its range.",
  "lesson": "primer/ml/hardware.html",
  "term": "exponent"
 },
 "exponential backoff": {
  "def": "Waiting longer after each failed attempt, doubling up to a cap.",
  "lesson": "primer/agents/failures.html",
  "term": "exponential backoff"
 },
 "external fragmentation": {
  "def": "Free memory split into gaps too small to use.",
  "lesson": "primer/ml/inference.html",
  "term": "external fragmentation"
 },
 "external verification": {
  "def": "Checking a model's output with something outside the model, such as tests, schemas or database queries.",
  "lesson": "primer/agents/planning.html",
  "term": "external verification"
 },
 "f1": {
  "def": "The harmonic mean of precision and recall: high only when both are high.",
  "lesson": "primer/ml/metrics.html",
  "term": "F1"
 },
 "fail-to-pass test": {
  "def": "A hidden test that fails before a fix and must pass after it.",
  "lesson": "primer/agents/coding_agents.html",
  "term": "fail-to-pass test"
 },
 "faithfulness": {
  "def": "The share of an answer's claims that are supported by the retrieved sources.",
  "lesson": "primer/agents/evals.html",
  "term": "faithfulness"
 },
 "false-positive rate": {
  "def": "The share of real negatives the model wrongly flags.",
  "lesson": "primer/ml/metrics.html",
  "term": "false-positive rate"
 },
 "fan-in": {
  "def": "The number of inputs a neuron adds up.",
  "lesson": "primer/ml/deep_nets.html",
  "term": "fan-in"
 },
 "feature": {
  "def": "A property a model tracks, stored as a direction across many neurons.",
  "lesson": "primer/ml/interpretability.html",
  "term": "feature"
 },
 "feature engineering": {
  "def": "Hand-transforming inputs (logarithms, one-hot columns) so a model can use them.",
  "lesson": "primer/ml/classical.html",
  "term": "feature engineering"
 },
 "feature flag": {
  "def": "A configuration switch that turns a capability on or off without redeploying.",
  "lesson": "primer/agents/deployment.html",
  "term": "feature flag"
 },
 "feature importance": {
  "def": "A score for how much a model relies on each input column.",
  "lesson": "primer/ml/classical.html",
  "term": "feature importance"
 },
 "feature map": {
  "def": "In linear attention, the function φ applied to queries and keys so that φ(q)·φ(k) replaces e^(q·k).",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "feature map"
 },
 "feed-forward network": {
  "def": "The small two-layer network in each transformer block that processes every token on its own after attention has mixed them.",
  "lesson": "primer/ml/transformer.html",
  "term": "feed-forward network"
 },
 "few-shot prompt": {
  "def": "A prompt that includes a few worked examples of the task before the real question.",
  "lesson": "primer/agents/context.html",
  "term": "few-shot prompt"
 },
 "filter": {
  "def": "In a CNN, the small grid of learned weights a convolution slides over an image; also called a kernel.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "filter"
 },
 "fine-tuning": {
  "def": "Training an existing model further on a smaller, targeted dataset to change its behavior.",
  "lesson": "primer/ml/training_stages.html",
  "term": "fine-tuning"
 },
 "finite-state machine": {
  "def": "A fixed set of states with one move per input character; it can check patterns but not unlimited nesting.",
  "lesson": "primer/ml/structured_output.html",
  "term": "finite-state machine"
 },
 "flat index": {
  "def": "Exact search that compares the query with every stored vector; the ground truth other indexes are measured against.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "flat index"
 },
 "flip rate": {
  "def": "The share of questions whose answer changes when the user asserts a wrong answer.",
  "lesson": "primer/ml/alignment.html",
  "term": "flip rate"
 },
 "floating point": {
  "def": "Storing a number as a sign, an exponent (range) and a mantissa (precision).",
  "lesson": "primer/ml/hardware.html",
  "term": "floating point"
 },
 "flops": {
  "def": "Floating-point operations: individual multiplies or adds, used to measure compute (about 2 per parameter per token to generate, 6 to train).",
  "lesson": "primer/ml/transformer.html",
  "term": "FLOPs"
 },
 "flow matching": {
  "def": "Training a network to output the velocity along paths from noise to data, then following it to generate.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "flow matching"
 },
 "forget gate": {
  "def": "The LSTM dial that decides what to erase from the cell state.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "forget gate"
 },
 "forward pass": {
  "def": "Running inputs through the network, layer by layer, to get a prediction.",
  "lesson": "primer/ml/neural_net.html",
  "term": "forward pass"
 },
 "forward process": {
  "def": "The fixed, unlearned procedure that mixes data with noise step by step until only noise is left.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "forward process"
 },
 "fourier transform": {
  "def": "Splitting a signal into how much of each frequency it contains.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "Fourier transform"
 },
 "fp16": {
  "def": "A 16-bit float with 5 exponent and 10 mantissa bits: more precision, range only up to 65,504.",
  "lesson": "primer/ml/hardware.html",
  "term": "fp16"
 },
 "fp8": {
  "def": "8-bit floats: E4M3 (range to 448) for weights and activations, E5M2 (to 57,344) for gradients.",
  "lesson": "primer/ml/hardware.html",
  "term": "fp8"
 },
 "frame sampling": {
  "def": "Keeping only some video frames, such as one a second, to save tokens.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "frame sampling"
 },
 "frobenius norm": {
  "def": "The size of a whole matrix: square every entry, add them up, take the square root.",
  "lesson": "primer/notation.html",
  "term": "frobenius norm"
 },
 "fsdp": {
  "def": "Fully sharded data parallel: all training state sharded across GPUs (ZeRO stage 3).",
  "lesson": "primer/ml/pretraining.html",
  "term": "FSDP"
 },
 "full fine-tune": {
  "def": "Updating every weight of a model. Rarely worth the cost.",
  "lesson": "primer/ml/training_stages.html",
  "term": "full fine-tune"
 },
 "fused multiply-add": {
  "def": "One instruction that multiplies two numbers and adds the result to a running total.",
  "lesson": "primer/ml/hardware.html",
  "term": "fused multiply-add"
 },
 "gan": {
  "def": "A generator and a discriminator trained against each other, so the generator learns to make samples the discriminator can't tell from real data.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "GAN"
 },
 "gelu": {
  "def": "A smooth version of ReLU used in most transformers.",
  "lesson": "primer/ml/neural_net.html",
  "term": "GELU"
 },
 "genai semantic conventions": {
  "def": "OpenTelemetry's standard attribute names for model and agent spans, such as gen_ai.request.model.",
  "lesson": "primer/agents/observability.html",
  "term": "genai semantic conventions"
 },
 "generalization gap": {
  "def": "Validation loss minus training loss; when it keeps growing, the model is memorising.",
  "lesson": "primer/ml/fine_tuning.html",
  "term": "generalization gap"
 },
 "generation failure": {
  "def": "A wrong answer produced even though a relevant document was retrieved.",
  "lesson": "primer/ml/embeddings/operations.html",
  "term": "generation failure"
 },
 "generator": {
  "def": "The network in a GAN that turns random noise into a sample.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "generator"
 },
 "gini impurity": {
  "def": "The chance that two examples drawn at random from a pile carry different labels; 0 means pure.",
  "lesson": "primer/ml/classical.html",
  "term": "Gini impurity"
 },
 "global token": {
  "def": "A token every other token may attend to, and which attends to all, so any two tokens are two hops apart.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "global token"
 },
 "glove": {
  "def": "A counting method that fits word vectors so their dot products predict how often words appear together.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "GloVe"
 },
 "golden set": {
  "def": "A curated set of real tasks with expected outcomes, run on every change to catch regressions.",
  "lesson": "primer/agents/evals.html",
  "term": "golden set"
 },
 "goodhart's law": {
  "def": "Once a measure becomes a target, optimizing it stops improving the thing it measured.",
  "lesson": "primer/ml/alignment.html",
  "term": "Goodhart's law"
 },
 "gpu": {
  "def": "A graphics processor: a chip with thousands of small cores that do many multiply-adds at once, which is exactly what neural networks need.",
  "lesson": "primer/notation.html",
  "term": "GPU"
 },
 "gradient": {
  "def": "One slope per input, collected into a vector. It points in the direction that increases the function fastest.",
  "lesson": "primer/notation.html",
  "term": "gradient"
 },
 "gradient boosting": {
  "def": "Adding small trees one at a time, each fit to the current errors, with each step shrunk by a learning rate.",
  "lesson": "primer/ml/classical.html",
  "term": "gradient boosting"
 },
 "gradient clipping": {
  "def": "Capping the size of the gradient so one bad batch can't throw the weights far off course.",
  "lesson": "primer/ml/deep_nets.html",
  "term": "gradient clipping"
 },
 "gradient descent": {
  "def": "Training by repeatedly taking a small step downhill: nudge every weight against its gradient to lower the loss.",
  "lesson": "primer/ml/optimizers.html",
  "term": "gradient descent"
 },
 "gradient penalty": {
  "def": "A discriminator loss term that punishes steep slopes of its output with respect to its input (R1, WGAN-GP).",
  "lesson": "primer/ml/generative/gans.html",
  "term": "gradient penalty"
 },
 "graduated autonomy": {
  "def": "Granting an agent more independence in steps (shadow, then approval, then autonomy), each earned with evidence.",
  "lesson": "primer/agents/deployment.html",
  "term": "graduated autonomy"
 },
 "grammar": {
  "def": "A set of rules defining which strings are valid in a format.",
  "lesson": "primer/ml/structured_output.html",
  "term": "grammar"
 },
 "graph": {
  "def": "A set of points (nodes) joined by links (edges).",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "graph"
 },
 "graphrag": {
  "def": "Retrieval over a graph of entities and relationships extracted from documents.",
  "lesson": "primer/agents/rag.html",
  "term": "GraphRAG"
 },
 "greedy decoding": {
  "def": "Always picking the single most probable next token, the same as sampling at temperature 0.",
  "lesson": "primer/ml/big_picture.html",
  "term": "greedy decoding"
 },
 "greedy search": {
  "def": "Always move to whichever neighbour is closest to the target, and stop when none is closer.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "greedy search"
 },
 "grounding": {
  "def": "Tying an answer to specific source passages so every claim can be checked.",
  "lesson": "primer/agents/rag.html",
  "term": "grounding"
 },
 "grouped-query attention": {
  "def": "Several query heads share one set of keys and values, shrinking the memory needed during generation.",
  "lesson": "primer/ml/attention.html",
  "term": "grouped-query attention"
 },
 "grpo": {
  "def": "Group Relative Policy Optimization: each answer is scored against other answers to the same prompt, so no value network is needed.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "GRPO"
 },
 "gru": {
  "def": "A simpler gated RNN with an update gate and a reset gate and no separate cell state.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "gru"
 },
 "guardrail": {
  "def": "A check around the model that screens inputs, validates outputs or limits actions.",
  "lesson": "primer/agents/guardrails.html",
  "term": "guardrail"
 },
 "guidance scale": {
  "def": "The weight w in classifier-free guidance: 0 ignores the prompt, 1 follows it plainly, above 1 exaggerates it.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "guidance scale"
 },
 "hallucination": {
  "def": "A confident answer that isn't supported by the sources or the facts.",
  "lesson": "primer/agents/rag.html",
  "term": "hallucination"
 },
 "hamming distance": {
  "def": "The number of positions where two bit strings differ.",
  "lesson": "primer/ml/embeddings/compression.html",
  "term": "hamming distance"
 },
 "hand-off": {
  "def": "The message that passes work and its context from one agent to another; anything not written into it is lost.",
  "lesson": "primer/agents/orchestration.html",
  "term": "hand-off"
 },
 "hann window": {
  "def": "A smooth rise and fall applied to each slice so its cut edges don't add false frequencies.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "hann window"
 },
 "hard negative": {
  "def": "A training example that looks relevant but isn't, such as the right topic with the wrong answer. Training on them teaches fine distinctions.",
  "lesson": "primer/ml/embeddings/contrastive.html",
  "term": "hard negative"
 },
 "harmful compliance": {
  "def": "Answering a request that should have been refused.",
  "lesson": "primer/ml/alignment.html",
  "term": "harmful compliance"
 },
 "hash chain": {
  "def": "Records that each store the previous record's hash, so any edit or deletion is detectable.",
  "lesson": "primer/agents/deployment.html",
  "term": "hash chain"
 },
 "hbm": {
  "def": "The GPU's large main memory: tens of GB, about 10× slower than on-chip SRAM.",
  "lesson": "primer/ml/inference.html",
  "term": "HBM"
 },
 "hdbscan": {
  "def": "A version of DBSCAN that considers every reach at once and keeps the most persistent clusters, so no reach setting is needed.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "HDBSCAN"
 },
 "he initialization": {
  "def": "Starting weights with variance 2 / fan-in, making up for ReLU zeroing half its inputs.",
  "lesson": "primer/ml/deep_nets.html",
  "term": "He initialization"
 },
 "hidden state": {
  "def": "The running summary an RNN carries from one step to the next.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "hidden state"
 },
 "hidden tests": {
  "def": "Grading tests the agent never sees, so it can't pass by satisfying the tests instead of the intent.",
  "lesson": "primer/agents/coding_agents.html",
  "term": "hidden tests"
 },
 "hierarchical softmax": {
  "def": "Replacing one softmax over the whole vocabulary with about log₂V yes-or-no decisions down a binary tree.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "hierarchical softmax"
 },
 "hit@k": {
  "def": "The share of queries with at least one relevant result in the top k.",
  "lesson": "primer/ml/embeddings/operations.html",
  "term": "hit@k"
 },
 "hnsw": {
  "def": "A layered graph index for vector search: long jumps on sparse upper layers, then a careful local search at the bottom.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "HNSW"
 },
 "host memory": {
  "def": "The CPU's main memory, reached from the GPU over a much slower link.",
  "lesson": "primer/ml/hardware.html",
  "term": "host memory"
 },
 "huffman tree": {
  "def": "A binary tree that gives frequent items short codes and rare items long ones.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "huffman tree"
 },
 "human approval gate": {
  "def": "A rule that parks irreversible or high-value actions until a person approves them.",
  "lesson": "primer/agents/tools.html",
  "term": "human approval gate"
 },
 "hybrid model": {
  "def": "A stack that mixes a few attention layers with many state-space layers.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "hybrid model"
 },
 "hybrid search": {
  "def": "Running keyword search and vector search together and merging the two ranked lists.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "hybrid search"
 },
 "hyde": {
  "def": "Hypothetical document embeddings: have the model write a plausible answer, then search with that, since answers resemble documents more than questions do.",
  "lesson": "primer/agents/rag.html",
  "term": "HyDE"
 },
 "hysteresis": {
  "def": "Making the bar for changing state higher than the bar for staying, so a decision doesn't flicker between two close options.",
  "lesson": null,
  "term": "hysteresis"
 },
 "idempotency key": {
  "def": "A unique id sent with a write so a retried request returns the first result instead of acting twice.",
  "lesson": "primer/agents/tools.html",
  "term": "idempotency key"
 },
 "idempotent": {
  "def": "Safe to repeat: doing it twice has the same effect as doing it once, so retries can't create duplicates.",
  "lesson": "primer/agents/tools.html",
  "term": "idempotent"
 },
 "idf": {
  "def": "Inverse document frequency: a word's rarity weight, higher for words found in fewer documents.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "IDF"
 },
 "implicit reward": {
  "def": "In DPO, β times the log of how much more likely training has made a response than the reference model did.",
  "lesson": "primer/ml/training_stages.html",
  "term": "implicit reward"
 },
 "in-batch negatives": {
  "def": "Using the other examples in a training batch as free wrong answers for each query.",
  "lesson": "primer/ml/embeddings/contrastive.html",
  "term": "in-batch negatives"
 },
 "in-context learning": {
  "def": "Picking up a task from instructions or examples in the prompt, with no change to the model's weights.",
  "lesson": "primer/ml/big_picture.html",
  "term": "in-context learning"
 },
 "index alias": {
  "def": "A pointer name like \"live\" that search uses, so switching indexes means repointing one name.",
  "lesson": "primer/ml/embeddings/operations.html",
  "term": "index alias"
 },
 "inertia": {
  "def": "The total squared distance from each point to its cluster's center: the quantity k-means minimizes.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "inertia"
 },
 "inference": {
  "def": "Using a trained model to make predictions, as opposed to training it.",
  "lesson": "primer/ml/inference.html",
  "term": "inference"
 },
 "infonce": {
  "def": "A contrastive loss that treats the other items in the batch as wrong answers for each matching pair.",
  "lesson": "primer/ml/losses.html",
  "term": "InfoNCE"
 },
 "information gain": {
  "def": "How much a split lowers entropy; the tree picks the split that lowers it most.",
  "lesson": "primer/ml/classical.html",
  "term": "information gain"
 },
 "initialization": {
  "def": "Choosing the size of a network's random starting weights so signals neither shrink nor grow layer by layer.",
  "lesson": "primer/ml/deep_nets.html",
  "term": "initialization"
 },
 "input gate": {
  "def": "The LSTM dial that decides what new information to write into the cell state.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "input gate"
 },
 "instrumentation": {
  "def": "Code that records spans or metrics around the work a program does.",
  "lesson": "primer/agents/observability.html",
  "term": "instrumentation"
 },
 "int4 quantization": {
  "def": "Storing each weight as a 4-bit integer times a shared scale.",
  "lesson": "primer/ml/inference.html",
  "term": "int4 quantization"
 },
 "int8 quantization": {
  "def": "Storing each weight as an 8-bit integer times a shared scale.",
  "lesson": "primer/ml/inference.html",
  "term": "int8 quantization"
 },
 "interference": {
  "def": "Signals getting in each other's way: other features leaking into one feature's reading (superposition), or merged task vectors changing the same weights in opposite directions (model merging).",
  "lesson": "primer/ml/interpretability.html",
  "term": "interference"
 },
 "internal fragmentation": {
  "def": "Memory reserved for a request that it never uses.",
  "lesson": "primer/ml/inference.html",
  "term": "internal fragmentation"
 },
 "interpretability": {
  "def": "Studying what a model's internal numbers represent, and which of them cause its outputs.",
  "lesson": "primer/ml/interpretability.html",
  "term": "interpretability"
 },
 "inverted file": {
  "def": "In an IVF index, the list of vectors assigned to one cluster, so a search scans only the lists it picks.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "inverted file"
 },
 "isoflop profile": {
  "def": "A curve of final loss against model size at a fixed training compute; its minimum is the best size for that budget.",
  "lesson": "primer/ml/training_stages.html",
  "term": "isoflop profile"
 },
 "item response theory": {
  "def": "Modelling the chance of a right answer from a taker's ability minus a question's difficulty.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "item response theory"
 },
 "ivf": {
  "def": "A vector index that clusters vectors ahead of time and searches only the clusters nearest the query.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "IVF"
 },
 "ivf-pq": {
  "def": "IVF clustering combined with compressed offsets from each cluster centre: the standard index for billion-scale search.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "IVF-PQ"
 },
 "jaccard similarity": {
  "def": "Shared items divided by all distinct items across two sets, from 0 to 1.",
  "lesson": "primer/ml/fine_tuning.html",
  "term": "Jaccard similarity"
 },
 "jensen-shannon divergence": {
  "def": "A measure of how different two distributions are; stuck at log 2 whenever they don't overlap.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "Jensen-Shannon divergence"
 },
 "jitter": {
  "def": "Random variation added to retry delays so clients don't all retry at the same moment.",
  "lesson": "primer/agents/failures.html",
  "term": "jitter"
 },
 "json mode": {
  "def": "A decoding option that guarantees parseable JSON, but not any particular shape.",
  "lesson": "primer/ml/structured_output.html",
  "term": "JSON mode"
 },
 "json schema": {
  "def": "A standard way to describe the shape of JSON data: which fields exist, their types and which are required.",
  "lesson": "primer/agents/tools.html",
  "term": "JSON Schema"
 },
 "json-rpc": {
  "def": "A simple convention for calling functions on another program by exchanging JSON requests, replies and notifications.",
  "lesson": "primer/agents/mcp.html",
  "term": "JSON-RPC"
 },
 "k-means": {
  "def": "Clustering by repeatedly assigning each point to its nearest center and moving each center to the average of its points.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "k-means"
 },
 "k-means++": {
  "def": "A way to pick k-means starting centers spread far apart, which avoids bad results.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "k-means++"
 },
 "kernel": {
  "def": "In a CNN, the small grid of learned weights a convolution slides over an image; also called a filter.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "kernel"
 },
 "kernel fusion": {
  "def": "Doing several steps in one GPU program, so data is read from and written to main memory only once.",
  "lesson": "primer/ml/inference.html",
  "term": "kernel fusion"
 },
 "key": {
  "def": "In attention, the vector a token offers so others can decide whether it is relevant to them.",
  "lesson": "primer/ml/attention.html",
  "term": "key"
 },
 "kill switch": {
  "def": "A way to stop an agent instantly, per customer or globally.",
  "lesson": "primer/agents/deployment.html",
  "term": "kill switch"
 },
 "kl divergence": {
  "def": "The extra surprise from using distribution q when the truth is p; zero only when they match.",
  "lesson": "primer/ml/training_stages.html",
  "term": "KL divergence"
 },
 "kv cache": {
  "def": "Stored keys and values from earlier tokens, so each new token is computed without redoing work. It trades GPU memory for speed.",
  "lesson": "primer/ml/inference.html",
  "term": "KV cache"
 },
 "kv-cache quantization": {
  "def": "Storing cached keys and values in fewer bits, with one scale per vector.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "kv-cache quantization"
 },
 "l1 regularization": {
  "def": "Penalizing the sum of absolute weights, which drives unneeded weights to exactly zero.",
  "lesson": "primer/ml/regularization.html",
  "term": "l1 regularization"
 },
 "l2 normalization": {
  "def": "Dividing a vector by its own length so it has length 1.",
  "lesson": "primer/ml/embeddings/similarity.html",
  "term": "l2 normalization"
 },
 "l2 regularization": {
  "def": "Penalizing the sum of squared weights, which shrinks all weights toward zero without eliminating them.",
  "lesson": "primer/ml/regularization.html",
  "term": "L2 regularization"
 },
 "label noise": {
  "def": "Reference labels that are wrong, which cap what a model can learn and what an eval can measure.",
  "lesson": "primer/ml/fine_tuning.html",
  "term": "label noise"
 },
 "label smoothing": {
  "def": "Training against a softened target, such as 0.9 on the right class and the rest spread evenly, to discourage over-confidence.",
  "lesson": "primer/ml/losses.html",
  "term": "label smoothing"
 },
 "language identification": {
  "def": "Guessing which language a text is in, so a pipeline keeps only the ones it wants.",
  "lesson": "primer/ml/pretraining.html",
  "term": "language identification"
 },
 "late interaction": {
  "def": "Keeping one vector per word and matching each query word to its best document word.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "late interaction"
 },
 "latency": {
  "def": "The delay between something happening and the system's response to it.",
  "lesson": "primer/agents/cost.html",
  "term": "latency"
 },
 "latent diffusion": {
  "def": "Running diffusion on an autoencoder's small compressed code instead of on pixels, then decoding once at the end.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "latent diffusion"
 },
 "latent space": {
  "def": "The space of codes a model works in, where position means something and nearby codes decode to similar data.",
  "lesson": "primer/ml/generative/autoencoders.html",
  "term": "latent space"
 },
 "layer normalization": {
  "def": "Rescaling each token's numbers to a steady average and spread, which keeps training stable.",
  "lesson": "primer/ml/deep_nets.html",
  "term": "layer normalization"
 },
 "layout shift": {
  "def": "Controls moving between runs, so clicks at remembered coordinates miss.",
  "lesson": "primer/agents/coding_agents.html",
  "term": "layout shift"
 },
 "leaf": {
  "def": "An end box of a decision tree, predicting the majority label or average value of the training examples that reached it.",
  "lesson": "primer/ml/classical.html",
  "term": "leaf"
 },
 "learning rate": {
  "def": "How big a step each training update takes. Too high and training blows up; too low and it crawls.",
  "lesson": "primer/ml/optimizers.html",
  "term": "learning rate"
 },
 "learning-rate schedule": {
  "def": "A rule that changes the learning rate during training, such as warming up and then decaying along a cosine curve.",
  "lesson": "primer/ml/optimizers.html",
  "term": "learning-rate schedule"
 },
 "least privilege": {
  "def": "Giving each agent or tool only the permissions its job needs.",
  "lesson": "primer/agents/tools.html",
  "term": "least privilege"
 },
 "length normalization": {
  "def": "BM25's discount for mentions in documents longer than average, controlled by b.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "length normalization"
 },
 "lethal trifecta": {
  "def": "Private data, untrusted content and a way to send data out, all in one agent: together they allow data theft.",
  "lesson": "primer/agents/guardrails.html",
  "term": "lethal trifecta"
 },
 "linear attention": {
  "def": "Attention variants that avoid scoring every pair of tokens, so cost grows in proportion to n instead of n².",
  "lesson": "primer/ml/attention.html",
  "term": "linear attention"
 },
 "linear probe": {
  "def": "Freezing a model and training only a simple linear classifier on its embeddings or internal activations, to measure what they encode.",
  "lesson": "primer/ml/embeddings/contrastive.html",
  "term": "linear probe"
 },
 "linear representation hypothesis": {
  "def": "The idea that features are directions, readable with a dot product.",
  "lesson": "primer/ml/interpretability.html",
  "term": "linear representation hypothesis"
 },
 "llm": {
  "def": "Large language model: a transformer trained to predict the next token, then tuned to follow instructions.",
  "lesson": "primer/agents/llm.html",
  "term": "LLM"
 },
 "llm-as-judge": {
  "def": "Using a language model with a rubric to grade open-ended outputs, checked against human ratings.",
  "lesson": "primer/agents/evals.html",
  "term": "llm-as-judge"
 },
 "load-balancing loss": {
  "def": "A small extra training penalty that pushes a mixture-of-experts router to spread tokens evenly across experts.",
  "lesson": "primer/ml/transformer.html",
  "term": "load-balancing loss"
 },
 "local model": {
  "def": "A language model running on your own machine rather than a provider's servers: free per call, private and offline, but smaller.",
  "lesson": "primer/agents/llm.html",
  "term": "local model"
 },
 "locality-sensitive hashing": {
  "def": "Hashing vectors so that similar vectors are likely to land in the same bucket, for fast approximate search.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "locality-sensitive hashing"
 },
 "log-derivative trick": {
  "def": "Rewriting ∇π as π·∇log π, so a sampled action gives an estimate of the gradient.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "log-derivative trick"
 },
 "log-likelihood": {
  "def": "The log of how probable the observed data is under a model.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "log-likelihood"
 },
 "log-mel spectrogram": {
  "def": "A spectrogram pooled into mel bands with loudness on a log scale: what speech models read.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "log-mel spectrogram"
 },
 "log-probability": {
  "def": "The logarithm of a probability; for a whole response, the sum of its tokens' log-probabilities.",
  "lesson": "primer/ml/training_stages.html",
  "term": "log-probability"
 },
 "log-sum-exp": {
  "def": "Computing the log of a sum of exponentials without overflow, by factoring out the largest term first.",
  "lesson": "primer/ml/losses.html",
  "term": "log-sum-exp"
 },
 "logarithm": {
  "def": "The undo button for exponentiation: ln(y) asks what power of e gives y. Logs turn multiplication into addition.",
  "lesson": "primer/notation.html",
  "term": "logarithm"
 },
 "logistic regression": {
  "def": "A linear model whose weighted sum passes through a sigmoid to give a probability.",
  "lesson": "primer/ml/neural_net.html",
  "term": "logistic regression"
 },
 "logit difference": {
  "def": "The correct answer's score minus a wrong answer's score.",
  "lesson": "primer/ml/interpretability.html",
  "term": "logit difference"
 },
 "logit lens": {
  "def": "Applying the model's own output layer to intermediate layers to see what it would predict so far.",
  "lesson": "primer/ml/interpretability.html",
  "term": "logit lens"
 },
 "logits": {
  "def": "The raw scores a model outputs for every possible next token, before softmax turns them into probabilities.",
  "lesson": "primer/ml/big_picture.html",
  "term": "logits"
 },
 "long-term memory": {
  "def": "Facts, events and procedures stored outside the model and recalled into later conversations.",
  "lesson": "primer/agents/memory.html",
  "term": "long-term memory"
 },
 "loop detection": {
  "def": "Stopping an agent that keeps requesting the same tool with the same arguments.",
  "lesson": "primer/agents/agent_loop.html",
  "term": "loop detection"
 },
 "lora": {
  "def": "Low-rank adaptation: freeze the model and train a small pair of matrices whose product is added to a weight matrix.",
  "lesson": "primer/ml/training_stages.html",
  "term": "LoRA"
 },
 "loss": {
  "def": "A single number measuring how wrong the model's predictions are. Training tries to make it smaller.",
  "lesson": "primer/ml/losses.html",
  "term": "loss"
 },
 "loss function": {
  "def": "The rule that turns predictions and correct answers into the loss. Choosing it defines what the model learns.",
  "lesson": "primer/ml/losses.html",
  "term": "loss function"
 },
 "loss mask": {
  "def": "A per-position switch deciding which predictions count toward the loss, such as only the reply in SFT.",
  "lesson": "primer/ml/training_stages.html",
  "term": "loss mask"
 },
 "loss scaling": {
  "def": "Multiplying the loss so small fp16 gradients stay above zero, then dividing back.",
  "lesson": "primer/ml/hardware.html",
  "term": "loss scaling"
 },
 "loss spike": {
  "def": "A sudden jump in training loss that may recover or diverge.",
  "lesson": "primer/ml/pretraining.html",
  "term": "loss spike"
 },
 "lost in the middle": {
  "def": "Models use information at the start and end of a long input more reliably than information in the middle.",
  "lesson": "primer/agents/context.html",
  "term": "lost in the middle"
 },
 "lstm": {
  "def": "An RNN with gates that decide what to forget, what to write and what to output, so it can remember across long sequences.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "LSTM"
 },
 "luhn check": {
  "def": "A checksum every real card number satisfies, used to tell card numbers from look-alike IDs.",
  "lesson": "primer/agents/guardrails.html",
  "term": "Luhn check"
 },
 "machine translation": {
  "def": "Turning text in one language into another.",
  "lesson": "primer/ml/transformer.html",
  "term": "machine translation"
 },
 "majority voting": {
  "def": "Choosing the answer that the most samples agree on.",
  "lesson": "primer/ml/reasoning.html",
  "term": "majority voting"
 },
 "mamba": {
  "def": "A state-space model that trains in parallel and runs in time linear in sequence length.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "Mamba"
 },
 "mantissa": {
  "def": "The bits of a float that store its digits, which set its precision.",
  "lesson": "primer/ml/hardware.html",
  "term": "mantissa"
 },
 "map@10": {
  "def": "Mean average precision over the top 10 results: rewards putting correct matches in the top 10, and higher up within it.",
  "lesson": "primer/ml/metrics.html",
  "term": "map@10"
 },
 "margin of error": {
  "def": "The ± range around a measured score that the true score probably falls in; it shrinks with the square root of the sample size.",
  "lesson": "primer/ml/fine_tuning.html",
  "term": "margin of error"
 },
 "master weights": {
  "def": "An fp32 copy of the weights that receives optimizer updates, so tiny updates aren't lost.",
  "lesson": "primer/ml/pretraining.html",
  "term": "master weights"
 },
 "matrix": {
  "def": "A table of numbers with rows and columns. A batch of vectors stacked as rows is a matrix, and most model weights are matrices.",
  "lesson": "primer/notation.html",
  "term": "matrix"
 },
 "matrix multiply": {
  "def": "A grid of dot products: each output cell is one row of the first matrix dotted with one column of the second.",
  "lesson": "primer/notation.html",
  "term": "matrix multiply"
 },
 "matryoshka embedding": {
  "def": "An embedding trained so its first few dimensions work as a smaller embedding on their own.",
  "lesson": "primer/ml/embeddings/compression.html",
  "term": "matryoshka embedding"
 },
 "max pooling": {
  "def": "Keeping only the largest value in each small block of a feature map.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "max pooling"
 },
 "max-norm constraint": {
  "def": "Capping the length of each neuron's incoming weight vector at a fixed radius, shrinking it back whenever an update pushes it past.",
  "lesson": "primer/ml/regularization.html",
  "term": "max-norm constraint"
 },
 "maxsim": {
  "def": "For each query word, its highest similarity to any document word, summed over the query.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "MaxSim"
 },
 "mcp": {
  "def": "Model Context Protocol: an open standard for connecting AI apps to tools and data through reusable servers.",
  "lesson": "primer/agents/mcp.html",
  "term": "MCP"
 },
 "mcp resource": {
  "def": "Read-only data an MCP server offers for the app to load into context, addressed by a URI.",
  "lesson": "primer/agents/mcp.html",
  "term": "mcp resource"
 },
 "mcp server": {
  "def": "The program that wraps a real system (files, a database, an API) and offers it to AI apps over MCP.",
  "lesson": "primer/agents/mcp.html",
  "term": "MCP server"
 },
 "mean absolute error": {
  "def": "The average of absolute prediction errors, which is robust to outliers.",
  "lesson": "primer/ml/losses.html",
  "term": "mean absolute error"
 },
 "mean average precision": {
  "def": "How well a system ranks correct results above wrong ones, averaged over queries or classes.",
  "lesson": "primer/ml/metrics.html",
  "term": "mean average precision"
 },
 "mean squared error": {
  "def": "The average of squared prediction errors, so big misses dominate.",
  "lesson": "primer/ml/losses.html",
  "term": "mean squared error"
 },
 "mean-centering": {
  "def": "Subtracting the average vector of a collection from every vector, removing the direction they all share.",
  "lesson": "primer/ml/embeddings/similarity.html",
  "term": "mean-centering"
 },
 "mel scale": {
  "def": "A relabelling of frequency so equal steps sound equally far apart to human ears.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "mel scale"
 },
 "memory bandwidth": {
  "def": "How many bytes per second a memory can deliver.",
  "lesson": "primer/ml/hardware.html",
  "term": "memory bandwidth"
 },
 "memory hierarchy": {
  "def": "The chain of storage from registers to the network, each level bigger and slower than the last.",
  "lesson": "primer/ml/hardware.html",
  "term": "memory hierarchy"
 },
 "memory-bound": {
  "def": "Limited by how fast data arrives from memory rather than by arithmetic, as in decoding.",
  "lesson": "primer/ml/inference.html",
  "term": "memory-bound"
 },
 "micro-batch": {
  "def": "A slice of a batch that moves through a pipeline on its own.",
  "lesson": "primer/ml/pretraining.html",
  "term": "micro-batch"
 },
 "minhash": {
  "def": "A short signature per document whose matching slots estimate Jaccard similarity.",
  "lesson": "primer/ml/pretraining.html",
  "term": "MinHash"
 },
 "minimax": {
  "def": "A game where one player tries to make a number as large as possible and the other as small as possible.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "minimax"
 },
 "mips": {
  "def": "Maximum inner product search: finding the stored vectors with the largest dot product against a query vector, usually approximately with an index.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "mips"
 },
 "mixed precision": {
  "def": "Doing the big multiplies in 16- or 8-bit while keeping master weights and sums in fp32.",
  "lesson": "primer/ml/hardware.html",
  "term": "mixed precision"
 },
 "mixture of experts": {
  "def": "Replacing one feed-forward network with many expert networks and a router that sends each token to a few of them.",
  "lesson": "primer/ml/transformer.html",
  "term": "Mixture of Experts"
 },
 "modality": {
  "def": "One kind of input a model can take: text, images, audio or video.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "modality"
 },
 "mode collapse": {
  "def": "A generator producing only a few kinds of output instead of the full variety in the data.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "mode collapse"
 },
 "model collapse": {
  "def": "Losing rare data (the tails) when models are trained on their own outputs.",
  "lesson": "primer/ml/pretraining.html",
  "term": "model collapse"
 },
 "model context protocol": {
  "def": "An open standard that lets any AI application connect to any tool server the same way.",
  "lesson": "primer/agents/mcp.html",
  "term": "Model Context Protocol"
 },
 "model merging": {
  "def": "Building one model from several fine-tunes by arithmetic on their weights, with no extra training.",
  "lesson": "primer/ml/fine_tuning.html",
  "term": "model merging"
 },
 "model routing": {
  "def": "Sending each request to the cheapest model that can handle it well.",
  "lesson": "primer/agents/cost.html",
  "term": "model routing"
 },
 "momentum": {
  "def": "Keeping a running average of recent gradients so updates roll smoothly through noise, like a ball gathering speed downhill.",
  "lesson": "primer/ml/optimizers.html",
  "term": "momentum"
 },
 "mrr": {
  "def": "Mean reciprocal rank: the average of 1 / (position of the first relevant result).",
  "lesson": "primer/ml/metrics.html",
  "term": "MRR"
 },
 "multi-agent system": {
  "def": "Several agents that coordinate, typically a supervisor delegating to specialist workers.",
  "lesson": "primer/agents/orchestration.html",
  "term": "multi-agent system"
 },
 "multi-armed bandit": {
  "def": "The simplest RL problem: pick among options with hidden payouts, and learn which pays best by trying them.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "multi-armed bandit"
 },
 "multi-head attention": {
  "def": "Several attention computations run in parallel on slices of the vectors, so each head can track a different kind of relationship.",
  "lesson": "primer/ml/attention.html",
  "term": "multi-head attention"
 },
 "multi-head latent attention": {
  "def": "Caching one small latent vector per token and rebuilding every head's keys and values from it.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "multi-head latent attention"
 },
 "multi-query attention": {
  "def": "All query heads share a single key/value head.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "multi-query attention"
 },
 "multi-query retrieval": {
  "def": "Searching several phrasings of a question and fusing the results.",
  "lesson": "primer/agents/rag.html",
  "term": "multi-query retrieval"
 },
 "multi-tenant": {
  "def": "One system serving many separate customers whose data must never mix.",
  "lesson": "primer/agents/memory.html",
  "term": "multi-tenant"
 },
 "multimodal model": {
  "def": "A model that takes in more than one kind of input (text, images, audio, video) as one sequence of tokens.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "multimodal model"
 },
 "n-gram overlap": {
  "def": "The share of a text's n-word runs that also appear in another corpus.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "n-gram overlap"
 },
 "named-entity recognition": {
  "def": "A model that tags names, places and organisations in text.",
  "lesson": "primer/agents/guardrails.html",
  "term": "named-entity recognition"
 },
 "navigable small world": {
  "def": "A graph where always stepping to the neighbour closest to the target reaches it in few hops.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "navigable small world"
 },
 "ndcg": {
  "def": "A ranking score that gives more credit for relevant results near the top and handles degrees of relevance.",
  "lesson": "primer/ml/metrics.html",
  "term": "nDCG"
 },
 "near-duplicate detection": {
  "def": "Finding items whose embeddings are almost identical, such as reworded copies.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "near-duplicate detection"
 },
 "negative sampling": {
  "def": "Training by scoring the true pair up and a few random noise pairs down, instead of scoring the whole vocabulary.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "negative sampling"
 },
 "ner": {
  "def": "Named-entity recognition: tagging names, places and organisations in text.",
  "lesson": "primer/agents/guardrails.html",
  "term": "NER"
 },
 "neuron": {
  "def": "Multiply each input by a weight, add them up with a bias, and pass the result through a nonlinear function.",
  "lesson": "primer/ml/neural_net.html",
  "term": "neuron"
 },
 "nf4": {
  "def": "4-bit NormalFloat: a 4-bit number format whose 16 levels are spaced to match the bell-curve shape of neural network weights.",
  "lesson": "primer/ml/inference.html",
  "term": "NF4"
 },
 "noise schedule": {
  "def": "How much noise each step adds (the betas), which sets how fast the signal fades.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "noise schedule"
 },
 "non-saturating loss": {
  "def": "The generator loss −log D(G(z)), which keeps a strong gradient when the discriminator confidently rejects fakes.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "non-saturating loss"
 },
 "norm": {
  "def": "The length of a vector: square the entries, add them, take the square root.",
  "lesson": "primer/notation.html",
  "term": "norm"
 },
 "nprobe": {
  "def": "How many IVF clusters a query scans; raising it trades speed for recall.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "nprobe"
 },
 "ntk-aware scaling": {
  "def": "Extending a RoPE model's context by raising the rotation base, which slows the slow frequencies while keeping the fast ones that encode local order.",
  "lesson": "primer/ml/positional.html",
  "term": "NTK-aware scaling"
 },
 "oauth": {
  "def": "A standard way for a user to grant an app limited, revocable access without sharing their password.",
  "lesson": "primer/agents/mcp.html",
  "term": "oauth"
 },
 "ocr": {
  "def": "Optical character recognition: reading text from an image of a page.",
  "lesson": "primer/agents/rag.html",
  "term": "ocr"
 },
 "off-by-one error": {
  "def": "An index or count one position away from the right one.",
  "lesson": "primer/agents/coding_agents.html",
  "term": "off-by-one error"
 },
 "ollama": {
  "def": "A free app that downloads open language models and runs them on your own computer, served over a small local web API.",
  "lesson": "primer/agents/llm.html",
  "term": "ollama"
 },
 "one-hot": {
  "def": "A vector with a 1 at the correct class and 0 everywhere else.",
  "lesson": "primer/ml/losses.html",
  "term": "one-hot"
 },
 "one-hot encoding": {
  "def": "Turning a category into one yes/no column per possible value.",
  "lesson": "primer/ml/classical.html",
  "term": "one-hot encoding"
 },
 "opentelemetry": {
  "def": "The open standard for traces, metrics and logs.",
  "lesson": "primer/agents/observability.html",
  "term": "OpenTelemetry"
 },
 "optimal discriminator": {
  "def": "p_data/(p_data + p_g): the best possible verdict against a fixed generator; 1/2 everywhere at equilibrium.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "optimal discriminator"
 },
 "optimizer": {
  "def": "The rule that turns gradients into weight updates, such as SGD, momentum or Adam.",
  "lesson": "primer/ml/optimizers.html",
  "term": "optimizer"
 },
 "orchestrator-workers": {
  "def": "One model decides the subtasks, workers handle them, and one model combines the results.",
  "lesson": "primer/agents/orchestration.html",
  "term": "orchestrator-workers"
 },
 "otlp": {
  "def": "The OpenTelemetry Protocol: the wire format exporters use to ship telemetry.",
  "lesson": "primer/agents/observability.html",
  "term": "otlp"
 },
 "out-of-bag": {
  "def": "The rows left out of a tree's bootstrap sample, usable as free validation data for that tree.",
  "lesson": "primer/ml/classical.html",
  "term": "out-of-bag"
 },
 "outcome reward model": {
  "def": "A verifier that scores only the final answer.",
  "lesson": "primer/ml/reasoning.html",
  "term": "outcome reward model"
 },
 "outer product": {
  "def": "A column vector times a row vector, giving a table whose entry (m, c) is the product of their m-th and c-th numbers.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "outer product"
 },
 "output gate": {
  "def": "The LSTM dial that decides how much of the cell state to show as output.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "output gate"
 },
 "over-refusal": {
  "def": "Refusing a harmless request because a safety check is too strict.",
  "lesson": "primer/ml/alignment.html",
  "term": "over-refusal"
 },
 "overfitting": {
  "def": "When a model memorizes its training data, noise included, and does worse on new data.",
  "lesson": "primer/ml/regularization.html",
  "term": "overfitting"
 },
 "overthinking": {
  "def": "Spending many reasoning tokens where few would do, wasting cost and time and sometimes losing a right answer.",
  "lesson": "primer/ml/reasoning.html",
  "term": "overthinking"
 },
 "p95": {
  "def": "The 95th percentile: the value that 95% of measurements are at or below.",
  "lesson": "primer/agents/evals.html",
  "term": "p95"
 },
 "padding": {
  "def": "Zeros added around an image so a filter can centre on the edge pixels.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "padding"
 },
 "pagedattention": {
  "def": "Storing the KV cache in fixed-size pages, like virtual memory, to avoid wasted GPU memory.",
  "lesson": "primer/ml/inference.html",
  "term": "pagedattention"
 },
 "paired bootstrap": {
  "def": "Resampling questions with both models' results kept together, to test whether a gap is real.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "paired bootstrap"
 },
 "parallel scan": {
  "def": "Computing every state of a linear recurrence in about log₂ n parallel rounds by merging steps.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "parallel scan"
 },
 "parallel tool calls": {
  "def": "Several tool requests in one model turn, run at the same time, with all results returned in one message.",
  "lesson": "primer/agents/agent_loop.html",
  "term": "parallel tool calls"
 },
 "parallelism": {
  "def": "Doing many pieces of work at the same time instead of one after another.",
  "lesson": "primer/ml/attention.html",
  "term": "parallelism"
 },
 "parameters": {
  "def": "All the learned numbers in a model (its weights and biases). A 70B model has 70 billion of them.",
  "lesson": "primer/ml/neural_net.html",
  "term": "parameters"
 },
 "parent-child retrieval": {
  "def": "Matching small chunks but returning the larger section around them.",
  "lesson": "primer/agents/rag.html",
  "term": "parent-child retrieval"
 },
 "pass-to-pass test": {
  "def": "A hidden test that passed before a fix and must still pass after it.",
  "lesson": "primer/agents/coding_agents.html",
  "term": "pass-to-pass test"
 },
 "pass@1": {
  "def": "The share of problems solved by the first program submitted for each, judged by hidden tests.",
  "lesson": "primer/agents/evals.html",
  "term": "pass@1"
 },
 "pass@k": {
  "def": "The chance that at least one of k sampled answers passes the tests.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "pass@k"
 },
 "pass@n": {
  "def": "The chance that at least one of n samples is right: 1 − (1 − p)ⁿ.",
  "lesson": "primer/ml/reasoning.html",
  "term": "pass@n"
 },
 "patch": {
  "def": "A small square cut from an image and flattened into one token.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "patch"
 },
 "patch embedding": {
  "def": "The learned matrix that turns a flattened image patch into a token vector.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "patch embedding"
 },
 "pca": {
  "def": "Principal component analysis: finding the directions along which data varies most, to draw or compress it with fewer numbers.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "PCA"
 },
 "per-channel quantization": {
  "def": "One scale per weight row, so a single outlier doesn't coarsen all the others.",
  "lesson": "primer/ml/inference.html",
  "term": "per-channel quantization"
 },
 "permission-aware retrieval": {
  "def": "Filtering out documents a user may not read before anything is ranked or shown to the model.",
  "lesson": "primer/agents/rag.html",
  "term": "permission-aware retrieval"
 },
 "permutation equivariance": {
  "def": "Shuffling a layer's inputs just shuffles its outputs the same way, which is why attention alone cannot see word order.",
  "lesson": "primer/ml/positional.html",
  "term": "permutation equivariance"
 },
 "permutation importance": {
  "def": "The accuracy lost on held-out data when one column is shuffled.",
  "lesson": "primer/ml/classical.html",
  "term": "permutation importance"
 },
 "perplexity": {
  "def": "e raised to the average cross-entropy: roughly how many options the model is torn between at each step.",
  "lesson": "primer/ml/losses.html",
  "term": "perplexity"
 },
 "petaflop/s-day": {
  "def": "10¹⁵ operations per second for one day, 8.64 × 10¹⁹ operations: a unit for training budgets.",
  "lesson": "primer/ml/training_stages.html",
  "term": "petaflop/s-day"
 },
 "pii": {
  "def": "Personally identifiable information: names, emails, phone numbers, card numbers and similar.",
  "lesson": "primer/agents/guardrails.html",
  "term": "PII"
 },
 "pipeline bubble": {
  "def": "Time pipeline stages sit idle while the pipeline fills and drains.",
  "lesson": "primer/ml/pretraining.html",
  "term": "pipeline bubble"
 },
 "pipeline parallelism": {
  "def": "Giving each GPU a stage of layers and passing activations between neighbouring stages.",
  "lesson": "primer/ml/hardware.html",
  "term": "pipeline parallelism"
 },
 "plan-and-execute": {
  "def": "An agent pattern that writes a plan first, then executes it step by step, replanning when results surprise it.",
  "lesson": "primer/agents/planning.html",
  "term": "plan-and-execute"
 },
 "pmi": {
  "def": "Pointwise mutual information: the log of how much more often two words appear together than chance predicts.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "PMI"
 },
 "policy": {
  "def": "In reinforcement learning and preference tuning, the model being trained, viewed as a probability distribution over actions or responses.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "policy"
 },
 "policy gradient": {
  "def": "Raising expected reward by making the actions that earned more reward more likely.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "policy gradient"
 },
 "polysemantic neuron": {
  "def": "A neuron that responds to several unrelated features.",
  "lesson": "primer/ml/interpretability.html",
  "term": "polysemantic neuron"
 },
 "pooling": {
  "def": "Shrinking a feature map by keeping only the strongest (or average) value in each small window.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "pooling"
 },
 "popcount": {
  "def": "Counting the 1-bits in a number; combined with XOR it computes Hamming distance in one step.",
  "lesson": "primer/ml/embeddings/compression.html",
  "term": "popcount"
 },
 "position interpolation": {
  "def": "Stretching a model to longer contexts by scaling every position down into the range it saw during training.",
  "lesson": "primer/ml/positional.html",
  "term": "position interpolation"
 },
 "positional encoding": {
  "def": "Information added to token vectors so the model knows word order, which attention alone ignores.",
  "lesson": "primer/ml/positional.html",
  "term": "positional encoding"
 },
 "posterior collapse": {
  "def": "When a VAE's code carries no information because the KL penalty outweighs what the code saves in rebuild error, so every output is the same average.",
  "lesson": "primer/ml/generative/autoencoders.html",
  "term": "posterior collapse"
 },
 "power iteration": {
  "def": "Repeatedly multiplying a vector by a matrix (and its transpose) until it points along the most-stretched direction.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "power iteration"
 },
 "ppo": {
  "def": "Proximal Policy Optimization: a reinforcement learning algorithm that improves a policy in small, clipped steps; widely used for RLHF.",
  "lesson": "primer/ml/training_stages.html",
  "term": "PPO"
 },
 "pre-norm": {
  "def": "Normalizing before each sub-layer of a transformer block, leaving the residual path untouched, which trains more stably than normalizing after.",
  "lesson": "primer/ml/transformer.html",
  "term": "pre-norm"
 },
 "pre-tokenization": {
  "def": "Splitting text into chunks, such as words with their leading space, before BPE runs, so merges never cross chunk boundaries.",
  "lesson": "primer/ml/tokenization.html",
  "term": "pre-tokenization"
 },
 "precision": {
  "def": "Of the items flagged, the fraction that were right.",
  "lesson": "primer/ml/metrics.html",
  "term": "precision"
 },
 "precision-recall curve": {
  "def": "Precision plotted against recall across thresholds; more honest than ROC for rare events.",
  "lesson": "primer/ml/metrics.html",
  "term": "precision-recall curve"
 },
 "precision@k": {
  "def": "The share of the top k search results that are relevant.",
  "lesson": "primer/ml/metrics.html",
  "term": "precision@k"
 },
 "preference tuning": {
  "def": "Training on which of two responses people preferred, to shape tone, helpfulness and safety.",
  "lesson": "primer/ml/training_stages.html",
  "term": "preference tuning"
 },
 "prefill": {
  "def": "The first phase of generation: the whole prompt is processed in one parallel pass. It sets the time to the first token.",
  "lesson": "primer/ml/inference.html",
  "term": "prefill"
 },
 "prefix tuning": {
  "def": "Training a few special virtual-token vectors placed before every input while the model stays frozen.",
  "lesson": "primer/ml/training_stages.html",
  "term": "prefix tuning"
 },
 "pretraining": {
  "def": "The first, most expensive training stage: predicting the next token over trillions of tokens of text.",
  "lesson": "primer/ml/training_stages.html",
  "term": "pretraining"
 },
 "privilege separation": {
  "def": "Splitting work so the part that reads untrusted content can't take dangerous actions.",
  "lesson": "primer/agents/guardrails.html",
  "term": "privilege separation"
 },
 "probability distribution": {
  "def": "A list of probabilities over all possible outcomes, each between 0 and 1, adding up to 1.",
  "lesson": "primer/notation.html",
  "term": "probability distribution"
 },
 "probability ratio": {
  "def": "The current policy's probability of a sampled action divided by its probability when the action was sampled.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "probability ratio"
 },
 "probe": {
  "def": "A small classifier trained on a frozen model's activations to test whether a property is encoded there.",
  "lesson": "primer/ml/interpretability.html",
  "term": "probe"
 },
 "procedural memory": {
  "def": "Memory of how to do things, such as a learned workflow or saved skill.",
  "lesson": "primer/agents/memory.html",
  "term": "procedural memory"
 },
 "process reward model": {
  "def": "A verifier that scores each intermediate step.",
  "lesson": "primer/ml/reasoning.html",
  "term": "process reward model"
 },
 "product quantization": {
  "def": "Compressing a vector by splitting it into chunks and replacing each chunk with the ID of its nearest entry in a small codebook.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "product quantization"
 },
 "projector": {
  "def": "A small layer that maps an encoder's vectors into a language model's embedding space.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "projector"
 },
 "prompt": {
  "def": "The text sent to a language model: instructions, context and the question.",
  "lesson": "primer/agents/llm.html",
  "term": "prompt"
 },
 "prompt caching": {
  "def": "Reusing the processed form of a prompt prefix that repeats across requests, which cuts cost and time to first token.",
  "lesson": "primer/ml/inference.html",
  "term": "prompt caching"
 },
 "prompt chaining": {
  "def": "A fixed sequence of model calls where each output feeds the next, with code checks in between.",
  "lesson": "primer/agents/orchestration.html",
  "term": "prompt chaining"
 },
 "prompt injection": {
  "def": "Hostile instructions hidden in content the model reads, such as an email or web page, trying to hijack it.",
  "lesson": "primer/agents/guardrails.html",
  "term": "prompt injection"
 },
 "pushdown automaton": {
  "def": "A finite-state machine plus a stack: enough to check nested formats like JSON or SQL.",
  "lesson": "primer/ml/structured_output.html",
  "term": "pushdown automaton"
 },
 "qlora": {
  "def": "LoRA adapters trained on top of base weights stored in 4 bits.",
  "lesson": "primer/ml/training_stages.html",
  "term": "QLoRA"
 },
 "quality filter": {
  "def": "A rule or classifier that throws out low-quality pages before training.",
  "lesson": "primer/ml/pretraining.html",
  "term": "quality filter"
 },
 "quantization": {
  "def": "Storing numbers with fewer bits (say 8 or 4 instead of 16 or 32), which shrinks memory with a small loss of precision.",
  "lesson": "primer/ml/inference.html",
  "term": "quantization"
 },
 "query": {
  "def": "In attention, the vector a token uses to ask what it is looking for.",
  "lesson": "primer/ml/attention.html",
  "term": "query"
 },
 "query and passage prefixes": {
  "def": "Labels some embedding models expect on questions and documents; forgetting one silently lowers recall.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "query and passage prefixes"
 },
 "query rewriting": {
  "def": "Turning a follow-up or vague question into a standalone search query.",
  "lesson": "primer/agents/rag.html",
  "term": "query rewriting"
 },
 "rag": {
  "def": "Retrieval-augmented generation: fetch relevant passages first, then have the model answer using them, with citations.",
  "lesson": "primer/agents/rag.html",
  "term": "RAG"
 },
 "random forest": {
  "def": "Many deep trees on bootstrap samples, each split limited to a random subset of features, with their votes averaged.",
  "lesson": "primer/ml/classical.html",
  "term": "random forest"
 },
 "rank": {
  "def": "How many independent directions a matrix really contains; a table made by multiplying one column by one row has rank 1.",
  "lesson": "primer/ml/training_stages.html",
  "term": "rank"
 },
 "rate limit": {
  "def": "A cap on how many actions may happen per unit of time.",
  "lesson": "primer/agents/deployment.html",
  "term": "rate limit"
 },
 "re-scoring": {
  "def": "Recomputing exact scores for a shortlist that an approximate index returned, to recover accuracy.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "re-scoring"
 },
 "react": {
  "def": "Reason plus act: the agent pattern that interleaves reasoning steps with tool calls.",
  "lesson": "primer/agents/agent_loop.html",
  "term": "ReAct"
 },
 "reasoning model": {
  "def": "A model trained to write out and check intermediate steps before it commits to an answer.",
  "lesson": "primer/ml/reasoning.html",
  "term": "reasoning model"
 },
 "recall": {
  "def": "Of the items that truly mattered, the fraction that were found.",
  "lesson": "primer/ml/metrics.html",
  "term": "recall"
 },
 "recall@k": {
  "def": "The fraction of relevant documents that appear in the top k results. For RAG, usually the metric that matters most.",
  "lesson": "primer/ml/metrics.html",
  "term": "recall@k"
 },
 "receptive field": {
  "def": "How much of the original input one neuron can see.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "receptive field"
 },
 "reciprocal rank fusion": {
  "def": "Merging ranked lists by giving each document 1 / (k + rank) from every list it appears in, then adding those up.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "reciprocal rank fusion"
 },
 "rectified flow": {
  "def": "Flow matching with straight-line paths, retrained on its own outputs so the paths get straighter and need fewer steps.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "rectified flow"
 },
 "red-teaming": {
  "def": "Searching systematically for inputs that make a model or its safety checks fail.",
  "lesson": "primer/ml/alignment.html",
  "term": "red-teaming"
 },
 "reference model": {
  "def": "A frozen copy of the starting model that DPO and RLHF measure drift against.",
  "lesson": "primer/ml/training_stages.html",
  "term": "reference model"
 },
 "reflection": {
  "def": "Having a model review and revise its own output. Useful, but external checks such as tests are more reliable.",
  "lesson": "primer/agents/planning.html",
  "term": "reflection"
 },
 "register": {
  "def": "The tiny storage right beside the arithmetic units, holding the numbers being worked on this instant.",
  "lesson": "primer/ml/hardware.html",
  "term": "register"
 },
 "regression": {
  "def": "A task that used to pass and now fails after a change.",
  "lesson": "primer/agents/evals.html",
  "term": "regression"
 },
 "regret": {
  "def": "In online learning, the total extra loss from choosing each step's parameters before seeing that step's data, compared with the best fixed choice in hindsight.",
  "lesson": "primer/ml/optimizers.html",
  "term": "regret"
 },
 "regular expression": {
  "def": "A pattern language (such as [0-9]+ or cat|car|dog) that can always be compiled into a finite-state machine.",
  "lesson": "primer/ml/structured_output.html",
  "term": "regular expression"
 },
 "regularization": {
  "def": "Any technique that discourages memorizing, such as dropout, weight decay or early stopping.",
  "lesson": "primer/ml/regularization.html",
  "term": "regularization"
 },
 "reinforce": {
  "def": "The basic policy-gradient algorithm: step along reward times the gradient of log π(action).",
  "lesson": "primer/ml/reinforcement.html",
  "term": "REINFORCE"
 },
 "reinforcement learning": {
  "def": "Learning from a score for what you did, rather than from the correct answer.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "reinforcement learning"
 },
 "release gate": {
  "def": "Limits set before measuring; a release goes ahead only if every evaluation is within its limit.",
  "lesson": "primer/ml/alignment.html",
  "term": "release gate"
 },
 "relu": {
  "def": "An activation function that keeps positive numbers and turns negatives into zero.",
  "lesson": "primer/ml/neural_net.html",
  "term": "ReLU"
 },
 "reparameterization trick": {
  "def": "Writing a random draw as z = μ + σ·ε with the noise ε as a separate input, so gradients can flow through sampling.",
  "lesson": "primer/ml/generative/autoencoders.html",
  "term": "reparameterization trick"
 },
 "replay": {
  "def": "Mixing a small sample of old-task examples into new training data so the old skill keeps getting practised.",
  "lesson": "primer/ml/fine_tuning.html",
  "term": "replay"
 },
 "reranker": {
  "def": "A second, more accurate model that reorders the top results of a fast first-stage search.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "reranker"
 },
 "residual": {
  "def": "The true value minus the current prediction; for squared error it is the negative gradient.",
  "lesson": "primer/ml/classical.html",
  "term": "residual"
 },
 "residual connection": {
  "def": "Adding a layer's input back to its output (x + f(x)), giving gradients a shortcut through deep networks.",
  "lesson": "primer/ml/deep_nets.html",
  "term": "residual connection"
 },
 "residual stream": {
  "def": "The running vector for each token that every transformer block reads from and adds a correction to, instead of replacing it.",
  "lesson": "primer/ml/transformer.html",
  "term": "residual stream"
 },
 "residual vector quantization": {
  "def": "Stacking codebooks, each one encoding the error the previous ones left.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "residual vector quantization"
 },
 "resolved rate": {
  "def": "The share of tasks whose patch passes every hidden fail-to-pass and pass-to-pass test.",
  "lesson": "primer/agents/coding_agents.html",
  "term": "resolved rate"
 },
 "retrieval failure": {
  "def": "A wrong answer caused because no relevant document was retrieved.",
  "lesson": "primer/ml/embeddings/operations.html",
  "term": "retrieval failure"
 },
 "reward": {
  "def": "The single number the environment returns to say how good an action was.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "reward"
 },
 "reward hacking": {
  "def": "A policy maximising the reward as written while the real goal gets worse.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "reward hacking"
 },
 "reward model": {
  "def": "A model trained to predict which of two responses a human would prefer.",
  "lesson": "primer/ml/training_stages.html",
  "term": "reward model"
 },
 "right to erasure": {
  "def": "A user's legal right, for example under GDPR, to have their personal data deleted.",
  "lesson": "primer/agents/memory.html",
  "term": "right to erasure"
 },
 "ring all-reduce": {
  "def": "All-reduce by passing chunks around a ring; each GPU sends about twice its data, however many GPUs there are.",
  "lesson": "primer/ml/hardware.html",
  "term": "ring all-reduce"
 },
 "rlaif": {
  "def": "Reinforcement learning from AI feedback: preference labels come from a model applying written principles.",
  "lesson": "primer/ml/alignment.html",
  "term": "RLAIF"
 },
 "rlhf": {
  "def": "Reinforcement learning from human feedback: a reward model learns human preferences, and the language model is tuned to score well on it.",
  "lesson": "primer/ml/training_stages.html",
  "term": "RLHF"
 },
 "rmsnorm": {
  "def": "A cheaper layer normalization that divides each token's numbers by their root-mean-square without subtracting the mean first.",
  "lesson": "primer/ml/transformer.html",
  "term": "RMSNorm"
 },
 "rmsprop": {
  "def": "An optimizer that divides each step by the square root of a running average of recent squared gradients.",
  "lesson": "primer/ml/optimizers.html",
  "term": "rmsprop"
 },
 "rnn": {
  "def": "A recurrent neural network: it reads a sequence one step at a time, carrying a running summary called the hidden state.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "RNN"
 },
 "roc curve": {
  "def": "True-positive rate plotted against false-positive rate as the decision threshold sweeps.",
  "lesson": "primer/ml/metrics.html",
  "term": "ROC curve"
 },
 "roc-auc": {
  "def": "The probability that the model ranks a random positive above a random negative.",
  "lesson": "primer/ml/metrics.html",
  "term": "ROC-AUC"
 },
 "rolling buffer cache": {
  "def": "A KV cache with w slots where each new token overwrites the oldest.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "rolling buffer cache"
 },
 "roofline": {
  "def": "A chart of the speed a chip can reach at each arithmetic intensity, capped first by memory and then by compute.",
  "lesson": "primer/ml/inference.html",
  "term": "roofline"
 },
 "rope": {
  "def": "Rotary position embedding: encodes position by rotating query and key vectors, so attention depends on the distance between tokens.",
  "lesson": "primer/ml/positional.html",
  "term": "RoPE"
 },
 "rouge-l": {
  "def": "An overlap score based on the longest in-order sequence of words shared with a reference.",
  "lesson": "primer/ml/metrics.html",
  "term": "ROUGE-L"
 },
 "router": {
  "def": "A step that classifies a request and sends it down the right path, or to the right model.",
  "lesson": "primer/agents/orchestration.html",
  "term": "router"
 },
 "routing": {
  "def": "One model call classifies a request and sends it to a specialised handler.",
  "lesson": "primer/agents/orchestration.html",
  "term": "routing"
 },
 "rubric": {
  "def": "A short list of explicit pass/fail criteria given to a grader.",
  "lesson": "primer/agents/evals.html",
  "term": "rubric"
 },
 "rug pull": {
  "def": "A tool server changing its definitions after they were approved.",
  "lesson": "primer/agents/mcp.html",
  "term": "rug pull"
 },
 "sample rate": {
  "def": "How many measurements of a signal are taken per second, such as 16,000 for speech.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "sample rate"
 },
 "sandbox": {
  "def": "An isolated place to run untrusted code, where the worst it can do is fail: no network, no secrets, time and memory limits.",
  "lesson": "primer/agents/coding_agents.html",
  "term": "sandbox"
 },
 "sandwich ordering": {
  "def": "Placing the best retrieved chunks at the start and end of the context and the weakest in the middle.",
  "lesson": "primer/agents/context.html",
  "term": "sandwich ordering"
 },
 "saturation": {
  "def": "When an activation like sigmoid or tanh sits on its flat tail, so almost no gradient passes through.",
  "lesson": "primer/ml/neural_net.html",
  "term": "saturation"
 },
 "scalar quantization": {
  "def": "Storing each number as one of 256 levels (one byte) instead of a 4-byte float.",
  "lesson": "primer/ml/embeddings/compression.html",
  "term": "scalar quantization"
 },
 "score": {
  "def": "The direction in which data gets more crowded fastest; the noise guess, flipped and rescaled.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "score"
 },
 "screenshot": {
  "def": "The image of the screen a computer-use agent receives after each action.",
  "lesson": "primer/agents/coding_agents.html",
  "term": "screenshot"
 },
 "selective state-space model": {
  "def": "A state-space model whose step size (how much to keep and write) is computed from each token, as in Mamba.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "selective state-space model"
 },
 "self-attention": {
  "def": "Attention where a sequence attends to itself: every token scores every other token in the same text.",
  "lesson": "primer/ml/attention.html",
  "term": "self-attention"
 },
 "self-consistency": {
  "def": "Sampling several chains of thought and returning the most common final answer.",
  "lesson": "primer/ml/reasoning.html",
  "term": "self-consistency"
 },
 "semantic cache": {
  "def": "Reusing a stored answer when a new question's embedding is nearly identical to a previous question's.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "semantic cache"
 },
 "semantic memory": {
  "def": "Memory of facts, such as 'the user's fiscal year starts in April'.",
  "lesson": "primer/agents/memory.html",
  "term": "semantic memory"
 },
 "semantic validation": {
  "def": "Checking that well-formed arguments are also true, such as that a customer id actually exists.",
  "lesson": "primer/agents/tools.html",
  "term": "semantic validation"
 },
 "sentencepiece": {
  "def": "A tokenizer library that runs BPE or Unigram directly on raw text, writing spaces as the visible symbol ▁.",
  "lesson": "primer/ml/tokenization.html",
  "term": "sentencepiece"
 },
 "sft": {
  "def": "Supervised fine-tuning: training on examples of instructions paired with good responses.",
  "lesson": "primer/ml/training_stages.html",
  "term": "SFT"
 },
 "sha-256": {
  "def": "A hash function: a fixed-length fingerprint of data that changes completely if the data changes at all.",
  "lesson": "primer/agents/deployment.html",
  "term": "SHA-256"
 },
 "shadow mode": {
  "def": "Running a new system on real inputs and recording what it would do, without letting it act.",
  "lesson": "primer/agents/deployment.html",
  "term": "shadow mode"
 },
 "shingle": {
  "def": "A window of k neighbouring words, used to compare texts for near-duplicates.",
  "lesson": "primer/ml/fine_tuning.html",
  "term": "shingle"
 },
 "short-term memory": {
  "def": "The current conversation, kept within a token budget: recent turns verbatim, older ones summarized.",
  "lesson": "primer/agents/memory.html",
  "term": "short-term memory"
 },
 "short-time fourier transform": {
  "def": "A Fourier transform on each short, overlapping slice of a signal.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "short-time fourier transform"
 },
 "shortlist": {
  "def": "The small set of top candidates from a cheap first-stage search that a slower, more precise stage reorders.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "shortlist"
 },
 "shots": {
  "def": "Worked examples placed in the prompt before the question.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "shots"
 },
 "shrinkage": {
  "def": "Pulling values toward zero: an L1 penalty's pull on every activation, or in gradient boosting the learning rate that keeps only part of each new tree's correction.",
  "lesson": "primer/ml/interpretability.html",
  "term": "shrinkage"
 },
 "siamese network": {
  "def": "Two copies of one network with shared weights, each reading one input, so their outputs can be compared directly.",
  "lesson": "primer/ml/embeddings/contrastive.html",
  "term": "siamese network"
 },
 "sigmoid": {
  "def": "Squashes any number into the range 0 to 1, with an S-shaped curve.",
  "lesson": "primer/ml/neural_net.html",
  "term": "sigmoid"
 },
 "silhouette score": {
  "def": "How much closer each point is to its own cluster than to the nearest other one, from −1 to 1.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "silhouette score"
 },
 "singular value decomposition": {
  "def": "Breaking a matrix into independent directions ranked by importance, so the top few capture most of what it does.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "singular value decomposition"
 },
 "sinusoidal positional encoding": {
  "def": "The original transformer's fixed position codes, made of sines and cosines at many frequencies, added to each token's embedding.",
  "lesson": "primer/ml/positional.html",
  "term": "sinusoidal positional encoding"
 },
 "skip list": {
  "def": "A sorted list with random express lanes on top, searched by running along the top lane and dropping down, in about log(n) steps.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "skip list"
 },
 "skip-gram": {
  "def": "The word2vec task of predicting each word's neighbours from the word itself.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "skip-gram"
 },
 "sliding-window attention": {
  "def": "Each token attends only to the last w tokens, so cost and cache stop growing with context length.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "sliding-window attention"
 },
 "soft targets": {
  "def": "A teacher model's temperature-softened probabilities, used to train a student model.",
  "lesson": "primer/ml/training_stages.html",
  "term": "soft targets"
 },
 "soft thresholding": {
  "def": "Moving each weight a fixed amount toward zero and snapping any weight within that amount to exactly zero.",
  "lesson": "primer/ml/regularization.html",
  "term": "soft thresholding"
 },
 "softmax": {
  "def": "Turns a list of scores into shares that are all positive and add up to 1, with bigger scores getting disproportionately more.",
  "lesson": "primer/ml/attention.html",
  "term": "softmax"
 },
 "softplus": {
  "def": "log(1 + eˣ): a smooth ramp that is always positive.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "softplus"
 },
 "span": {
  "def": "One step inside a trace, with its start time, duration and details.",
  "lesson": "primer/agents/observability.html",
  "term": "span"
 },
 "sparse attention": {
  "def": "Attention that scores only a chosen pattern of token pairs (local, global, strided) instead of every pair.",
  "lesson": "primer/ml/efficient_architectures.html",
  "term": "sparse attention"
 },
 "sparse autoencoder": {
  "def": "A wide encoder and decoder trained with an L1 penalty so each input uses a few learned features.",
  "lesson": "primer/ml/interpretability.html",
  "term": "sparse autoencoder"
 },
 "sparse gradients": {
  "def": "Gradients that are zero most of the time for a given weight, such as the weight for a rare word.",
  "lesson": "primer/ml/optimizers.html",
  "term": "sparse gradients"
 },
 "sparse retrieval": {
  "def": "Keyword search such as BM25, which scores exact word matches.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "sparse retrieval"
 },
 "spearman correlation": {
  "def": "How well two rankings agree, from −1 (reversed) to 1 (identical).",
  "lesson": "primer/ml/metrics.html",
  "term": "spearman correlation"
 },
 "specification gaming": {
  "def": "Another name for reward hacking: satisfying the letter of an objective but not its intent.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "specification gaming"
 },
 "spectral normalization": {
  "def": "Dividing each layer's weights by the most they can stretch any input, which caps how fast the discriminator can change.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "spectral normalization"
 },
 "spectrogram": {
  "def": "A picture of sound: time across, frequency up, brightness for loudness.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "spectrogram"
 },
 "speculative decoding": {
  "def": "A small fast model drafts several tokens, and the big model checks them all in one pass, keeping the ones it agrees with.",
  "lesson": "primer/ml/inference.html",
  "term": "speculative decoding"
 },
 "speculative execution": {
  "def": "Starting likely work early, in parallel, and throwing it away if the guess was wrong.",
  "lesson": "primer/ml/inference.html",
  "term": "speculative execution"
 },
 "sram": {
  "def": "The tiny, very fast on-chip memory next to a GPU's arithmetic units.",
  "lesson": "primer/ml/inference.html",
  "term": "SRAM"
 },
 "standard deviation": {
  "def": "The square root of the variance: the typical distance of a value from the average.",
  "lesson": "primer/notation.html",
  "term": "standard deviation"
 },
 "standard error": {
  "def": "The typical distance between a measured score and the true rate: √(p(1−p)/n).",
  "lesson": "primer/ml/benchmarks.html",
  "term": "standard error"
 },
 "standard normal distribution": {
  "def": "The bell curve centred on 0 with spread 1; N(0, I) draws each number from it independently.",
  "lesson": "primer/ml/generative/autoencoders.html",
  "term": "standard normal distribution"
 },
 "state machine": {
  "def": "A fixed set of states and allowed transitions, with code deciding every move.",
  "lesson": "primer/agents/orchestration.html",
  "term": "state machine"
 },
 "state-space model": {
  "def": "A recurrent-style model, such as Mamba, that trains in parallel and runs in time linear in sequence length.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "state-space model"
 },
 "static batching": {
  "def": "Serving a fixed group of requests until the longest one finishes.",
  "lesson": "primer/ml/inference.html",
  "term": "static batching"
 },
 "static embedding": {
  "def": "One fixed vector per word, whatever the sentence.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "static embedding"
 },
 "stdio transport": {
  "def": "Running a server as a child process and exchanging one JSON message per line over its input and output.",
  "lesson": "primer/agents/mcp.html",
  "term": "stdio transport"
 },
 "stochastic gradient descent": {
  "def": "Gradient descent where each step's slope is estimated from a small random batch instead of the whole dataset.",
  "lesson": "primer/ml/optimizers.html",
  "term": "stochastic gradient descent"
 },
 "stop reason": {
  "def": "Why a model response ended: finished, wants a tool, hit the length limit, or declined.",
  "lesson": "primer/agents/agent_loop.html",
  "term": "stop reason"
 },
 "strict mode": {
  "def": "A tool or output option that guarantees the model's JSON fits a given schema.",
  "lesson": "primer/ml/structured_output.html",
  "term": "strict mode"
 },
 "strict tool use": {
  "def": "An API setting that guarantees the model's tool arguments match the tool's JSON Schema exactly.",
  "lesson": "primer/agents/tools.html",
  "term": "strict tool use"
 },
 "stride": {
  "def": "How many pixels a convolution filter jumps between positions.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "stride"
 },
 "structured output": {
  "def": "Making a model's answer follow an exact format, such as JSON that fits a schema, so a program can read it.",
  "lesson": "primer/ml/structured_output.html",
  "term": "structured output"
 },
 "stump": {
  "def": "A decision tree with a single question and two leaves.",
  "lesson": "primer/ml/classical.html",
  "term": "stump"
 },
 "style control": {
  "def": "Adding length or format as extra factors in the rating fit, so style is separated from quality.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "style control"
 },
 "subnormal": {
  "def": "A float below the smallest normal value, with the hidden leading 1 dropped so it fades toward zero.",
  "lesson": "primer/ml/hardware.html",
  "term": "subnormal"
 },
 "subsampling": {
  "def": "Randomly skipping most occurrences of very frequent words during training, which is faster and improves rare-word vectors.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "subsampling"
 },
 "subword unit": {
  "def": "A piece of a word, from a single character to a whole frequent word; a fixed set of them can spell any word.",
  "lesson": "primer/ml/tokenization.html",
  "term": "subword unit"
 },
 "superposition": {
  "def": "Storing more features than there are neurons, as nearly perpendicular directions.",
  "lesson": "primer/ml/interpretability.html",
  "term": "superposition"
 },
 "supervisor": {
  "def": "In a multi-agent system, the agent that assigns tasks to specialist agents and assembles their answers.",
  "lesson": "primer/agents/orchestration.html",
  "term": "supervisor"
 },
 "svd": {
  "def": "Singular value decomposition: splits a table into a few directions that capture its main patterns, used to compress it.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "SVD"
 },
 "sycophancy": {
  "def": "A model changing its answer to agree with a view the user states.",
  "lesson": "primer/ml/alignment.html",
  "term": "sycophancy"
 },
 "system prompt": {
  "def": "Standing instructions sent before the conversation that set a model's role, rules and style.",
  "lesson": "primer/agents/llm.html",
  "term": "system prompt"
 },
 "t-sne": {
  "def": "An older neighbour-preserving 2-D projection; cluster sizes and gaps in its plots are not meaningful.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "t-SNE"
 },
 "tabular data": {
  "def": "Data in rows and columns, where each column is a meaningful quantity in its own units.",
  "lesson": "primer/ml/classical.html",
  "term": "tabular data"
 },
 "tanh": {
  "def": "Squashes any number into the range −1 to 1.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "tanh"
 },
 "task arithmetic": {
  "def": "Combining or removing skills by adding, scaling or subtracting task vectors from a base model's weights.",
  "lesson": "primer/ml/fine_tuning.html",
  "term": "task arithmetic"
 },
 "task budget": {
  "def": "Hard limits on steps and tokens for one agent task.",
  "lesson": "primer/agents/cost.html",
  "term": "task budget"
 },
 "task vector": {
  "def": "The change a fine-tune made to a model's weights: fine-tuned minus base.",
  "lesson": "primer/ml/fine_tuning.html",
  "term": "task vector"
 },
 "temperature": {
  "def": "A knob that sharpens (low) or flattens (high) the probability distribution before sampling a token.",
  "lesson": "primer/ml/inference.html",
  "term": "temperature"
 },
 "tenant": {
  "def": "One customer organisation on a shared platform.",
  "lesson": "primer/agents/cost.html",
  "term": "tenant"
 },
 "tenant isolation": {
  "def": "Partitioning storage so a request can only ever reach its own tenant's and user's data.",
  "lesson": "primer/agents/memory.html",
  "term": "tenant isolation"
 },
 "tensor": {
  "def": "A grid of numbers with any number of dimensions: a vector is 1-D, a matrix 2-D, a batch of images 4-D.",
  "lesson": "primer/notation.html",
  "term": "tensor"
 },
 "tensor parallelism": {
  "def": "Splitting each matrix multiply across GPUs, which must talk inside every layer.",
  "lesson": "primer/ml/hardware.html",
  "term": "tensor parallelism"
 },
 "term frequency": {
  "def": "How many times a word appears in a document.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "term frequency"
 },
 "test set": {
  "def": "Held-out data used once at the end to estimate real-world performance honestly.",
  "lesson": "primer/ml/regularization.html",
  "term": "test set"
 },
 "test-time compute": {
  "def": "Computation spent while answering rather than while training, such as longer chains or more samples.",
  "lesson": "primer/ml/reasoning.html",
  "term": "test-time compute"
 },
 "text extraction": {
  "def": "Pulling a web page's main text out of its HTML, leaving menus and adverts behind.",
  "lesson": "primer/ml/pretraining.html",
  "term": "text extraction"
 },
 "thinking budget": {
  "def": "The maximum number of tokens a model may spend reasoning before it must answer.",
  "lesson": "primer/ml/reasoning.html",
  "term": "thinking budget"
 },
 "threshold calibration": {
  "def": "Choosing a similarity cut-off by measuring precision and recall on labeled pairs for one specific model.",
  "lesson": "primer/ml/embeddings/similarity.html",
  "term": "threshold calibration"
 },
 "tiling": {
  "def": "Loading a block of data into fast memory once and doing all its work before evicting it.",
  "lesson": "primer/ml/hardware.html",
  "term": "tiling"
 },
 "token": {
  "def": "A piece of text from a model's fixed vocabulary: often a whole common word, or a fragment of a rare one. Models read and write tokens, not words.",
  "lesson": "primer/ml/tokenization.html",
  "term": "token"
 },
 "token bucket": {
  "def": "A rate limiter that allows bursts up to a capacity and refills at a steady rate.",
  "lesson": "primer/agents/deployment.html",
  "term": "token bucket"
 },
 "token budget": {
  "def": "A cap on the tokens, and so the money, one agent run may spend.",
  "lesson": "primer/agents/agent_loop.html",
  "term": "token budget"
 },
 "token healing": {
  "def": "Backing up over the last prompt token so the model can rewrite it, when a prompt ends partway through what would normally be one token.",
  "lesson": "primer/ml/structured_output.html",
  "term": "token healing"
 },
 "tokenizer": {
  "def": "The component that splits text into tokens and maps each to an integer ID.",
  "lesson": "primer/ml/tokenization.html",
  "term": "tokenizer"
 },
 "tool call": {
  "def": "A structured request from the model to run one of your functions with specific arguments; your code decides whether to run it.",
  "lesson": "primer/agents/tools.html",
  "term": "tool call"
 },
 "tool calling": {
  "def": "The model outputs a structured request to run a function; your code runs it and sends the result back. The model never executes anything itself.",
  "lesson": "primer/agents/llm.html",
  "term": "tool calling"
 },
 "tool poisoning": {
  "def": "Hiding instructions for the model inside a tool's description.",
  "lesson": "primer/agents/mcp.html",
  "term": "tool poisoning"
 },
 "tool_result": {
  "def": "The message block that returns a tool's output or error to the model, matched to its request by id.",
  "lesson": "primer/agents/agent_loop.html",
  "term": "tool_result"
 },
 "top-k": {
  "def": "Sampling only from the k most likely next tokens.",
  "lesson": "primer/ml/inference.html",
  "term": "top-k"
 },
 "top-k retrieval accuracy": {
  "def": "The share of questions for which at least one of the top k retrieved passages contains the answer.",
  "lesson": "primer/ml/metrics.html",
  "term": "top-k retrieval accuracy"
 },
 "top-p": {
  "def": "Nucleus sampling: pick only from the smallest set of tokens whose probabilities add up to p.",
  "lesson": "primer/ml/inference.html",
  "term": "top-p"
 },
 "trace": {
  "def": "A record of one run as a tree of steps (model calls, tool calls, retrievals) with inputs, outputs, timing and cost.",
  "lesson": "primer/agents/observability.html",
  "term": "trace"
 },
 "trajectory": {
  "def": "The full record of an agent run: its answer, its tool calls with arguments, and the end state.",
  "lesson": "primer/agents/evals.html",
  "term": "trajectory"
 },
 "transformer": {
  "def": "The neural network architecture behind modern language models: stacked blocks of attention followed by a small feed-forward network.",
  "lesson": "primer/ml/transformer.html",
  "term": "transformer"
 },
 "transpose": {
  "def": "Flip a matrix so its rows become columns. Written with a superscript T.",
  "lesson": "primer/notation.html",
  "term": "transpose"
 },
 "triplet loss": {
  "def": "A loss requiring an anchor to be closer to its positive than to its negative by at least a margin.",
  "lesson": "primer/ml/losses.html",
  "term": "triplet loss"
 },
 "true-positive rate": {
  "def": "The share of real positives the model flags; the same as recall.",
  "lesson": "primer/ml/metrics.html",
  "term": "true-positive rate"
 },
 "ttl": {
  "def": "Time to live: how long a cached entry may be served before it is treated as expired.",
  "lesson": "primer/agents/cost.html",
  "term": "TTL"
 },
 "tubelet": {
  "def": "A video patch that spans several frames as well as a square of pixels.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "tubelet"
 },
 "tuned lens": {
  "def": "A logit lens with a small learned translator per layer.",
  "lesson": "primer/ml/interpretability.html",
  "term": "tuned lens"
 },
 "two time-scale update rule": {
  "def": "Giving the generator and discriminator different learning rates so the game converges.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "two time-scale update rule"
 },
 "two-stage retrieval": {
  "def": "A cheap, wide first pass to shortlist candidates, then an expensive, precise pass over only those.",
  "lesson": "primer/ml/embeddings/compression.html",
  "term": "two-stage retrieval"
 },
 "umap": {
  "def": "A projection to 2-D that keeps each point's neighbours close but distorts other distances.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "UMAP"
 },
 "underfitting": {
  "def": "When a model is too simple, or undertrained, to capture the pattern at all.",
  "lesson": "primer/ml/regularization.html",
  "term": "underfitting"
 },
 "underflow": {
  "def": "A number too small for its format, rounded to zero.",
  "lesson": "primer/ml/pretraining.html",
  "term": "underflow"
 },
 "unigram tokenizer": {
  "def": "A subword tokenizer that starts from a huge vocabulary and repeatedly removes the pieces whose loss hurts least.",
  "lesson": "primer/ml/tokenization.html",
  "term": "unigram tokenizer"
 },
 "unit vector": {
  "def": "A vector rescaled to length 1, so it only carries a direction.",
  "lesson": "primer/notation.html",
  "term": "unit vector"
 },
 "utf-8": {
  "def": "The standard way to store text as bytes: 1 byte for basic Latin letters, 2 to 4 bytes for other characters and emoji.",
  "lesson": "primer/ml/tokenization.html",
  "term": "UTF-8"
 },
 "vae": {
  "def": "Variational autoencoder: an autoencoder whose codes are pulled towards a bell curve, so random codes decode to new data.",
  "lesson": "primer/ml/generative/autoencoders.html",
  "term": "VAE"
 },
 "validation set": {
  "def": "Held-out data used to tune choices like model size and when to stop training.",
  "lesson": "primer/ml/regularization.html",
  "term": "validation set"
 },
 "value": {
  "def": "In attention, the information a token hands over when others attend to it.",
  "lesson": "primer/ml/attention.html",
  "term": "value"
 },
 "value network": {
  "def": "A second model (the critic) that predicts expected reward, used as PPO's baseline.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "value network"
 },
 "vanishing gradient": {
  "def": "When gradients shrink towards zero as they pass back through many layers, so early layers stop learning.",
  "lesson": "primer/ml/deep_nets.html",
  "term": "vanishing gradient"
 },
 "variance": {
  "def": "How widely numbers are spread around their average: the average squared distance from the mean.",
  "lesson": "primer/notation.html",
  "term": "variance"
 },
 "variational autoencoder": {
  "def": "An autoencoder whose encoder outputs a fuzzy region (mean and spread) pulled towards the standard normal, so random codes decode to new data.",
  "lesson": "primer/ml/generative/autoencoders.html",
  "term": "variational autoencoder"
 },
 "vector": {
  "def": "A list of numbers, like (3, 1, 2). In AI, a word, sentence or image is represented as a vector.",
  "lesson": "primer/notation.html",
  "term": "vector"
 },
 "vector quantization": {
  "def": "Replacing a vector with the number of its nearest codebook entry.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "vector quantization"
 },
 "velocity field": {
  "def": "A map giving, at every point and time, which way and how fast a sample should move.",
  "lesson": "primer/ml/generative/diffusion.html",
  "term": "velocity field"
 },
 "verifiable reward": {
  "def": "A reward computed by a check that can't be argued with, such as a correct answer or passing tests.",
  "lesson": "primer/ml/reinforcement.html",
  "term": "verifiable reward"
 },
 "verifier": {
  "def": "Anything that scores a candidate solution, from a unit test to a learned model.",
  "lesson": "primer/ml/reasoning.html",
  "term": "verifier"
 },
 "vision transformer": {
  "def": "A transformer that treats small image patches as tokens.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "Vision Transformer"
 },
 "visual instruction tuning": {
  "def": "Fine-tuning a vision-language model on images paired with instructions and good answers.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "visual instruction tuning"
 },
 "vocabulary": {
  "def": "The fixed set of tokens a model knows, typically 32,000 to 200,000 entries.",
  "lesson": "primer/ml/tokenization.html",
  "term": "vocabulary"
 },
 "voronoi cell": {
  "def": "All the points closer to one centroid than to any other: the section an IVF index searches.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "Voronoi cell"
 },
 "vq-vae": {
  "def": "A VAE that snaps each code vector to the nearest entry of a learned codebook, turning images or audio into tokens.",
  "lesson": "primer/ml/generative/autoencoders.html",
  "term": "VQ-VAE"
 },
 "warmup": {
  "def": "Starting training with a tiny learning rate and ramping it up, which keeps the first updates from destabilizing the model.",
  "lesson": "primer/ml/optimizers.html",
  "term": "warmup"
 },
 "wasserstein distance": {
  "def": "The least work to reshape one distribution into another (mass moved times distance); also called earth mover's distance.",
  "lesson": "primer/ml/generative/gans.html",
  "term": "Wasserstein distance"
 },
 "waveform": {
  "def": "Sound recorded as a list of air-pressure measurements over time.",
  "lesson": "primer/ml/generative/multimodal.html",
  "term": "waveform"
 },
 "weight": {
  "def": "A learned number that says how strongly one input influences an output. Training adjusts the weights.",
  "lesson": "primer/ml/neural_net.html",
  "term": "weight"
 },
 "weight averaging": {
  "def": "Merging fine-tunes of the same base by averaging their weights (task arithmetic with λ = 1/T).",
  "lesson": "primer/ml/fine_tuning.html",
  "term": "weight averaging"
 },
 "weight decay": {
  "def": "Shrinking every weight slightly at each step, which penalizes large weights and keeps the model smoother.",
  "lesson": "primer/ml/regularization.html",
  "term": "weight decay"
 },
 "weight sharing": {
  "def": "Reusing the same filter at every position, so the parameter count doesn't grow with image size.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "weight sharing"
 },
 "weight tying": {
  "def": "Reusing the token embedding table as the output layer, so the vector that reads a token in also scores it on the way out.",
  "lesson": "primer/ml/transformer.html",
  "term": "weight tying"
 },
 "whitening": {
  "def": "Rescaling vectors so every direction has equal spread and no two directions are correlated.",
  "lesson": "primer/ml/embeddings/similarity.html",
  "term": "whitening"
 },
 "win rate": {
  "def": "The share of head-to-head comparisons one model wins; 50% means the two are indistinguishable.",
  "lesson": "primer/agents/evals.html",
  "term": "win rate"
 },
 "winner's curse": {
  "def": "The best of many versions chosen on a test looks better on it than it really is.",
  "lesson": "primer/ml/benchmarks.html",
  "term": "winner's curse"
 },
 "word2vec": {
  "def": "A 2013 method that learns one vector per word by predicting nearby words.",
  "lesson": "primer/ml/embeddings/word2vec.html",
  "term": "word2vec"
 },
 "wordpiece": {
  "def": "BERT's subword tokenizer, which merges the pair whose parts most rarely appear apart rather than simply the most frequent pair.",
  "lesson": "primer/ml/tokenization.html",
  "term": "WordPiece"
 },
 "workflow": {
  "def": "A fixed sequence of steps written in code, with a language model doing a task inside each step.",
  "lesson": "primer/agents/orchestration.html",
  "term": "workflow"
 },
 "xavier initialization": {
  "def": "Starting weights with variance 2 / (fan-in + fan-out), suited to tanh and sigmoid layers.",
  "lesson": "primer/ml/deep_nets.html",
  "term": "Xavier initialization"
 },
 "yarn": {
  "def": "A refinement of RoPE context extension that rescales each frequency band differently, used by many long-context models.",
  "lesson": "primer/ml/positional.html",
  "term": "YaRN"
 },
 "zero": {
  "def": "Sharding optimizer state, then gradients, then weights across data-parallel GPUs.",
  "lesson": "primer/ml/pretraining.html",
  "term": "zero"
 },
 "zero-shot": {
  "def": "Doing a task with no task-specific training examples.",
  "lesson": "primer/ml/embeddings/contrastive.html",
  "term": "zero-shot"
 },
 "zero-shot classification": {
  "def": "Labeling items by comparing them to a text description of each label, with no training on those labels.",
  "lesson": "primer/ml/embeddings/contrastive.html",
  "term": "zero-shot classification"
 },
 "β-vae": {
  "def": "A VAE whose KL penalty is weighted by β, trading rebuild sharpness for a smoother, more organized code space.",
  "lesson": "primer/ml/generative/autoencoders.html",
  "term": "β-VAE"
 }
};

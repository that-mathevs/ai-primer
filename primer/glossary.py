"""
# Glossary

Every term this primer uses, in plain English, with a link to the lesson
that builds it. On the HTML site, these definitions also appear when you hover
over (or tap) the term anywhere in a lesson or an annotated paper.

This module is the single source of truth: `make docs` turns it into
`glossary.js` for the browser. The table below is generated from `GLOSSARY`.
"""

from __future__ import annotations

import functools
import json
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    definition: str
    lesson: str | None = None  # dotted module that teaches it, e.g. "primer.ml.attention"
    # Everyday words with a technical meaning ("value", "policy", "patch") are only
    # given hover definitions on pages under these path prefixes, so a "travel
    # policy" never pops up a definition from preference tuning.
    scope: tuple[str, ...] = ()


_E = Entry
N = "primer.notation"
NN, OPT, DEEP = "primer.ml.neural_net", "primer.ml.optimizers", "primer.ml.deep_nets"
ATT, POS, TF, TOK = "primer.ml.attention", "primer.ml.positional", "primer.ml.transformer", "primer.ml.tokenization"
TRAIN, INF, LOSS, MET = "primer.ml.training_stages", "primer.ml.inference", "primer.ml.losses", "primer.ml.metrics"
REG, CNN, BIG = "primer.ml.regularization", "primer.ml.cnn_rnn", "primer.ml.big_picture"
W2V, SIM, CON = "primer.ml.embeddings.word2vec", "primer.ml.embeddings.similarity", "primer.ml.embeddings.contrastive"
CMP, ANN, RET = "primer.ml.embeddings.compression", "primer.ml.embeddings.ann", "primer.ml.embeddings.retrieval"
CLU, OPS = "primer.ml.embeddings.clustering", "primer.ml.embeddings.operations"
LLM, ORC, LOOP, TOOLS, MCP = "primer.agents.llm", "primer.agents.orchestration", "primer.agents.agent_loop", "primer.agents.tools", "primer.agents.mcp"
RAG, CTX, MEM, PLAN = "primer.agents.rag", "primer.agents.context", "primer.agents.memory", "primer.agents.planning"
EVAL, GUARD, COST, OBS, DEP = "primer.agents.evals", "primer.agents.guardrails", "primer.agents.cost", "primer.agents.observability", "primer.agents.deployment"

GLOSSARY: dict[str, Entry] = {
    # --- notation and maths -------------------------------------------------
    "vector": _E("A list of numbers, like (3, 1, 2). In AI, a word, sentence or image is represented as a vector.", N),
    "matrix": _E("A table of numbers with rows and columns. A batch of vectors stacked as rows is a matrix, and most model weights are matrices.", N),
    "tensor": _E("A grid of numbers with any number of dimensions: a vector is 1-D, a matrix 2-D, a batch of images 4-D.", N),
    "dot product": _E("Multiply two lists position by position and add up the products. It is large when two vectors point the same way.", N),
    "matrix multiply": _E("A grid of dot products: each output cell is one row of the first matrix dotted with one column of the second.", N),
    "transpose": _E("Flip a matrix so its rows become columns. Written with a superscript T.", N),
    "norm": _E("The length of a vector: square the entries, add them, take the square root.", N),
    "unit vector": _E("A vector rescaled to length 1, so it only carries a direction.", N),
    "logarithm": _E("The undo button for exponentiation: ln(y) asks what power of e gives y. Logs turn multiplication into addition.", N),
    "variance": _E("How widely numbers are spread around their average: the average squared distance from the mean.", N),
    "standard deviation": _E("The square root of the variance: the typical distance of a value from the average.", N),
    "derivative": _E("The slope of a function at a point: how much the output changes per tiny nudge of the input.", N),
    "gradient": _E("One slope per input, collected into a vector. It points in the direction that increases the function fastest.", N),
    "chain rule": _E("To get the slope through a chain of functions, multiply the slopes of each link.", N),
    "argmax": _E("The position of the largest value in a list, rather than the value itself.", N),
    "big-o": _E("A way to say how cost grows with input size, ignoring constant factors. O(n²) means doubling n quadruples the cost.", N),
    "probability distribution": _E("A list of probabilities over all possible outcomes, each between 0 and 1, adding up to 1.", N),

    # --- neural network basics ----------------------------------------------
    "neuron": _E("Multiply each input by a weight, add them up with a bias, and pass the result through a nonlinear function.", NN),
    "weight": _E("A learned number that says how strongly one input influences an output. Training adjusts the weights.", NN),
    'bias': _E("A learned number added to a neuron's weighted sum, letting it shift its output up or down.", NN, scope=('primer/ml/neural_net', 'primer/ml/deep_nets')),
    "parameters": _E("All the learned numbers in a model (its weights and biases). A 70B model has 70 billion of them.", NN),
    "activation function": _E("The nonlinear function inside a neuron, such as ReLU or GELU. Without it, stacked layers collapse into one.", NN),
    "relu": _E("An activation function that keeps positive numbers and turns negatives into zero.", NN),
    "gelu": _E("A smooth version of ReLU used in most transformers.", NN),
    "sigmoid": _E("Squashes any number into the range 0 to 1, with an S-shaped curve.", NN),
    "softmax": _E("Turns a list of scores into shares that are all positive and add up to 1, with bigger scores getting disproportionately more.", ATT),
    "forward pass": _E("Running inputs through the network, layer by layer, to get a prediction.", NN),
    "backpropagation": _E("The chain rule run backwards through the network to find, for every weight, how much nudging it would change the loss.", NN),
    "loss": _E("A single number measuring how wrong the model's predictions are. Training tries to make it smaller.", LOSS),
    "loss function": _E("The rule that turns predictions and correct answers into the loss. Choosing it defines what the model learns.", LOSS),
    "gradient descent": _E("Training by repeatedly taking a small step downhill: nudge every weight against its gradient to lower the loss.", OPT),
    "learning rate": _E("How big a step each training update takes. Too high and training blows up; too low and it crawls.", OPT),
    "optimizer": _E("The rule that turns gradients into weight updates, such as SGD, momentum or Adam.", OPT),
    "adam": _E("An optimizer that adapts the step size for each weight using running averages of its gradients.", OPT),
    "adamw": _E("Adam with weight decay applied separately from the gradient step. The default optimizer for transformers.", OPT),
    "momentum": _E("Keeping a running average of recent gradients so updates roll smoothly through noise, like a ball gathering speed downhill.", OPT),
    "batch": _E("The group of examples processed before one weight update.", NN),
    "epoch": _E("One full pass over the training data.", NN),
    "warmup": _E("Starting training with a tiny learning rate and ramping it up, which keeps the first updates from destabilizing the model.", OPT),
    "vanishing gradient": _E("When gradients shrink towards zero as they pass back through many layers, so early layers stop learning.", DEEP),
    "exploding gradient": _E("When gradients grow huge as they pass back through many layers, so training diverges.", DEEP),
    "residual connection": _E("Adding a layer's input back to its output (x + f(x)), giving gradients a shortcut through deep networks.", DEEP),
    "layer normalization": _E("Rescaling each token's numbers to a steady average and spread, which keeps training stable.", DEEP),
    "batch normalization": _E("Rescaling each feature using statistics from the current batch of examples. Common in image models.", DEEP),
    "gradient clipping": _E("Capping the size of the gradient so one bad batch can't throw the weights far off course.", DEEP),

    # --- transformers ---------------------------------------------------------
    "attention": _E("The mechanism that lets each token look at every other token, score how relevant each one is, and blend in information from the relevant ones.", ATT),
    "self-attention": _E("Attention where a sequence attends to itself: every token scores every other token in the same text.", ATT),
    'query': _E("In attention, the vector a token uses to ask what it is looking for.", ATT, scope=('primer/ml/attention', 'primer/ml/transformer')),
    'key': _E("In attention, the vector a token offers so others can decide whether it is relevant to them.", ATT, scope=('primer/ml/attention', 'primer/ml/transformer', 'primer/ml/inference', 'primer/ml/big_picture')),
    'value': _E("In attention, the information a token hands over when others attend to it.", ATT, scope=('primer/ml/attention', 'primer/ml/transformer', 'primer/ml/inference', 'primer/ml/big_picture')),
    "multi-head attention": _E("Several attention computations run in parallel on slices of the vectors, so each head can track a different kind of relationship.", ATT),
    "grouped-query attention": _E("Several query heads share one set of keys and values, shrinking the memory needed during generation.", ATT),
    "causal mask": _E("Hides future tokens from each token during attention, so a model predicting the next word can't peek at it.", ATT),
    "transformer": _E("The neural network architecture behind modern language models: stacked blocks of attention followed by a small feed-forward network.", TF),
    "feed-forward network": _E("The small two-layer network in each transformer block that processes every token on its own after attention has mixed them.", TF),
    "encoder": _E("A transformer that reads the whole input at once, with every token seeing every other. Used for embeddings and classification.", TF),
    "decoder": _E("A transformer that generates text left to right, each token seeing only the ones before it. GPT, Claude and Llama are decoders.", TF),
    "mixture of experts": _E("Replacing one feed-forward network with many expert networks and a router that sends each token to a few of them.", TF),
    "positional encoding": _E("Information added to token vectors so the model knows word order, which attention alone ignores.", POS),
    "rope": _E("Rotary position embedding: encodes position by rotating query and key vectors, so attention depends on the distance between tokens.", POS),
    "context window": _E("The maximum number of tokens a model can take in at once.", INF),
    "rmsnorm": _E("A cheaper layer normalization that divides each token's numbers by their root-mean-square without subtracting the mean first.", TF),
    "pre-norm": _E("Normalizing before each sub-layer of a transformer block, leaving the residual path untouched, which trains more stably than normalizing after.", TF),
    "residual stream": _E("The running vector for each token that every transformer block reads from and adds a correction to, instead of replacing it.", TF),
    "weight tying": _E("Reusing the token embedding table as the output layer, so the vector that reads a token in also scores it on the way out.", TF),
    "encoder-decoder": _E("A transformer where an encoder reads the whole input and a decoder writes the output one token at a time while attending to the encoder.", TF),
    "load-balancing loss": _E("A small extra training penalty that pushes a mixture-of-experts router to spread tokens evenly across experts.", TF),
    "flops": _E("Floating-point operations: individual multiplies or adds, used to measure compute (about 2 per parameter per token to generate, 6 to train).", TF),
    "permutation equivariance": _E("Shuffling a layer's inputs just shuffles its outputs the same way, which is why attention alone cannot see word order.", POS),
    "sinusoidal positional encoding": _E("The original transformer's fixed position codes, made of sines and cosines at many frequencies, added to each token's embedding.", POS),
    "position interpolation": _E("Stretching a model to longer contexts by scaling every position down into the range it saw during training.", POS),
    "ntk-aware scaling": _E("Extending a RoPE model's context by raising the rotation base, which slows the slow frequencies while keeping the fast ones that encode local order.", POS),
    "yarn": _E("A refinement of RoPE context extension that rescales each frequency band differently, used by many long-context models.", POS),
    "alibi": _E("A position scheme that adds no position vectors and instead subtracts a penalty proportional to distance from each attention score.", POS),
    "autoregressive generation": _E("Producing text one token at a time, where each new token is predicted from all the tokens before it and then appended.", BIG),
    "greedy decoding": _E("Always picking the single most probable next token, the same as sampling at temperature 0.", BIG),
    "logits": _E("The raw scores a model outputs for every possible next token, before softmax turns them into probabilities.", BIG),

    # --- tokens ---------------------------------------------------------------
    "token": _E("A piece of text from a model's fixed vocabulary: often a whole common word, or a fragment of a rare one. Models read and write tokens, not words.", TOK),
    "tokenizer": _E("The component that splits text into tokens and maps each to an integer ID.", TOK),
    "vocabulary": _E("The fixed set of tokens a model knows, typically 32,000 to 200,000 entries.", TOK),
    "byte pair encoding": _E("A tokenizer-building method that starts from characters and repeatedly merges the most frequent adjacent pair into a new token.", TOK),
    "byte-level bpe": _E("Byte pair encoding run on the raw bytes of UTF-8 text, so any string can be tokenized and there is never an unknown token.", TOK),
    "pre-tokenization": _E("Splitting text into chunks, such as words with their leading space, before BPE runs, so merges never cross chunk boundaries.", TOK),
    "wordpiece": _E("BERT's subword tokenizer, which merges the pair whose parts most rarely appear apart rather than simply the most frequent pair.", TOK),
    "unigram tokenizer": _E("A subword tokenizer that starts from a huge vocabulary and repeatedly removes the pieces whose loss hurts least.", TOK),
    "sentencepiece": _E("A tokenizer library that runs BPE or Unigram directly on raw text, writing spaces as the visible symbol ▁.", TOK),
    "utf-8": _E("The standard way to store text as bytes: 1 byte for basic Latin letters, 2 to 4 bytes for other characters and emoji.", TOK),
    "bpe": _E("Byte pair encoding: build a vocabulary by repeatedly merging the most frequent adjacent pair of symbols.", TOK),

    # --- training and inference -----------------------------------------------
    "pretraining": _E("The first, most expensive training stage: predicting the next token over trillions of tokens of text.", TRAIN),
    "base model": _E("A model after pretraining only: knowledgeable, but it continues text rather than following instructions.", TRAIN),
    "fine-tuning": _E("Training an existing model further on a smaller, targeted dataset to change its behavior.", TRAIN),
    "sft": _E("Supervised fine-tuning: training on examples of instructions paired with good responses.", TRAIN),
    "rlhf": _E("Reinforcement learning from human feedback: a reward model learns human preferences, and the language model is tuned to score well on it.", TRAIN),
    "dpo": _E("Direct preference optimization: tuning a model straight from pairs of preferred and rejected answers, without a separate reward model.", TRAIN),
    "reward model": _E("A model trained to predict which of two responses a human would prefer.", TRAIN),
    "lora": _E("Low-rank adaptation: freeze the model and train a small pair of matrices whose product is added to a weight matrix.", TRAIN),
    "distillation": _E("Training a small model to imitate a large one's outputs, often the biggest cost saving in production.", TRAIN),
    "inference": _E("Using a trained model to make predictions, as opposed to training it.", INF),
    "prefill": _E("The first phase of generation: the whole prompt is processed in one parallel pass. It sets the time to the first token.", INF),
    'decode': _E("The second phase of generation: output tokens are produced one at a time. It sets the tokens per second.", INF, scope=('primer/ml/inference',)),
    "kv cache": _E("Stored keys and values from earlier tokens, so each new token is computed without redoing work. It trades GPU memory for speed.", INF),
    "temperature": _E("A knob that sharpens (low) or flattens (high) the probability distribution before sampling a token.", INF),
    "top-p": _E("Nucleus sampling: pick only from the smallest set of tokens whose probabilities add up to p.", INF),
    "top-k": _E("Sampling only from the k most likely next tokens.", INF),
    "quantization": _E("Storing numbers with fewer bits (say 8 or 4 instead of 16 or 32), which shrinks memory with a small loss of precision.", INF),
    "speculative decoding": _E("A small fast model drafts several tokens, and the big model checks them all in one pass, keeping the ones it agrees with.", INF),
    "prompt caching": _E("Reusing the processed form of a prompt prefix that repeats across requests, which cuts cost and time to first token.", INF),
    "cross-entropy": _E("The standard loss for predicting categories: minus the log of the probability given to the right answer.", LOSS),
    "perplexity": _E("e raised to the average cross-entropy: roughly how many options the model is torn between at each step.", LOSS),
    "contrastive loss": _E("A loss that pulls matching pairs together in vector space and pushes non-matching pairs apart.", LOSS),
    "infonce": _E("A contrastive loss that treats the other items in the batch as wrong answers for each matching pair.", LOSS),
    "overfitting": _E("When a model memorizes its training data, noise included, and does worse on new data.", REG),
    "underfitting": _E("When a model is too simple, or undertrained, to capture the pattern at all.", REG),
    "regularization": _E("Any technique that discourages memorizing, such as dropout, weight decay or early stopping.", REG),
    "dropout": _E("Randomly switching off neurons during training so the network can't rely on any single path.", REG),
    "weight decay": _E("Shrinking every weight slightly at each step, which penalizes large weights and keeps the model smoother.", REG),
    "early stopping": _E("Stopping training when the score on held-out data stops improving.", REG),
    "data leakage": _E("When information from the test set, or from the future, sneaks into training and inflates offline scores.", REG),
    "convolution": _E("Sliding a small grid of learned weights across an image and computing a weighted sum at every position, to find where a pattern appears.", CNN),
    "pooling": _E("Shrinking a feature map by keeping only the strongest (or average) value in each small window.", CNN),
    "rnn": _E("A recurrent neural network: it reads a sequence one step at a time, carrying a running summary called the hidden state.", CNN),
    "lstm": _E("An RNN with gates that decide what to forget, what to write and what to output, so it can remember across long sequences.", CNN),

    # --- metrics --------------------------------------------------------------
    "precision": _E("Of the items flagged, the fraction that were right.", MET),
    'recall': _E("Of the items that truly mattered, the fraction that were found.", MET, scope=('primer/ml/metrics', 'primer/ml/embeddings/', 'primer/agents/rag', 'primer/agents/evals')),
    "f1": _E("The harmonic mean of precision and recall: high only when both are high.", MET),
    "roc-auc": _E("The probability that the model ranks a random positive above a random negative.", MET),
    "recall@k": _E("The fraction of relevant documents that appear in the top k results. For RAG, usually the metric that matters most.", MET),
    "mrr": _E("Mean reciprocal rank: the average of 1 / (position of the first relevant result).", MET),
    "ndcg": _E("A ranking score that gives more credit for relevant results near the top and handles degrees of relevance.", MET),
    "bleu": _E("A score that counts overlapping word sequences with a reference text. Cheap, but blind to paraphrase.", MET),

    # --- embeddings -----------------------------------------------------------
    "embedding": _E("A learned vector for a piece of content, arranged so that similar meanings end up close together.", W2V),
    "cosine similarity": _E("How closely two vectors point in the same direction, from −1 (opposite) to 1 (identical), ignoring their lengths.", SIM),
    "euclidean distance": _E("The straight-line distance between two vectors.", SIM),
    "anisotropy": _E("When a model's vectors all crowd into a narrow cone, so even unrelated texts score as fairly similar.", SIM),
    "curse of dimensionality": _E("In very high dimensions, random points are all nearly the same distance apart, which makes exact nearest-neighbor search slow and fragile.", SIM),
    "hard negative": _E("A training example that looks relevant but isn't, such as the right topic with the wrong answer. Training on them teaches fine distinctions.", CON),
    "bi-encoder": _E("Embeds the query and each document separately, so document vectors can be computed once and searched fast.", RET),
    "cross-encoder": _E("Reads the query and one document together and outputs a relevance score. Accurate but slow, so it is used to rerank a shortlist.", RET),
    "reranker": _E("A second, more accurate model that reorders the top results of a fast first-stage search.", RET),
    "clip": _E("A model that trains an image encoder and a text encoder together so pictures and their captions land near each other in one vector space.", CON),
    "matryoshka embedding": _E("An embedding trained so its first few dimensions work as a smaller embedding on their own.", CMP),
    "approximate nearest neighbor": _E("Finding vectors close to a query quickly by searching only part of the collection, accepting a small chance of missing the true closest.", ANN),
    "ann": _E("Approximate nearest neighbor search: trade a little recall for a lot of speed.", ANN),
    "hnsw": _E("A layered graph index for vector search: long jumps on sparse upper layers, then a careful local search at the bottom.", ANN),
    "ivf": _E("A vector index that clusters vectors ahead of time and searches only the clusters nearest the query.", ANN),
    "product quantization": _E("Compressing a vector by splitting it into chunks and replacing each chunk with the ID of its nearest entry in a small codebook.", ANN),
    "bm25": _E("The classic keyword-search score: rewards documents containing the query's rarer words, with diminishing returns for repeats.", RET),
    "hybrid search": _E("Running keyword search and vector search together and merging the two ranked lists.", RET),
    "reciprocal rank fusion": _E("Merging ranked lists by giving each document 1 / (k + rank) from every list it appears in, then adding those up.", RET),
    "chunking": _E("Splitting documents into passages before embedding them, so retrieval can return just the relevant part.", RET),
    "colbert": _E("A retrieval model that keeps one vector per token and matches each query token to its best document token.", RET),
    "k-means": _E("Clustering by repeatedly assigning each point to its nearest center and moving each center to the average of its points.", CLU),
    "semantic cache": _E("Reusing a stored answer when a new question's embedding is nearly identical to a previous question's.", CLU),

    # --- applied AI -----------------------------------------------------------
    "llm": _E("Large language model: a transformer trained to predict the next token, then tuned to follow instructions.", LLM),
    "prompt": _E("The text sent to a language model: instructions, context and the question.", LLM),
    "system prompt": _E("Standing instructions sent before the conversation that set a model's role, rules and style.", LLM),
    "tool calling": _E("The model outputs a structured request to run a function; your code runs it and sends the result back. The model never executes anything itself.", LLM),
    "agent": _E("A system where a language model decides which tools to call, looks at the results and decides the next step, in a loop until done.", LOOP),
    "agent loop": _E("The cycle an agent repeats: think, call a tool, observe the result, decide what next.", LOOP),
    "react": _E("Reason plus act: the agent pattern that interleaves reasoning steps with tool calls.", LOOP),
    "workflow": _E("A fixed sequence of steps written in code, with a language model doing a task inside each step.", ORC),
    "router": _E("A step that classifies a request and sends it down the right path, or to the right model.", ORC),
    "multi-agent system": _E("Several agents that coordinate, typically a supervisor delegating to specialist workers.", ORC),
    "json schema": _E("A standard way to describe the shape of JSON data: which fields exist, their types and which are required.", TOOLS),
    "idempotent": _E("Safe to repeat: doing it twice has the same effect as doing it once, so retries can't create duplicates.", TOOLS),
    "mcp": _E("Model Context Protocol: an open standard for connecting AI apps to tools and data through reusable servers.", MCP),
    "rag": _E("Retrieval-augmented generation: fetch relevant passages first, then have the model answer using them, with citations.", RAG),
    "hyde": _E("Hypothetical document embeddings: have the model write a plausible answer, then search with that, since answers resemble documents more than questions do.", RAG),
    "grounding": _E("Tying an answer to specific source passages so every claim can be checked.", RAG),
    "hallucination": _E("A confident answer that isn't supported by the sources or the facts.", RAG),
    "context engineering": _E("Deciding exactly what goes into the model's context window on each call: instructions, facts, tool results, memory.", CTX),
    "context rot": _E("Quality dropping as a long session fills the context with stale or irrelevant material.", CTX),
    "episodic memory": _E("Memory of what happened, such as 'last week the user rejected this vendor'.", MEM),
    "semantic memory": _E("Memory of facts, such as 'the user's fiscal year starts in April'.", MEM),
    "procedural memory": _E("Memory of how to do things, such as a learned workflow or saved skill.", MEM),
    "plan-and-execute": _E("An agent pattern that writes a plan first, then executes it step by step, replanning when results surprise it.", PLAN),
    "reflection": _E("Having a model review and revise its own output. Useful, but external checks such as tests are more reliable.", PLAN),
    "compounding error": _E("Small per-step failure rates multiplying over many steps: ten steps at 95% succeed only about 60% of the time.", PLAN),
    "eval": _E("A repeatable test of a model or agent's behavior on a fixed set of tasks with known good outcomes.", EVAL),
    "golden set": _E("A curated set of real tasks with expected outcomes, run on every change to catch regressions.", EVAL),
    "llm-as-judge": _E("Using a language model with a rubric to grade open-ended outputs, checked against human ratings.", EVAL),
    "guardrail": _E("A check around the model that screens inputs, validates outputs or limits actions.", GUARD),
    "prompt injection": _E("Hostile instructions hidden in content the model reads, such as an email or web page, trying to hijack it.", GUARD),
    "pii": _E("Personally identifiable information: names, emails, phone numbers, card numbers and similar.", GUARD),
    "model routing": _E("Sending each request to the cheapest model that can handle it well.", COST),
    'trace': _E("A record of one run as a tree of steps (model calls, tool calls, retrievals) with inputs, outputs, timing and cost.", OBS, scope=('primer/agents/observability', 'primer/agents/evals')),
    'span': _E("One step inside a trace, with its start time, duration and details.", OBS, scope=('primer/agents/observability',)),
    "shadow mode": _E("Running a new system on real inputs and recording what it would do, without letting it act.", DEP),
    "canary release": _E("Sending a small share of traffic to a new version first and watching its metrics before rolling out further.", DEP),
    "kill switch": _E("A way to stop an agent instantly, per customer or globally.", DEP),
    "audit trail": _E("A tamper-evident log of which agent did what, for whom and with what authority.", DEP),
    # --- added from lesson reports -----------------------------------------
    'accuracy': _E('The share of all predictions that were correct. Misleading when one class is rare.', 'primer.ml.metrics'),
    'confusion matrix': _E('The four counts behind every classification metric: hits, false alarms, misses and correct passes.', 'primer.ml.metrics'),
    'roc curve': _E('True-positive rate plotted against false-positive rate as the decision threshold sweeps.', 'primer.ml.metrics'),
    'true-positive rate': _E('The share of real positives the model flags; the same as recall.', 'primer.ml.metrics'),
    'false-positive rate': _E('The share of real negatives the model wrongly flags.', 'primer.ml.metrics'),
    'precision-recall curve': _E('Precision plotted against recall across thresholds; more honest than ROC for rare events.', 'primer.ml.metrics'),
    'precision@k': _E('The share of the top k search results that are relevant.', 'primer.ml.metrics'),
    'rouge-l': _E('An overlap score based on the longest in-order sequence of words shared with a reference.', 'primer.ml.metrics'),
    'bertscore': _E('Compares generated and reference text by embedding similarity instead of exact words.', 'primer.ml.metrics'),
    "cohen's kappa": _E('Agreement between two raters after subtracting the agreement expected by chance.', 'primer.ml.metrics'),
    'loss mask': _E('A per-position switch deciding which predictions count toward the loss, such as only the reply in SFT.', 'primer.ml.training_stages'),
    'preference tuning': _E('Training on which of two responses people preferred, to shape tone, helpfulness and safety.', 'primer.ml.training_stages'),
    'bradley-terry model': _E('The rule that the probability A beats B is the sigmoid of their score difference.', 'primer.ml.training_stages'),
    'reference model': _E('A frozen copy of the starting model that DPO and RLHF measure drift against.', 'primer.ml.training_stages'),
    'policy': _E('In preference tuning, the model being trained, viewed as a probability distribution over responses.', 'primer.ml.training_stages', scope=('primer/ml/training_stages',)),
    'implicit reward': _E('In DPO, how much more likely training has made a response than the reference model did.', 'primer.ml.training_stages'),
    'log-probability': _E("The logarithm of a probability; for a whole response, the sum of its tokens' log-probabilities.", 'primer.ml.training_stages'),
    'qlora': _E('LoRA adapters trained on top of base weights stored in 4 bits.', 'primer.ml.training_stages'),
    'adapter': _E('A small trainable module added beside frozen weights to specialise a model.', 'primer.ml.training_stages', scope=('primer/ml/training_stages',)),
    'full fine-tune': _E('Updating every weight of a model. Rarely worth the cost.', 'primer.ml.training_stages'),
    'soft targets': _E("A teacher model's temperature-softened probabilities, used to train a student model.", 'primer.ml.training_stages'),
    'kl divergence': _E('The extra surprise from using distribution q when the truth is p; zero only when they match.', 'primer.ml.training_stages'),
    'dark knowledge': _E('How a teacher model ranks the wrong answers, revealed by its soft targets.', 'primer.ml.training_stages'),
    'arithmetic intensity': _E('Operations performed per byte read from memory.', 'primer.ml.inference'),
    'roofline': _E('A chart of the speed a chip can reach at each arithmetic intensity, capped first by memory and then by compute.', 'primer.ml.inference'),
    'memory-bound': _E('Limited by how fast data arrives from memory rather than by arithmetic, as in decoding.', 'primer.ml.inference'),
    'compute-bound': _E('Limited by arithmetic throughput, as in prefill.', 'primer.ml.inference'),
    'draft model': _E('The small, fast model that proposes tokens in speculative decoding.', 'primer.ml.inference'),
    'acceptance rate': _E('How often the big model keeps a drafted token in speculative decoding.', 'primer.ml.inference'),
    'int8 quantization': _E('Storing each weight as an 8-bit integer times a shared scale.', 'primer.ml.inference'),
    'int4 quantization': _E('Storing each weight as a 4-bit integer times a shared scale.', 'primer.ml.inference'),
    'per-channel quantization': _E("One scale per weight row, so a single outlier doesn't coarsen all the others.", 'primer.ml.inference'),
    'static batching': _E('Serving a fixed group of requests until the longest one finishes.', 'primer.ml.inference'),
    'continuous batching': _E("Refilling a finished request's slot in the batch immediately with the next request.", 'primer.ml.inference'),
    'pagedattention': _E('Storing the KV cache in fixed-size pages, like virtual memory, to avoid wasted GPU memory.', 'primer.ml.inference'),
    'filter': _E('In a CNN, the small grid of learned weights a convolution slides over an image; also called a kernel.', 'primer.ml.cnn_rnn', scope=('primer/ml/cnn_rnn',)),
    'kernel': _E('In a CNN, the small grid of learned weights a convolution slides over an image; also called a filter.', 'primer.ml.cnn_rnn', scope=('primer/ml/cnn_rnn',)),
    'stride': _E('How many pixels a convolution filter jumps between positions.', 'primer.ml.cnn_rnn', scope=('primer/ml/cnn_rnn',)),
    'padding': _E('Zeros added around an image so a filter can centre on the edge pixels.', 'primer.ml.cnn_rnn', scope=('primer/ml/cnn_rnn',)),
    'max pooling': _E('Keeping only the largest value in each small block of a feature map.', 'primer.ml.cnn_rnn'),
    'weight sharing': _E("Reusing the same filter at every position, so the parameter count doesn't grow with image size.", 'primer.ml.cnn_rnn'),
    'receptive field': _E('How much of the original input one neuron can see.', 'primer.ml.cnn_rnn'),
    'vision transformer': _E('A transformer that treats small image patches as tokens.', 'primer.ml.cnn_rnn'),
    'patch': _E('A small square cut from an image and flattened into one token.', 'primer.ml.cnn_rnn', scope=('primer/ml/cnn_rnn',)),
    'backpropagation through time': _E('Backpropagation applied to an RNN unrolled across its time steps.', 'primer.ml.cnn_rnn'),
    'cell state': _E("An LSTM's notebook, edited by adding rather than overwriting, so information survives many steps.", 'primer.ml.cnn_rnn'),
    'forget gate': _E('The LSTM dial that decides what to erase from the cell state.', 'primer.ml.cnn_rnn'),
    'input gate': _E('The LSTM dial that decides what new information to write into the cell state.', 'primer.ml.cnn_rnn'),
    'output gate': _E('The LSTM dial that decides how much of the cell state to show as output.', 'primer.ml.cnn_rnn'),
    'gru': _E('A simpler gated RNN with an update gate and a reset gate and no separate cell state.', 'primer.ml.cnn_rnn'),
    'state-space model': _E('A recurrent-style model, such as Mamba, that trains in parallel and runs in time linear in sequence length.', 'primer.ml.cnn_rnn'),
    'mamba': _E('A state-space model that trains in parallel and runs in time linear in sequence length.', 'primer.ml.cnn_rnn'),
    'tanh': _E('Squashes any number into the range −1 to 1.', 'primer.ml.cnn_rnn'),
    'tool call': _E('A structured request from the model to run one of your functions with specific arguments; your code decides whether to run it.', 'primer.agents.tools'),
    'tool_result': _E("The message block that returns a tool's output or error to the model, matched to its request by id.", 'primer.agents.agent_loop'),
    'stop reason': _E('Why a model response ended: finished, wants a tool, hit the length limit, or declined.', 'primer.agents.agent_loop'),
    'loop detection': _E('Stopping an agent that keeps requesting the same tool with the same arguments.', 'primer.agents.agent_loop'),
    'token budget': _E('A cap on the tokens, and so the money, one agent run may spend.', 'primer.agents.agent_loop'),
    'parallel tool calls': _E('Several tool requests in one model turn, run at the same time, with all results returned in one message.', 'primer.agents.agent_loop'),
    'strict tool use': _E("An API setting that guarantees the model's tool arguments match the tool's JSON Schema exactly.", 'primer.agents.tools'),
    'semantic validation': _E('Checking that well-formed arguments are also true, such as that a customer id actually exists.', 'primer.agents.tools'),
    'idempotency key': _E('A unique id sent with a write so a retried request returns the first result instead of acting twice.', 'primer.agents.tools'),
    'dry run': _E('Running every check for an action and describing it without actually doing it.', 'primer.agents.tools'),
    'human approval gate': _E('A rule that parks irreversible or high-value actions until a person approves them.', 'primer.agents.tools'),
    'least privilege': _E('Giving each agent or tool only the permissions its job needs.', 'primer.agents.tools'),
    'dynamic tool loading': _E('Sending the model only the few tool definitions most relevant to the current request.', 'primer.agents.tools'),
    'model context protocol': _E('An open standard that lets any AI application connect to any tool server the same way.', 'primer.agents.mcp'),
    'json-rpc': _E('A simple convention for calling functions on another program by exchanging JSON requests, replies and notifications.', 'primer.agents.mcp'),
    'stdio transport': _E('Running a server as a child process and exchanging one JSON message per line over its input and output.', 'primer.agents.mcp'),
    'mcp server': _E('The program that wraps a real system (files, a database, an API) and offers it to AI apps over MCP.', 'primer.agents.mcp'),
    'mcp resource': _E('Read-only data an MCP server offers for the app to load into context, addressed by a URI.', 'primer.agents.mcp'),
    'capability negotiation': _E('The start-of-session exchange where client and server announce which optional features they support.', 'primer.agents.mcp'),
    'tool poisoning': _E("Hiding instructions for the model inside a tool's description.", 'primer.agents.mcp'),
    'rug pull': _E('A tool server changing its definitions after they were approved.', 'primer.agents.mcp'),
    'confused deputy': _E('A program with broad authority tricked into using it for someone who lacks that authority.', 'primer.agents.mcp'),
    'oauth': _E('A standard way for a user to grant an app limited, revocable access without sharing their password.', 'primer.agents.mcp'),
    'decomposition': _E('Splitting a big goal into small subtasks, each with a checkable definition of done.', 'primer.agents.planning'),
    'external verification': _E("Checking a model's output with something outside the model, such as tests, schemas or database queries.", 'primer.agents.planning'),
    'checkpoint': _E('Saved progress after a completed step, so a failure resumes from there instead of from the start.', 'primer.agents.planning', scope=('primer/agents/planning', 'primer/agents/orchestration')),
    'durable execution': _E("Running a workflow so its progress survives crashes, by storing each completed step's result.", 'primer.agents.orchestration'),
    'state machine': _E('A fixed set of states and allowed transitions, with code deciding every move.', 'primer.agents.orchestration'),
    'short-term memory': _E('The current conversation, kept within a token budget: recent turns verbatim, older ones summarized.', 'primer.agents.memory'),
    'long-term memory': _E('Facts, events and procedures stored outside the model and recalled into later conversations.', 'primer.agents.memory'),
    'multi-tenant': _E('One system serving many separate customers whose data must never mix.', 'primer.agents.memory'),
    'tenant isolation': _E("Partitioning storage so a request can only ever reach its own tenant's and user's data.", 'primer.agents.memory'),
    'right to erasure': _E("A user's legal right, for example under GDPR, to have their personal data deleted.", 'primer.agents.memory'),
    'prompt chaining': _E('A fixed sequence of model calls where each output feeds the next, with code checks in between.', 'primer.agents.orchestration'),
    'routing': _E('One model call classifies a request and sends it to a specialised handler.', 'primer.agents.orchestration', scope=('primer/agents/orchestration', 'primer/agents/cost')),
    'orchestrator-workers': _E('One model decides the subtasks, workers handle them, and one model combines the results.', 'primer.agents.orchestration'),
    'evaluator-optimizer': _E('A generator drafts and an evaluator critiques, repeated until the draft passes or the rounds run out.', 'primer.agents.orchestration'),
    'supervisor': _E('In a multi-agent system, the agent that assigns tasks to specialist agents and assembles their answers.', 'primer.agents.orchestration', scope=('primer/agents/orchestration',)),
    'hand-off': _E('The message that passes work and its context from one agent to another; anything not written into it is lost.', 'primer.agents.orchestration', scope=('primer/agents/orchestration', 'primer/agents/agent_loop')),
    'stochastic gradient descent': _E("Gradient descent where each step's slope is estimated from a small random batch instead of the whole dataset.", 'primer.ml.optimizers'),
    'learning-rate schedule': _E('A rule that changes the learning rate during training, such as warming up and then decaying along a cosine curve.', 'primer.ml.optimizers'),
    'cosine decay': _E('Lowering the learning rate from its peak to a floor along half a cosine wave.', 'primer.ml.optimizers'),
    'bias correction': _E("Adam's rescaling of its running averages early in training, since they start at zero and would otherwise be too small.", 'primer.ml.optimizers'),
    'initialization': _E("Choosing the size of a network's random starting weights so signals neither shrink nor grow layer by layer.", 'primer.ml.deep_nets', scope=('primer/ml/deep_nets', 'primer/ml/neural_net')),
    'xavier initialization': _E('Starting weights with variance 2 / (fan-in + fan-out), suited to tanh and sigmoid layers.', 'primer.ml.deep_nets'),
    'he initialization': _E('Starting weights with variance 2 / fan-in, making up for ReLU zeroing half its inputs.', 'primer.ml.deep_nets'),
    'fan-in': _E('The number of inputs a neuron adds up.', 'primer.ml.deep_nets'),
    'saturation': _E('When an activation like sigmoid or tanh sits on its flat tail, so almost no gradient passes through.', 'primer.ml.neural_net', scope=('primer/ml/neural_net', 'primer/ml/deep_nets', 'primer/ml/attention')),
    'dying relu': _E('A ReLU neuron whose input stays negative, so it outputs zero and receives zero gradient forever.', 'primer.ml.neural_net'),
    'one-hot': _E('A vector with a 1 at the correct class and 0 everywhere else.', 'primer.ml.losses'),
    'log-sum-exp': _E('Computing the log of a sum of exponentials without overflow, by factoring out the largest term first.', 'primer.ml.losses'),
    'binary cross-entropy': _E('Cross-entropy for yes/no predictions: −[y·ln p + (1 − y)·ln(1 − p)].', 'primer.ml.losses'),
    'mean squared error': _E('The average of squared prediction errors, so big misses dominate.', 'primer.ml.losses'),
    'mean absolute error': _E('The average of absolute prediction errors, which is robust to outliers.', 'primer.ml.losses'),
    'label smoothing': _E('Training against a softened target, such as 0.9 on the right class and the rest spread evenly, to discourage over-confidence.', 'primer.ml.losses'),
    'bias-variance trade-off': _E("Splitting a model's error into systematic error from being too simple (bias) and sensitivity to the training sample from being too flexible (variance).", 'primer.ml.regularization'),
    'l1 regularization': _E('Penalizing the sum of absolute weights, which drives unneeded weights to exactly zero.', 'primer.ml.regularization'),
    'l2 regularization': _E('Penalizing the sum of squared weights, which shrinks all weights toward zero without eliminating them.', 'primer.ml.regularization'),
    'soft thresholding': _E('Moving each weight a fixed amount toward zero and snapping any weight within that amount to exactly zero.', 'primer.ml.regularization'),
    'validation set': _E('Held-out data used to tune choices like model size and when to stop training.', 'primer.ml.regularization'),
    'test set': _E('Held-out data used once at the end to estimate real-world performance honestly.', 'primer.ml.regularization', scope=('primer/ml/',)),
    'cross-validation': _E('Rotating which slice of the data is held out, training once per slice, and averaging the scores.', 'primer.ml.regularization'),
    'double descent': _E('The observation that past a point, even larger models generalize better again.', 'primer.ml.regularization'),
    'benchmark contamination': _E("When test questions appeared in a model's training data, inflating its scores.", 'primer.ml.regularization'),
    'gpu': _E('A graphics processor: a chip with thousands of small cores that do many multiply-adds at once, which is exactly what neural networks need.', 'primer.notation'),
    'parallelism': _E('Doing many pieces of work at the same time instead of one after another.', 'primer.ml.attention'),
    'hidden state': _E('The running summary an RNN carries from one step to the next.', 'primer.ml.cnn_rnn'),
    'auto-regressive': _E('Generating one token at a time, each predicted from everything before it and then fed back in.', 'primer.ml.big_picture'),
    'beam search': _E('Decoding that keeps the few most probable partial outputs at each step instead of just one, then picks the best finished one.', 'primer.ml.inference'),
    'checkpoint averaging': _E('Averaging the weights saved at the last few points of training, which usually gives a slightly better model.', 'primer.ml.training_stages'),
    'encoder-decoder attention': _E("Attention where the decoder's queries look at the encoder's keys and values, so each output word can consult the whole input.", 'primer.ml.transformer'),
    'luhn check': _E('A checksum every real card number satisfies, used to tell card numbers from look-alike IDs.', 'primer.agents.guardrails'),
    'named-entity recognition': _E('A model that tags names, places and organisations in text.', 'primer.agents.guardrails'),
    'ner': _E('Named-entity recognition: tagging names, places and organisations in text.', 'primer.agents.guardrails'),
    'entailment': _E('Whether one text logically follows from another, checked by a model trained for it.', 'primer.agents.guardrails'),
    'privilege separation': _E("Splitting work so the part that reads untrusted content can't take dangerous actions.", 'primer.agents.guardrails'),
    'lethal trifecta': _E('Private data, untrusted content and a way to send data out, all in one agent: together they allow data theft.', 'primer.agents.guardrails'),
    'lost in the middle': _E('Models use information at the start and end of a long input more reliably than information in the middle.', 'primer.agents.context'),
    'sandwich ordering': _E('Placing the best retrieved chunks at the start and end of the context and the weakest in the middle.', 'primer.agents.context'),
    'escaping': _E("Replacing characters like < and > so data can't pose as markup or instructions.", 'primer.agents.context'),
    'trajectory': _E('The full record of an agent run: its answer, its tool calls with arguments, and the end state.', 'primer.agents.evals'),
    'code grader': _E("An automatic check written in code, such as exact match, schema or database state, that scores an agent's output.", 'primer.agents.evals'),
    'rubric': _E('A short list of explicit pass/fail criteria given to a grader.', 'primer.agents.evals'),
    'regression': _E('A task that used to pass and now fails after a change.', 'primer.agents.evals'),
    'p95': _E('The 95th percentile: the value that 95% of measurements are at or below.', 'primer.agents.evals'),
    'continuous integration': _E('The automated checks that run on every proposed change before it can merge.', 'primer.agents.evals'),
    'drift': _E('A metric creeping away from its usual level over time or after a deploy.', 'primer.agents.evals'),
    'faithfulness': _E("The share of an answer's claims that are supported by the retrieved sources.", 'primer.agents.evals'),
    'cost per successful task': _E('Total spend divided by the number of tasks that succeeded, counting retries and cleanup.', 'primer.agents.cost'),
    'ttl': _E('Time to live: how long a cached entry may be served before it is treated as expired.', 'primer.agents.cost'),
    'batch api': _E('A provider interface that processes many requests asynchronously at a discount.', 'primer.agents.cost'),
    'task budget': _E('Hard limits on steps and tokens for one agent task.', 'primer.agents.cost'),
    'tenant': _E('One customer organisation on a shared platform.', 'primer.agents.cost'),
    'anomaly alert': _E('An alert when a metric jumps far outside its usual range.', 'primer.agents.cost'),
    'instrumentation': _E('Code that records spans or metrics around the work a program does.', 'primer.agents.observability'),
    'opentelemetry': _E('The open standard for traces, metrics and logs.', 'primer.agents.observability'),
    'genai semantic conventions': _E("OpenTelemetry's standard attribute names for model and agent spans, such as gen_ai.request.model.", 'primer.agents.observability'),
    'otlp': _E('The OpenTelemetry Protocol: the wire format exporters use to ship telemetry.', 'primer.agents.observability'),
    'collector': _E('In OpenTelemetry, the service that receives spans from apps and forwards them to storage backends.', 'primer.agents.observability'),
    'graduated autonomy': _E('Granting an agent more independence in steps (shadow, then approval, then autonomy), each earned with evidence.', 'primer.agents.deployment'),
    'content-addressed': _E('Named by a hash of its own content, so identical content always gets the identical name.', 'primer.agents.deployment'),
    'sha-256': _E('A hash function: a fixed-length fingerprint of data that changes completely if the data changes at all.', 'primer.agents.deployment'),
    'feature flag': _E('A configuration switch that turns a capability on or off without redeploying.', 'primer.agents.deployment'),
    'rate limit': _E('A cap on how many actions may happen per unit of time.', 'primer.agents.deployment'),
    'token bucket': _E('A rate limiter that allows bursts up to a capacity and refills at a steady rate.', 'primer.agents.deployment'),
    'blast radius': _E('How much damage a failure can do before it is noticed and stopped.', 'primer.agents.deployment'),
    'audit log': _E('A record of which actor took which action, for whom, with what inputs.', 'primer.agents.deployment'),
    'hash chain': _E("Records that each store the previous record's hash, so any edit or deletion is detectable.", 'primer.agents.deployment'),
    'erp': _E("Enterprise resource planning: a company's finance and operations system of record.", 'primer.agents.deployment'),
    'exponential backoff': _E('Waiting longer after each failed attempt, doubling up to a cap.', 'primer.agents.failures'),
    'jitter': _E("Random variation added to retry delays so clients don't all retry at the same moment.", 'primer.agents.failures'),
    'circuit breaker': _E('A wrapper that stops calling a failing dependency for a cool-down period, failing fast instead.', 'primer.agents.failures'),
    'contract test': _E('An automated check that an external API still accepts and returns what your code expects.', 'primer.agents.failures'),
    'ocr': _E('Optical character recognition: reading text from an image of a page.', 'primer.agents.rag'),
    'acl': _E('Access-control list: the users or groups allowed to read a document.', 'primer.agents.rag'),
    'permission-aware retrieval': _E('Filtering out documents a user may not read before anything is ranked or shown to the model.', 'primer.agents.rag'),
    'dense retrieval': _E('Search by comparing embedding vectors, so matches are by meaning rather than exact words.', 'primer.agents.rag'),
    'query rewriting': _E('Turning a follow-up or vague question into a standalone search query.', 'primer.agents.rag'),
    'multi-query retrieval': _E('Searching several phrasings of a question and fusing the results.', 'primer.agents.rag'),
    'contextual retrieval': _E('Indexing each chunk together with a line saying where in its document it comes from.', 'primer.agents.rag'),
    'parent-child retrieval': _E('Matching small chunks but returning the larger section around them.', 'primer.agents.rag'),
    'agentic rag': _E('RAG where the model decides whether, what and how often to search.', 'primer.agents.rag'),
    'graphrag': _E('Retrieval over a graph of entities and relationships extracted from documents.', 'primer.agents.rag'),
    'centroid': _E("The average position of a group of vectors, used as the group's representative point.", 'primer.ml.embeddings.ann'),
    'voronoi cell': _E('All the points closer to one centroid than to any other: the section an IVF index searches.', 'primer.ml.embeddings.ann'),
    'nprobe': _E('How many IVF clusters a query scans; raising it trades speed for recall.', 'primer.ml.embeddings.ann'),
    'efsearch': _E("HNSW's query-time beam width; raising it trades speed for recall.", 'primer.ml.embeddings.ann'),
    'efconstruction': _E("HNSW's beam width while building the graph; higher means a better graph and a slower build.", 'primer.ml.embeddings.ann'),
    'greedy search': _E('Always move to whichever neighbour is closest to the target, and stop when none is closer.', 'primer.ml.embeddings.ann'),
    'graph': _E('A set of points (nodes) joined by links (edges).', 'primer.ml.embeddings.ann'),
    'codebook': _E("Product quantization's small catalogue of representative sub-vectors; each piece of a vector is stored as the number of its nearest entry.", 'primer.ml.embeddings.ann'),
    'asymmetric distance computation': _E('Scoring compressed vectors against an uncompressed query by adding up precomputed table lookups.', 'primer.ml.embeddings.ann'),
    'ivf-pq': _E('IVF clustering combined with compressed offsets from each cluster centre: the standard index for billion-scale search.', 'primer.ml.embeddings.ann'),
    're-scoring': _E('Recomputing exact scores for a shortlist that an approximate index returned, to recover accuracy.', 'primer.ml.embeddings.ann'),
    'flat index': _E('Exact search that compares the query with every stored vector; the ground truth other indexes are measured against.', 'primer.ml.embeddings.ann'),
    'idf': _E("Inverse document frequency: a word's rarity weight, higher for words found in fewer documents.", 'primer.ml.embeddings.retrieval'),
    'term frequency': _E('How many times a word appears in a document.', 'primer.ml.embeddings.retrieval'),
    'length normalization': _E("BM25's discount for mentions in documents longer than average, controlled by b.", 'primer.ml.embeddings.retrieval'),
    'late interaction': _E('Keeping one vector per word and matching each query word to its best document word.', 'primer.ml.embeddings.retrieval'),
    'maxsim': _E('For each query word, its highest similarity to any document word, summed over the query.', 'primer.ml.embeddings.retrieval'),
    'query and passage prefixes': _E('Labels some embedding models expect on questions and documents; forgetting one silently lowers recall.', 'primer.ml.embeddings.retrieval'),
    'shortlist': _E('The small set of top candidates from a cheap first-stage search that a slower, more precise stage reorders.', 'primer.ml.embeddings.retrieval'),
    'l2 normalization': _E('Dividing a vector by its own length so it has length 1.', 'primer.ml.embeddings.similarity'),
    'dimension': _E('One of the numbers in a vector: a 768-dimension embedding is a list of 768 numbers.', 'primer.ml.embeddings.similarity'),
    'mean-centering': _E('Subtracting the average vector of a collection from every vector, removing the direction they all share.', 'primer.ml.embeddings.similarity'),
    'whitening': _E('Rescaling vectors so every direction has equal spread and no two directions are correlated.', 'primer.ml.embeddings.similarity'),
    'threshold calibration': _E('Choosing a similarity cut-off by measuring precision and recall on labeled pairs for one specific model.', 'primer.ml.embeddings.similarity'),
    'word2vec': _E('A 2013 method that learns one vector per word by predicting nearby words.', 'primer.ml.embeddings.word2vec'),
    'skip-gram': _E("The word2vec task of predicting each word's neighbours from the word itself.", 'primer.ml.embeddings.word2vec'),
    'negative sampling': _E('Training by scoring the true pair up and a few random noise pairs down, instead of scoring the whole vocabulary.', 'primer.ml.embeddings.word2vec'),
    'pmi': _E('Pointwise mutual information: the log of how much more often two words appear together than chance predicts.', 'primer.ml.embeddings.word2vec'),
    'glove': _E('A counting method that fits word vectors so their dot products predict how often words appear together.', 'primer.ml.embeddings.word2vec'),
    'svd': _E('Singular value decomposition: splits a table into a few directions that capture its main patterns, used to compress it.', 'primer.ml.embeddings.word2vec'),
    'static embedding': _E('One fixed vector per word, whatever the sentence.', 'primer.ml.embeddings.word2vec'),
    'contextual embedding': _E('A vector computed for a word within its sentence, so the same word gets different vectors in different contexts.', 'primer.ml.embeddings.word2vec'),
    'contrastive learning': _E('Training that pulls matching pairs together and pushes non-matching pairs apart.', 'primer.ml.embeddings.contrastive'),
    'in-batch negatives': _E('Using the other examples in a training batch as free wrong answers for each query.', 'primer.ml.embeddings.contrastive'),
    'bag of words': _E('Representing text by counting how many times each word appears, ignoring order.', 'primer.ml.embeddings.contrastive'),
    'zero-shot classification': _E('Labeling items by comparing them to a text description of each label, with no training on those labels.', 'primer.ml.embeddings.contrastive'),
    'scalar quantization': _E('Storing each number as one of 256 levels (one byte) instead of a 4-byte float.', 'primer.ml.embeddings.compression'),
    'binary quantization': _E('Storing only the sign of each number, as a single bit.', 'primer.ml.embeddings.compression'),
    'hamming distance': _E('The number of positions where two bit strings differ.', 'primer.ml.embeddings.compression'),
    'popcount': _E('Counting the 1-bits in a number; combined with XOR it computes Hamming distance in one step.', 'primer.ml.embeddings.compression'),
    'two-stage retrieval': _E('A cheap, wide first pass to shortlist candidates, then an expensive, precise pass over only those.', 'primer.ml.embeddings.compression'),
    'hit@k': _E('The share of queries with at least one relevant result in the top k.', 'primer.ml.embeddings.operations'),
    'pca': _E('Principal component analysis: finding the directions along which data varies most, to draw or compress it with fewer numbers.', 'primer.ml.embeddings.clustering'),
    'clustering': _E('Grouping items so similar ones end up together, without being told the groups.', 'primer.ml.embeddings.clustering'),
    'inertia': _E("The total squared distance from each point to its cluster's center: the quantity k-means minimizes.", 'primer.ml.embeddings.clustering'),
    'k-means++': _E('A way to pick k-means starting centers spread far apart, which avoids bad results.', 'primer.ml.embeddings.clustering'),
    'silhouette score': _E('How much closer each point is to its own cluster than to the nearest other one, from −1 to 1.', 'primer.ml.embeddings.clustering'),
    'elbow method': _E('Choosing the number of clusters where adding more stops reducing inertia much.', 'primer.ml.embeddings.clustering'),
    'dbscan': _E('Density-based clustering that grows clusters from points with enough close neighbours and labels the rest as noise.', 'primer.ml.embeddings.clustering'),
    'hdbscan': _E('A version of DBSCAN that considers every reach at once and keeps the most persistent clusters, so no reach setting is needed.', 'primer.ml.embeddings.clustering'),
    'umap': _E("A projection to 2-D that keeps each point's neighbours close but distorts other distances.", 'primer.ml.embeddings.clustering'),
    't-sne': _E('An older neighbour-preserving 2-D projection; cluster sizes and gaps in its plots are not meaningful.', 'primer.ml.embeddings.clustering'),
    'near-duplicate detection': _E('Finding items whose embeddings are almost identical, such as reworded copies.', 'primer.ml.embeddings.clustering'),
    'anomaly detection': _E('Flagging items far from every known group.', 'primer.ml.embeddings.clustering'),
    'blue/green deployment': _E('Building the new version beside the old one, switching traffic once it proves itself, and keeping the old one for rollback.', 'primer.ml.embeddings.operations'),
    'dual write': _E('Writing every new record to both the old and the new system during a migration.', 'primer.ml.embeddings.operations'),
    'backfill': _E('Re-processing existing records into a new system, such as re-embedding a corpus with a new model.', 'primer.ml.embeddings.operations'),
    'index alias': _E('A pointer name like "live" that search uses, so switching indexes means repointing one name.', 'primer.ml.embeddings.operations'),
    'domain mismatch': _E('A general model underperforming on specialised language it never saw, such as company jargon.', 'primer.ml.embeddings.operations'),
    'retrieval failure': _E('A wrong answer caused because no relevant document was retrieved.', 'primer.ml.embeddings.operations'),
    'generation failure': _E('A wrong answer produced even though a relevant document was retrieved.', 'primer.ml.embeddings.operations'),
    'inverted file': _E('In an IVF index, the list of vectors assigned to one cluster, so a search scans only the lists it picks.', 'primer.ml.embeddings.ann'),
    'diversity heuristic': _E("HNSW's rule of keeping a neighbour only if it is closer to the new node than to any neighbour already kept, so links fan out in different directions.", 'primer.ml.embeddings.ann'),
    'sparse retrieval': _E('Keyword search such as BM25, which scores exact word matches.', 'primer.ml.embeddings.retrieval'),
    'hysteresis': _E("Making the bar for changing state higher than the bar for staying, so a decision doesn't flicker between two close options.", None),
    'debounce': _E('Waiting until a signal has settled before acting on it, so a burst of changes triggers one action.', None),
    'cooldown': _E('A quiet period after an action during which the same action is not repeated.', None),
    'latency': _E("The delay between something happening and the system's response to it.", 'primer.agents.cost'),
    'rank': _E('How many independent directions a matrix really contains; a table made by multiplying one column by one row has rank 1.', 'primer.ml.training_stages', scope=('primer/ml/training_stages',)),
    'singular value decomposition': _E('Breaking a matrix into independent directions ranked by importance, so the top few capture most of what it does.', 'primer.ml.embeddings.word2vec'),
    'frobenius norm': _E('The size of a whole matrix: square every entry, add them up, take the square root.', 'primer.notation'),
    'prefix tuning': _E('Training a few special virtual-token vectors placed before every input while the model stays frozen.', 'primer.ml.training_stages'),
    'ppo': _E('Proximal Policy Optimization: a reinforcement learning algorithm that improves a policy in small, clipped steps; widely used for RLHF.', 'primer.ml.training_stages'),
    'alignment tax': _E('Capability lost as a side effect of training a model to be helpful, honest and harmless.', 'primer.ml.training_stages'),
    'win rate': _E('The share of head-to-head comparisons one model wins; 50% means the two are indistinguishable.', 'primer.agents.evals'),
    'few-shot prompt': _E('A prompt that includes a few worked examples of the task before the real question.', 'primer.agents.context'),
    'linear probe': _E('Freezing a model and training only a simple linear classifier on its embeddings, to measure how useful they are.', 'primer.ml.embeddings.contrastive'),
    'zero-shot': _E('Doing a task with no task-specific training examples.', 'primer.ml.embeddings.contrastive'),
    'map@10': _E('Mean average precision over the top 10 results: rewards putting correct matches in the top 10, and higher up within it.', 'primer.ml.metrics'),
    'nf4': _E('4-bit NormalFloat: a 4-bit number format whose 16 levels are spaced to match the bell-curve shape of neural network weights.', 'primer.ml.inference'),
    'double quantization': _E('Compressing the per-block scale factors of a quantized model as well, saving about 0.37 bits per parameter.', 'primer.ml.inference'),
    'mips': _E('Maximum inner product search: finding the stored vectors with the largest dot product against a query vector, usually approximately with an index.', 'primer.ml.embeddings.ann'),
    'expected value': _E('The average you would get over many tries.', 'primer.notation'),
    'credit assignment': _E('Working out which of many earlier actions deserves the blame or credit for how an episode ended.', 'primer.agents.planning'),
    'pass@1': _E('The share of problems solved by the first program submitted for each, judged by hidden tests.', 'primer.agents.evals'),
    'machine translation': _E('Turning text in one language into another.', 'primer.ml.transformer'),
    'linear attention': _E('Attention variants that avoid scoring every pair of tokens, so cost grows in proportion to n instead of n².', 'primer.ml.attention'),
    'in-context learning': _E("Picking up a task from instructions or examples in the prompt, with no change to the model's weights.", 'primer.ml.big_picture'),
    'speculative execution': _E('Starting likely work early, in parallel, and throwing it away if the guess was wrong.', 'primer.ml.inference'),
    'petaflop/s-day': _E('10¹⁵ operations per second for one day, 8.64 × 10¹⁹ operations: a unit for training budgets.', 'primer.ml.training_stages'),
    'isoflop profile': _E('A curve of final loss against model size at a fixed training compute; its minimum is the best size for that budget.', 'primer.ml.training_stages'),
    'hbm': _E("The GPU's large main memory: tens of GB, about 10× slower than on-chip SRAM.", 'primer.ml.inference'),
    'sram': _E("The tiny, very fast on-chip memory next to a GPU's arithmetic units.", 'primer.ml.inference'),
    'kernel fusion': _E('Doing several steps in one GPU program, so data is read from and written to main memory only once.', 'primer.ml.inference'),
    'internal fragmentation': _E('Memory reserved for a request that it never uses.', 'primer.ml.inference'),
    'external fragmentation': _E('Free memory split into gaps too small to use.', 'primer.ml.inference'),
    'block table': _E('A per-request map from logical KV-cache blocks to physical blocks in GPU memory.', 'primer.ml.inference'),
    'copy-on-write': _E('Sharing one copy of data and making a private copy only when someone needs to change it.', 'primer.ml.inference'),
    'co-adaptation': _E("When neurons learn to depend on each other's exact behaviour, each correcting the others' quirks; it fits the training data but breaks on new data.", 'primer.ml.regularization'),
    'max-norm constraint': _E("Capping the length of each neuron's incoming weight vector at a fixed radius, shrinking it back whenever an update pushes it past.", 'primer.ml.regularization'),
    'regret': _E("In online learning, the total extra loss from choosing each step's parameters before seeing that step's data, compared with the best fixed choice in hindsight.", 'primer.ml.optimizers', scope=('primer/ml/optimizers',)),
    'convex': _E('Bowl-shaped everywhere with a single bottom, so any downhill path reaches the same minimum; neural network losses are not convex.', 'primer.notation'),
    'adagrad': _E("An optimizer that divides each weight's step by the square root of the sum of all its past squared gradients, so steps only ever shrink.", 'primer.ml.optimizers'),
    'rmsprop': _E('An optimizer that divides each step by the square root of a running average of recent squared gradients.', 'primer.ml.optimizers'),
    'sparse gradients': _E('Gradients that are zero most of the time for a given weight, such as the weight for a rare word.', 'primer.ml.optimizers'),
    'degradation problem': _E('Deeper plain networks reaching higher error even on their training data: an optimization failure, not overfitting.', 'primer.ml.deep_nets'),
    'subword unit': _E('A piece of a word, from a single character to a whole frequent word; a fixed set of them can spell any word.', 'primer.ml.tokenization'),
    'mean average precision': _E('How well a system ranks correct results above wrong ones, averaged over queries or classes.', 'primer.ml.metrics'),
    'cbow': _E("Continuous bag of words: the word2vec model that averages the surrounding words' vectors to predict the middle word.", 'primer.ml.embeddings.word2vec'),
    'hierarchical softmax': _E('Replacing one softmax over the whole vocabulary with about log₂V yes-or-no decisions down a binary tree.', 'primer.ml.embeddings.word2vec'),
    'subsampling': _E('Randomly skipping most occurrences of very frequent words during training, which is faster and improves rare-word vectors.', 'primer.ml.embeddings.word2vec'),
    'siamese network': _E('Two copies of one network with shared weights, each reading one input, so their outputs can be compared directly.', 'primer.ml.embeddings.contrastive'),
    'triplet loss': _E('A loss requiring an anchor to be closer to its positive than to its negative by at least a margin.', 'primer.ml.losses'),
    'spearman correlation': _E('How well two rankings agree, from −1 (reversed) to 1 (identical).', 'primer.ml.metrics'),
    'top-k retrieval accuracy': _E('The share of questions for which at least one of the top k retrieved passages contains the answer.', 'primer.ml.metrics'),
    'navigable small world': _E('A graph where always stepping to the neighbour closest to the target reaches it in few hops.', 'primer.ml.embeddings.ann'),
    'skip list': _E('A sorted list with random express lanes on top, searched by running along the top lane and dropping down, in about log(n) steps.', 'primer.ml.embeddings.ann'),
    'huffman tree': _E('A binary tree that gives frequent items short codes and rare items long ones.', 'primer.ml.embeddings.word2vec'),
    'logistic regression': _E('A linear model whose weighted sum passes through a sigmoid to give a probability.', 'primer.ml.neural_net'),
    'locality-sensitive hashing': _E('Hashing vectors so that similar vectors are likely to land in the same bucket, for fast approximate search.', 'primer.ml.embeddings.ann'),
    'ollama': _E('A free app that downloads open language models and runs them on your own computer, served over a small local web API.', 'primer.agents.llm'),
    'local model': _E("A language model running on your own machine rather than a provider's servers: free per call, private and offline, but smaller.", 'primer.agents.llm'),
}


@functools.lru_cache(maxsize=1)
def _usage_text() -> str:
    """Every lesson's prose, as the case of each term is judged from it."""
    import ast
    import re
    from pathlib import Path

    # Read the docstrings from the source, without importing the lessons.
    root = Path(__file__).resolve().parent
    text = "\n".join(ast.get_docstring(ast.parse(f.read_text(encoding="utf-8"))) or "" for f in sorted(root.rglob("*.py")))
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # fenced code and diagram labels
    # Inline code, italic titles (which may wrap onto a second line) and link text; then citation lines.
    text = re.sub(r"`[^`\n]*`|(?<!\*)\*(?!\s)[^*]+?(?<!\s)\*(?!\*)|\[[^\]\n]*\]\([^)]*\)", " ", text)
    return "\n".join(line for line in text.splitlines() if "http" not in line and "arxiv" not in line.lower())


@functools.lru_cache(maxsize=None)
def _usage(terms: tuple[str, ...]) -> dict[str, Counter]:
    """How often each casing of each term appears mid-sentence, found in one pass over the text."""
    import re

    # Only mid-sentence uses count: after a lower-case word and a space, or an opening bracket.
    # Headings, table cells, list items and sentence starts capitalise words for reasons that aren't the term's.
    alternation = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    pattern = re.compile(rf"(?:(?<=[a-z0-9,;)] )|(?<=\())(?:{alternation})(?![\w-])", re.I)
    counts: dict[str, Counter] = {}
    for m in pattern.finditer(_usage_text()):
        counts.setdefault(m.group(0).lower(), Counter())[m.group(0)] += 1
    return counts


def display_term(term: str) -> str:
    """How the lessons usually write a term: "acl" -> "ACL", "adamw" -> "AdamW", "attention" -> "attention".

    Counted over every lesson's text, mid-sentence only, where capitals mean something
    about the word. A term the lessons never use mid-sentence keeps its key.
    """
    terms = tuple(sorted(GLOSSARY)) if term in GLOSSARY else (term,)
    forms = _usage(terms).get(term.lower())
    return forms.most_common(1)[0][0] if forms else term


def lesson_path(dotted: str) -> str:
    """"primer.ml.attention" -> "primer/ml/attention.html", the pdoc page for that module."""
    return dotted.replace(".", "/") + ".html"


def glossary_js() -> str:
    """The glossary as a script that sets `window.PRIMER_GLOSSARY` (works from file:// too)."""
    data = {
        term: {"def": e.definition, "lesson": lesson_path(e.lesson) if e.lesson else None, "term": display_term(term)}
        for term, e in sorted(GLOSSARY.items())
    }
    return "window.PRIMER_GLOSSARY = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n"


def _render_doc() -> str:
    rows = "\n".join(
        f"| **{display_term(term)}** | {e.definition} | {f'`{e.lesson}`' if e.lesson else ''} |" for term, e in sorted(GLOSSARY.items())
    )
    return __doc__ + "\n| Term | Meaning | Taught in |\n|---|---|---|\n" + rows + "\n"


__doc__ = _render_doc()

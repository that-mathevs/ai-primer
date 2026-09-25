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
 "activation function": {
  "def": "The nonlinear function inside a neuron, such as ReLU or GELU. Without it, stacked layers collapse into one.",
  "lesson": "primer/ml/neural_net.html",
  "term": "activation function"
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
 "alignment tax": {
  "def": "Capability lost as a side effect of training a model to be helpful, honest and harmless.",
  "lesson": "primer/ml/training_stages.html",
  "term": "alignment tax"
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
 "attention": {
  "def": "The mechanism that lets each token look at every other token, score how relevant each one is, and blend in information from the relevant ones.",
  "lesson": "primer/ml/attention.html",
  "term": "attention"
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
 "base model": {
  "def": "A model after pretraining only: knowledgeable, but it continues text rather than following instructions.",
  "lesson": "primer/ml/training_stages.html",
  "term": "base model"
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
 "bpe": {
  "def": "Byte pair encoding: build a vocabulary by repeatedly merging the most frequent adjacent pair of symbols.",
  "lesson": "primer/ml/tokenization.html",
  "term": "BPE"
 },
 "bradley-terry model": {
  "def": "The rule that the probability A beats B is the sigmoid of their score difference.",
  "lesson": "primer/ml/training_stages.html",
  "term": "bradley-terry model"
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
 "capability negotiation": {
  "def": "The start-of-session exchange where client and server announce which optional features they support.",
  "lesson": "primer/agents/mcp.html",
  "term": "capability negotiation"
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
 "chain rule": {
  "def": "To get the slope through a chain of functions, multiply the slopes of each link.",
  "lesson": "primer/notation.html",
  "term": "chain rule"
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
 "chunking": {
  "def": "Splitting documents into passages before embedding them, so retrieval can return just the relevant part.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "chunking"
 },
 "circuit breaker": {
  "def": "A wrapper that stops calling a failing dependency for a cool-down period, failing fast instead.",
  "lesson": "primer/agents/failures.html",
  "term": "circuit breaker"
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
  "def": "Product quantization's small catalogue of representative sub-vectors; each piece of a vector is stored as the number of its nearest entry.",
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
 "dbscan": {
  "def": "Density-based clustering that grows clusters from points with enough close neighbours and labels the rest as noise.",
  "lesson": "primer/ml/embeddings/clustering.html",
  "term": "DBSCAN"
 },
 "debounce": {
  "def": "Waiting until a signal has settled before acting on it, so a burst of changes triggers one action.",
  "lesson": null,
  "term": "debounce"
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
 "degradation problem": {
  "def": "Deeper plain networks reaching higher error even on their training data: an optimization failure, not overfitting.",
  "lesson": "primer/ml/deep_nets.html",
  "term": "degradation problem"
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
 "dimension": {
  "def": "One of the numbers in a vector: a 768-dimension embedding is a list of 768 numbers.",
  "lesson": "primer/ml/embeddings/similarity.html",
  "term": "dimension"
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
 "entailment": {
  "def": "Whether one text logically follows from another, checked by a model trained for it.",
  "lesson": "primer/agents/guardrails.html",
  "term": "entailment"
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
 "feature flag": {
  "def": "A configuration switch that turns a capability on or off without redeploying.",
  "lesson": "primer/agents/deployment.html",
  "term": "feature flag"
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
 "flat index": {
  "def": "Exact search that compares the query with every stored vector; the ground truth other indexes are measured against.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "flat index"
 },
 "flops": {
  "def": "Floating-point operations: individual multiplies or adds, used to measure compute (about 2 per parameter per token to generate, 6 to train).",
  "lesson": "primer/ml/transformer.html",
  "term": "FLOPs"
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
 "frobenius norm": {
  "def": "The size of a whole matrix: square every entry, add them up, take the square root.",
  "lesson": "primer/notation.html",
  "term": "frobenius norm"
 },
 "full fine-tune": {
  "def": "Updating every weight of a model. Rarely worth the cost.",
  "lesson": "primer/ml/training_stages.html",
  "term": "full fine-tune"
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
 "generation failure": {
  "def": "A wrong answer produced even though a relevant document was retrieved.",
  "lesson": "primer/ml/embeddings/operations.html",
  "term": "generation failure"
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
 "graduated autonomy": {
  "def": "Granting an agent more independence in steps (shadow, then approval, then autonomy), each earned with evidence.",
  "lesson": "primer/agents/deployment.html",
  "term": "graduated autonomy"
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
 "hard negative": {
  "def": "A training example that looks relevant but isn't, such as the right topic with the wrong answer. Training on them teaches fine distinctions.",
  "lesson": "primer/ml/embeddings/contrastive.html",
  "term": "hard negative"
 },
 "hash chain": {
  "def": "Records that each store the previous record's hash, so any edit or deletion is detectable.",
  "lesson": "primer/agents/deployment.html",
  "term": "hash chain"
 },
 "hbm": {
  "def": "The GPU's large main memory: tens of GB, about 10× slower than on-chip SRAM.",
  "lesson": "primer/ml/inference.html",
  "term": "hbm"
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
 "internal fragmentation": {
  "def": "Memory reserved for a request that it never uses.",
  "lesson": "primer/ml/inference.html",
  "term": "internal fragmentation"
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
 "jitter": {
  "def": "Random variation added to retry delays so clients don't all retry at the same moment.",
  "lesson": "primer/agents/failures.html",
  "term": "jitter"
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
  "term": "kl divergence"
 },
 "kv cache": {
  "def": "Stored keys and values from earlier tokens, so each new token is computed without redoing work. It trades GPU memory for speed.",
  "lesson": "primer/ml/inference.html",
  "term": "KV cache"
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
 "label smoothing": {
  "def": "Training against a softened target, such as 0.9 on the right class and the rest spread evenly, to discourage over-confidence.",
  "lesson": "primer/ml/losses.html",
  "term": "label smoothing"
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
 "layer normalization": {
  "def": "Rescaling each token's numbers to a steady average and spread, which keeps training stable.",
  "lesson": "primer/ml/deep_nets.html",
  "term": "layer normalization"
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
  "def": "Freezing a model and training only a simple linear classifier on its embeddings, to measure how useful they are.",
  "lesson": "primer/ml/embeddings/contrastive.html",
  "term": "linear probe"
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
 "mamba": {
  "def": "A state-space model that trains in parallel and runs in time linear in sequence length.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "Mamba"
 },
 "map@10": {
  "def": "Mean average precision over the top 10 results: rewards putting correct matches in the top 10, and higher up within it.",
  "lesson": "primer/ml/metrics.html",
  "term": "map@10"
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
 "memory-bound": {
  "def": "Limited by how fast data arrives from memory rather than by arithmetic, as in decoding.",
  "lesson": "primer/ml/inference.html",
  "term": "memory-bound"
 },
 "mips": {
  "def": "Maximum inner product search: finding the stored vectors with the largest dot product against a query vector, usually approximately with an index.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "mips"
 },
 "mixture of experts": {
  "def": "Replacing one feed-forward network with many expert networks and a router that sends each token to a few of them.",
  "lesson": "primer/ml/transformer.html",
  "term": "Mixture of Experts"
 },
 "model context protocol": {
  "def": "An open standard that lets any AI application connect to any tool server the same way.",
  "lesson": "primer/agents/mcp.html",
  "term": "Model Context Protocol"
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
 "multi-head attention": {
  "def": "Several attention computations run in parallel on slices of the vectors, so each head can track a different kind of relationship.",
  "lesson": "primer/ml/attention.html",
  "term": "multi-head attention"
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
 "opentelemetry": {
  "def": "The open standard for traces, metrics and logs.",
  "lesson": "primer/agents/observability.html",
  "term": "OpenTelemetry"
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
 "output gate": {
  "def": "The LSTM dial that decides how much of the cell state to show as output.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "output gate"
 },
 "overfitting": {
  "def": "When a model memorizes its training data, noise included, and does worse on new data.",
  "lesson": "primer/ml/regularization.html",
  "term": "overfitting"
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
 "pass@1": {
  "def": "The share of problems solved by the first program submitted for each, judged by hidden tests.",
  "lesson": "primer/agents/evals.html",
  "term": "pass@1"
 },
 "patch": {
  "def": "A small square cut from an image and flattened into one token.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "patch"
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
  "def": "In preference tuning, the model being trained, viewed as a probability distribution over responses.",
  "lesson": "primer/ml/training_stages.html",
  "term": "policy"
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
 "procedural memory": {
  "def": "Memory of how to do things, such as a learned workflow or saved skill.",
  "lesson": "primer/agents/memory.html",
  "term": "procedural memory"
 },
 "product quantization": {
  "def": "Compressing a vector by splitting it into chunks and replacing each chunk with the ID of its nearest entry in a small codebook.",
  "lesson": "primer/ml/embeddings/ann.html",
  "term": "product quantization"
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
 "qlora": {
  "def": "LoRA adapters trained on top of base weights stored in 4 bits.",
  "lesson": "primer/ml/training_stages.html",
  "term": "QLoRA"
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
 "regularization": {
  "def": "Any technique that discourages memorizing, such as dropout, weight decay or early stopping.",
  "lesson": "primer/ml/regularization.html",
  "term": "regularization"
 },
 "relu": {
  "def": "An activation function that keeps positive numbers and turns negatives into zero.",
  "lesson": "primer/ml/neural_net.html",
  "term": "ReLU"
 },
 "reranker": {
  "def": "A second, more accurate model that reorders the top results of a fast first-stage search.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "reranker"
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
 "retrieval failure": {
  "def": "A wrong answer caused because no relevant document was retrieved.",
  "lesson": "primer/ml/embeddings/operations.html",
  "term": "retrieval failure"
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
 "self-attention": {
  "def": "Attention where a sequence attends to itself: every token scores every other token in the same text.",
  "lesson": "primer/ml/attention.html",
  "term": "self-attention"
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
 "short-term memory": {
  "def": "The current conversation, kept within a token budget: recent turns verbatim, older ones summarized.",
  "lesson": "primer/agents/memory.html",
  "term": "short-term memory"
 },
 "shortlist": {
  "def": "The small set of top candidates from a cheap first-stage search that a slower, more precise stage reorders.",
  "lesson": "primer/ml/embeddings/retrieval.html",
  "term": "shortlist"
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
 "span": {
  "def": "One step inside a trace, with its start time, duration and details.",
  "lesson": "primer/agents/observability.html",
  "term": "span"
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
  "term": "sram"
 },
 "standard deviation": {
  "def": "The square root of the variance: the typical distance of a value from the average.",
  "lesson": "primer/notation.html",
  "term": "standard deviation"
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
 "tanh": {
  "def": "Squashes any number into the range −1 to 1.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "tanh"
 },
 "task budget": {
  "def": "Hard limits on steps and tokens for one agent task.",
  "lesson": "primer/agents/cost.html",
  "term": "task budget"
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
 "threshold calibration": {
  "def": "Choosing a similarity cut-off by measuring precision and recall on labeled pairs for one specific model.",
  "lesson": "primer/ml/embeddings/similarity.html",
  "term": "threshold calibration"
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
 "vector": {
  "def": "A list of numbers, like (3, 1, 2). In AI, a word, sentence or image is represented as a vector.",
  "lesson": "primer/notation.html",
  "term": "vector"
 },
 "vision transformer": {
  "def": "A transformer that treats small image patches as tokens.",
  "lesson": "primer/ml/cnn_rnn.html",
  "term": "Vision Transformer"
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
 "warmup": {
  "def": "Starting training with a tiny learning rate and ramping it up, which keeps the first updates from destabilizing the model.",
  "lesson": "primer/ml/optimizers.html",
  "term": "warmup"
 },
 "weight": {
  "def": "A learned number that says how strongly one input influences an output. Training adjusts the weights.",
  "lesson": "primer/ml/neural_net.html",
  "term": "weight"
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
 "zero-shot": {
  "def": "Doing a task with no task-specific training examples.",
  "lesson": "primer/ml/embeddings/contrastive.html",
  "term": "zero-shot"
 },
 "zero-shot classification": {
  "def": "Labeling items by comparing them to a text description of each label, with no training on those labels.",
  "lesson": "primer/ml/embeddings/contrastive.html",
  "term": "zero-shot classification"
 }
};

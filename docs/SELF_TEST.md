# Self-test: every question in the primer

Generated from each lesson's `## Self-test questions` section by `make readme`; edit the lesson, not this file. Answer each question out loud before reading the answer.


## Before you begin


### 0. Math notation, from zero

From [`primer.notation`](../primer/notation.py).

**Read $\sum_{i=1}^{3} i^2$ aloud and evaluate it.**
"The sum, for i from 1 to 3, of i squared": 1 + 4 + 9 = 14.

**What is (2, −1, 3) · (1, 4, 0), and what does its sign tell you?**
2 − 4 + 0 = −2. It's negative, so the vectors point somewhat apart.

**A is 4 × 3 and B is 3 × 5. What shape is AB? What about BA?**
AB is 4 × 5. BA is not defined: B's 5 columns don't match A's 4 rows.

**Why do language models add log-probabilities instead of multiplying
probabilities?**
A product of thousands of numbers below 1 underflows to 0 in floating point.
log(a·b) = log a + log b, so the product becomes a sum of moderate
negative numbers.

**What does ∇L tell you, and which way does training move?**
For every weight, how fast the loss rises as that weight increases. Training
moves each weight the opposite way, scaled by the learning rate η.


## Part 1: how the model works inside


### 1. The big picture

From [`primer.ml.big_picture`](../primer/ml/big_picture.py).

**Walk through what happens when you send a prompt.**
Tokenizer turns text into ids; each id looks up an embedding; position
information is added; the vectors pass through N transformer blocks; the last
position's vector is scored against the whole vocabulary; softmax and sampling
pick a token; it's appended and the loop repeats until a stop token.

**Why does only the last position matter when generating?**
Its vector has attended to every earlier token and is the one trained to
predict what comes next; earlier rows predict tokens we already have.

**What does temperature do to the scores (2, 1, 0.5) at T = 0.5?**
Doubles them to (4, 2, 1) before softmax, sharpening the distribution: the
top token rises from 0.63 to 0.84.

**How many token positions does the naive loop process for a 2-token prompt and 4 new tokens?**
2 + 3 + 4 + 5 = 14. A KV cache avoids re-reading the unchanged prefix.

**What loss does an untrained model get over a 400-token vocabulary, and why?**
About ln 400 ≈ 5.99 (perplexity about 400), because with random, tiny weights
its predictions are close to uniform.

**How is training different from inference?**
Same forward pass, plus a loss comparing predictions to the real next tokens,
then backpropagation and a weight update. Inference only runs the forward pass.


### 2. Neural networks

From [`primer.ml.neural_net`](../primer/ml/neural_net.py).

**Why can't you just stack linear layers?**
Their composition is a single linear map (W1·W2 is just another matrix), so
extra layers add parameters but no expressive power.

**What does backprop actually compute?**
The gradient: for every weight, how much a tiny change would change the loss.
It applies the chain rule from the output backward, reusing values cached in
the forward pass.

**Why is ReLU preferred over sigmoid in hidden layers?**
Sigmoid's derivative is at most 0.25 and near 0 when saturated, so gradients
shrink multiplicatively with depth. ReLU's derivative is exactly 1 for
positive inputs. The cost: neurons whose input stays negative get zero
gradient and "die".

**Why does training need more memory than inference?**
The backward pass needs every layer's activations from the forward pass, so
they must be kept until the gradients are computed. Inference can discard
each activation as soon as the next layer has used it.

**What's the gradient of sigmoid + binary cross-entropy with respect to the logit?**
ŷ − y: prediction minus target. Clean, bounded and cheap, which is why the
two are always fused.

**Batch vs. step vs. epoch?**
A batch is the examples per update, a step is one update, an epoch is one
full pass over the data.


### 3. Optimizers

From [`primer.ml.optimizers`](../primer/ml/optimizers.py).

**What's the single most important hyperparameter, and what happens at each extreme?**
The learning rate. Too high: steps overshoot and training diverges or
oscillates (loss spikes, NaNs). Too low: progress is so slow that training
stalls or settles in a poor spot.

**Why does momentum help in a narrow valley?**
Gradients across the valley flip sign every step and cancel in the velocity,
while the small, consistent gradient along the valley accumulates, so the
optimizer speeds up in the useful direction.

**What does Adam's division by √v̂ achieve?**
It normalizes each weight's step by the typical size of its gradient, so
every weight moves roughly η per step regardless of gradient scale. One
learning rate then works for all parameters.

**What's the difference between Adam with L2 and AdamW?**
With L2, the decay term is added to the gradient and then rescaled by Adam,
so its effective strength varies per weight and λ loses its meaning. AdamW
applies decay directly to the weights, outside the rescaling.

**Why warm up the learning rate?**
At the start, weights are random and Adam's moment estimates are based on a
handful of steps, so full-size updates can be wildly wrong and destabilize
training. Ramping up gives the statistics time to settle.

**What does gradient clipping protect against, and why clip the global norm?**
Rare huge gradients (exploding gradients, bad batches) that would throw the
weights far off. Clipping the combined norm scales every tensor by the same
factor, which preserves the update's direction.


### 4. Training deep networks

From [`primer.ml.deep_nets`](../primer/ml/deep_nets.py).

**Why do gradients vanish in deep sigmoid networks?**
Backprop multiplies the gradient by each layer's slope, and sigmoid's slope
is at most 0.25. Thirty layers can shrink it by 0.25³⁰, so early layers
stop learning.

**What's the difference between Xavier and He initialization?**
Both choose the weight variance so signal size is preserved. Xavier uses
2 / (fan-in + fan-out), suited to symmetric activations like tanh. He uses
2 / fan-in, doubling the variance to compensate for ReLU zeroing half its
inputs.

**How do residual connections fix vanishing gradients?**
The block's output is input + F(input), so its derivative is 1 + F′. The
gradient always has an identity path back to early layers that isn't
multiplied by small slopes.

**Why do transformers use LayerNorm instead of BatchNorm?**
BatchNorm's statistics come from the batch, which breaks for variable-length
sequences, tiny or single-example batches at inference, and makes an
example's output depend on its batch-mates. LayerNorm normalizes each token
across its own features, independent of the batch.

**What is RMSNorm and why do modern LLMs use it?**
LayerNorm without the mean subtraction and shift: divide by the
root-mean-square and apply a learned scale. It's cheaper and trains as well.

**Gradient clipping or better initialization: which fixes exploding gradients?**
Initialization (and normalization) fix the cause, keeping per-layer gain
near 1. Clipping is a safety net for occasional spikes.


### 5. Attention

From [`primer.ml.attention`](../primer/ml/attention.py).

**Explain attention to a non-engineer in 30 seconds.**
When the model reads a word, it asks which other words here help it
understand this one. It scores every other word for relevance, then builds
its understanding of the word as a mix of the relevant ones. In "the animal
didn't cross the street because it was tired", "it" draws mostly from
"animal". It does this for every word, in parallel, dozens of times over.

**Now explain it to an ML engineer in two minutes.**
Project X into Q, K and V with learned matrices. Compute QKᵀ/√d_k, an n×n
matrix of scaled similarities; add a causal mask of −∞ above the diagonal
for a decoder; softmax each row; multiply by V. Run h heads in parallel on
d_model/h slices, concatenate them, and project with W_o. Wrap the whole
thing in a residual connection with pre-layer-norm and follow it with an
FFN. Cost is O(n²·d) in compute and O(n²) in memory for the scores, which is
why FlashAttention tiles it and GQA shrinks the KV cache.

**Why divide by √d_k? What breaks without it?**
The variance of q·k grows linearly with d_k. Without scaling, scores spread
out, softmax saturates towards one-hot, its Jacobian goes to about zero, and
the query/key projections stop receiving gradient. Training stalls or turns
unstable.

**Why is long context expensive? Name two techniques that reduce the cost.**
Scores are n×n per head per layer, so cost is quadratic, and the KV cache
grows linearly with every token held in memory. Two fixes: FlashAttention
(exact, memory-efficient tiling) and GQA/MQA (fewer KV heads). Others:
sliding-window or sparse attention, and prompt caching of stable prefixes.

**What does the causal mask buy you besides honest training?**
Earlier outputs never depend on later tokens, so during generation the
keys and values of past tokens can be computed once and cached. Each new
token then costs one row of attention instead of a full recomputation.

**What does GQA trade away, and for what?**
A little modelling capacity (query heads share keys and values) for a KV
cache that is several times smaller. That means more concurrent requests
and longer contexts per GPU.


### 6. Positional information

From [`primer.ml.positional`](../primer/ml/positional.py).

**Why does a transformer need positional information at all?**
Attention compares tokens by content only, so it's permutation-equivariant:
shuffling the input just shuffles the output. Without positions, word order
is invisible.

**How does RoPE encode position, and what property makes it attractive?**
It rotates each pair of dimensions in q and k by an angle proportional to
position. The q·k score then depends only on the relative distance, and
rotation preserves vector length.

**Compute a RoPE score by hand: 2 dimensions, θ = 1, q = k = (1, 0), query at 3, key at 7.**
cos(7 − 3) = cos 4 ≈ −0.654, and the same at positions 103 and 107.

**Why do sinusoidal codes use many frequencies?**
Fast frequencies distinguish neighbours; slow ones distinguish distant
positions. Together every position gets a unique, smoothly varying code, and
a shift by k is the same rotation everywhere.

**What goes wrong beyond the trained context length, and how do you fix it?**
The model meets position angles or indices it never saw, and attention
degrades. Position interpolation, NTK-aware scaling or YaRN rescale positions
or frequencies into the trained range, usually followed by a short fine-tune.

**Learned absolute position embeddings: pro and con?**
They are simple and effective within the trained length, but they cannot
represent any position beyond the table size.


### 7. The transformer

From [`primer.ml.transformer`](../primer/ml/transformer.py).

**What do attention and the feed-forward network each contribute?**
Attention mixes information across tokens; the feed-forward network
transforms each token independently and holds most of the parameters (and,
it's thought, much of the factual knowledge).

**Why residual connections?**
Each sub-layer adds a correction instead of replacing its input, so signal
and gradients have a direct path through very deep stacks.

**Pre-norm vs. post-norm?**
Pre-norm normalizes before each sub-layer and leaves the residual path
untouched; it trains more stably, so modern models use it.

**Why LayerNorm rather than BatchNorm in transformers?**
It normalizes within each token, so it doesn't depend on batch size or
sequence length.

**Roughly how many parameters does a model with 32 layers of width 4096 have, not counting embeddings?**
12 · 32 · 4096² ≈ 6.4 billion.

**Encoder-only vs. decoder-only?**
Encoders attend bidirectionally and suit classification and embeddings;
decoders attend causally and generate text.

**What does Mixture of Experts buy you, and what does it cost?**
More total parameters at the same compute per token, since each token runs
only k experts. It costs memory (all experts loaded), routing complexity and
load-balancing.

**Estimate the compute to train a 7B model on 2 trillion tokens.**
6 × 7e9 × 2e12 = 8.4e22 FLOPs.


### 8. Tokenization

From [`primer.ml.tokenization`](../primer/ml/tokenization.py).

**Why subwords instead of whole words or characters?**
Words give a huge vocabulary and unknown words; characters make sequences
several times longer, and attention cost grows with the square of length.
Subwords keep common strings short and still encode anything.

**Walk through BPE training on "low lower lowest".**
Split into letters, count neighbouring pairs, and merge the most frequent:
l+o (3), then lo+w (3), then low+e (2). Record each merge; to encode, replay
the merges in the order learned.

**Why can a byte-level BPE tokenizer never produce an unknown token?**
Its base vocabulary is all 256 byte values, and every string is a sequence of
UTF-8 bytes, so in the worst case text falls back to single bytes.

**What does WordPiece do differently from BPE?**
It merges the pair with the highest count(ab) / (count(a)·count(b)), which
favours pairs whose parts rarely appear apart, instead of the most frequent
pair.

**Your users write in Hindi. What changes in your estimates?**
Expect noticeably more tokens for the same content: higher cost and latency,
and less text per context window. Measure with the actual tokenizer.

**Why are LLMs bad at counting letters and at long arithmetic?**
They see token ids, not characters, and numbers split into chunks that don't
line up with place value.

**What does a call with 2,000 input and 500 output tokens cost at $3/$15 per million?**
0.006 + 0.0075 = $0.0135. And 1,000 tokens is about 750 English words.


### 9. Training stages

From [`primer.ml.training_stages`](../primer/ml/training_stages.py).

**Q: Why does a base model ramble instead of answering?**
A: Pretraining only teaches it to continue text. Answering questions in a
helpful format is learned later, in SFT and preference tuning.

**Q: What is the one code difference between pretraining loss and SFT loss?**
A: A mask. SFT computes the same next-token cross-entropy but averages it
only over the reply tokens, so the prompt is read but not trained on.

**Q: In RLHF, what does the reward model learn, and from what?**
A: A score for responses such that sigmoid(score gap) predicts which of two
responses a labeler preferred. It learns from pairwise comparisons, which
are far easier for people to give than perfect answers.

**Q: How does DPO avoid a reward model?**
A: It treats the log-probability ratio between the policy and a frozen
reference as an implicit reward, and applies the same pairwise loss
directly to the policy. One model, one supervised-style training loop.

**Q: What does β control in DPO?**
A: How tightly the policy is held to the reference model. It is the weight
on the drift penalty in the objective DPO optimises, reward − β × KL(policy
‖ reference), so larger β keeps the policy closer to the reference and
smaller β lets the preferences pull it further away. In the loss, a larger β
makes each unit of log-ratio count for more, so pairs are satisfied with a
smaller departure.

**Q: Why is B initialised to zero in LoRA?**
A: So B·A = 0 and the adapted model starts exactly equal to the pretrained
model. Training then learns only the change.

**Q: How many parameters does LoRA rank 8 train on a 4096 × 4096 layer?**
A: 2 × 4096 × 8 = 65,536, about 0.39% of the 16.8 million in the full
matrix.

**Q: Does LoRA slow down inference?**
A: Not if you merge: add B·A into W once and serve the result. Unmerged
adapters cost two thin extra matmuls, which is what lets you hot-swap
adapters on one base model.

**Q: A client wants the model to know their product catalogue, which
changes weekly. Fine-tune or RAG?**
A: RAG. The knowledge changes and answers should cite sources; fine-tuning
would be stale within a week and can't cite. Fine-tune only for behaviour
the prompt can't make consistent.

**Q: What does temperature do in distillation?**
A: It softens both distributions so the student learns the teacher's
relative preferences among wrong answers, not just its top pick; the T²
factor keeps gradient sizes comparable.


### 10. Inference

From [`primer.ml.inference`](../primer/ml/inference.py).

**Q: Why is decoding slow even on a huge GPU?**
A: Each token needs every weight read from memory but does only about one
operation per byte read, far below the GPU's break-even (~300 FLOPs/byte).
The arithmetic units mostly wait on memory.

**Q: What is the KV cache, and why does it matter for serving cost?**
A: Stored keys and values of all earlier tokens, so each new token is
computed once instead of recomputing the whole sequence. It grows with
context length and batch size, and that memory caps how many requests a GPU
serves at once.

**Q: How much memory does a 70B model need at 16-bit and at 4-bit?**
A: 140 GB and 35 GB for weights alone, plus KV cache and activations.

**Q: Estimate the KV cache for one 32k-token request on a Llama-3-8B-shaped
model.**
A: 2 × 32 × 8 × 128 × 2 = 131,072 bytes per token; × 32,000 ≈ 4.2 GB.

**Q: Why does grouped-query attention make serving cheaper?**
A: It shares each key/value head among several query heads, shrinking the
KV cache (4× for 32 → 8 KV heads), so more requests fit per GPU.

**Q: Why isn't temperature 0 perfectly deterministic?**
A: Floating-point addition isn't associative, and GPU reduction order can
change with kernels and batch composition, so nearly tied logits can flip.

**Q: How can speculative decoding be faster yet produce the same
distribution?**
A: Verifying several draft tokens costs one memory-bound pass of the big
model, about the same as generating one. The min(1, p/q) accept rule plus
resampling from max(0, p − q) on rejection makes the output exactly the big
model's distribution.

**Q: What does continuous batching fix?**
A: Idle slots: static batches wait for their longest request, while
continuous batching refills any freed slot immediately, raising utilisation.

**Q: How do you structure a prompt to benefit from prompt caching?**
A: Put stable content (system prompt, tool definitions, reference documents)
first and anything that varies (the question, timestamps, IDs) last, because
a cache is valid only up to the first differing token.


### 11. Loss functions

From [`primer.ml.losses`](../primer/ml/losses.py).

**The model puts 1% on the right token. What's the loss, and why so large?**
−ln(0.01) = 4.61, about 44× the loss at 90% (0.105). The log punishes confident
mistakes steeply, which pushes the model toward calibrated probabilities.

**Loss is 0.69 per token. What's the perplexity?**
e^0.69 = 2: as uncertain as a coin flip between two options.

**Why compute cross-entropy from logits instead of from softmax outputs?**
Softmax of large logits overflows, and the log of a tiny probability
underflows to −∞. Log-sum-exp subtracts the max first and never
exponentiates a large number. The fused gradient is simply softmax − onehot.

**When would you pick MAE over MSE?**
When outliers are noise you don't want to chase. MSE squares errors, so one
huge miss dominates; MAE weighs errors linearly.

**How does InfoNCE get negatives without labelling them?**
It uses the other items in the batch: for each query, every other query's
positive passage is a negative. Bigger batches mean more (and harder)
negatives for free.

**Why do hard negatives matter?**
Easy negatives are already far away and contribute almost no gradient. Hard
negatives score high and produce most of the learning signal, teaching the
model to separate "on topic" from "actually answers the question".


### 12. Metrics

From [`primer.ml.metrics`](../primer/ml/metrics.py).

**Q: A fraud model has 94% accuracy. Is it good?**
A: You can't tell from accuracy. If fraud is 10% of traffic, check recall
and precision. In the worked example recall is only 60%, so 4 in 10 frauds
slip through.

**Q: When do you optimize for precision vs. recall?**
A: By the cost of each error. If a miss is expensive (fraud, cancer
screening, a relevant legal document), favor recall. If a false alarm is
expensive (blocking a good customer, paging an engineer at 3 a.m.), favor
precision. Set the threshold to minimize expected cost.

**Q: What does an AUC of 0.8 mean?**
A: A randomly chosen positive gets a higher score than a randomly chosen
negative 80% of the time. It measures ranking quality across all
thresholds; 0.5 is random.

**Q: Which retrieval metric matters most for RAG, and why?**
A: Recall@k, where k is the number of chunks you pass to the model. If the
right passage isn't in the top k, no prompt engineering can recover the
answer. Add MRR or nDCG when position within the context matters.

**Q: Why not grade LLM answers with BLEU or ROUGE?**
A: They measure surface overlap with one reference. A correct paraphrase
scores low and a wrong answer that reuses the reference's words scores
high. Use code-based checks where possible and a rubric-driven LLM judge
otherwise, validated against human labels.

**Q: Your LLM judge agrees with humans 90% of the time. Good enough?**
A: Not necessarily. If 90% of answers are "pass", a judge that always says
"pass" also agrees 90%. Compute Cohen's kappa to correct for chance
agreement.


### 13. Overfitting and regularization

From [`primer.ml.regularization`](../primer/ml/regularization.py).

**Training loss drops, validation loss rises. What's happening and what do you do?**
Overfitting: the model is memorising training-set noise. Stop early (keep
the best-validation weights), add regularization (dropout, weight decay),
get more or more varied data, or reduce capacity.

**How do you tell underfitting from overfitting?**
Underfitting: training and validation errors are both high. Overfitting:
training error is low and validation error is much higher. They need
opposite fixes.

**Why does dropout scale the surviving activations by 1/(1 − p)?**
So the expected value of each activation is the same during training as at
evaluation, where nothing is dropped. The evaluation-time network can then
be used as is.

**Why does L1 produce exact zeros but L2 doesn't?**
L1's penalty has a constant slope, so small weights feel a fixed pull toward
zero that the data can't outweigh; they're thresholded to zero. L2's pull is
proportional to the weight, so it fades as a weight shrinks and never quite
reaches zero.

**Why must the test set be touched only once?**
Every decision made by looking at test scores fits the model to the test
set a little. Do it repeatedly and the test score stops being an unbiased
estimate of performance on new data.

**Give two examples of data leakage.**
The same (or near-duplicate) documents in both the training and test
splits; and a feature that's only known after the outcome (a chargeback flag
in a fraud model). For LLM evaluation: benchmark questions present in the
pretraining data.


### 14. CNNs and RNNs

From [`primer.ml.cnn_rnn`](../primer/ml/cnn_rnn.py).

**Q: What does one number in a feature map mean?**
A: How strongly the filter's pattern matches the image patch at that
position: the sum of filter weights times the pixels under them.

**Q: Why does a CNN need far fewer parameters than a dense layer on images?**
A: Weight sharing: one small filter is reused at every position, so the
parameter count depends on filter size and number of filters, not on image
size.

**Q: What does pooling buy you?**
A: Smaller maps (less computation), a faster-growing receptive field, and
tolerance to small shifts, since the strongest response in a neighbourhood
survives wherever exactly it was.

**Q: Why did VGG use stacks of 3×3 filters instead of larger ones?**
A: Two 3×3 layers see a 5×5 region with 18 weights instead of 25, and add an
extra nonlinearity between them.

**Q: How does a Vision Transformer turn an image into tokens?**
A: It cuts the image into fixed-size patches (e.g. 16×16), flattens each
into a vector, projects it to the model width and adds a position
embedding; the patches are then processed like words.

**Q: Why do plain RNNs struggle with long-range dependencies?**
A: Backpropagation through time multiplies the gradient by the recurrent
weights and tanh slopes at every step, so it shrinks geometrically (or
explodes) and early inputs stop influencing learning.

**Q: How do LSTM gates fix that?**
A: The cell state is updated additively, c = f ⊙ c_prev + i ⊙ g, so with the
forget gate near 1 information and gradients flow through many steps almost
unchanged.

**Q: Why did transformers replace RNNs?**
A: RNNs process tokens sequentially, which prevents parallel training, and
force all history through one fixed-size state. Attention connects any two
tokens in one step and trains all positions in parallel.

**Q: What did CNNs contribute that still matters?**
A: Residual connections (from ResNet), which make very deep networks
trainable and are in every transformer block, plus the general lesson that
built-in assumptions help when data is limited.


## Embeddings, the centerpiece


### 15. Word embeddings

From [`primer.ml.embeddings.word2vec`](../primer/ml/embeddings/word2vec.py).

**Q: What does skip-gram predict, and where do its training labels come from?**
Each center word predicts the words within a window around it. The labels
come from the text itself (which words actually appeared nearby), so no
human labeling is needed.

**Q: Why negative sampling instead of a softmax over the vocabulary?**
A full softmax needs a score for every vocabulary word for every training
pair. Negative sampling turns it into k + 1 yes/no decisions (true context
vs. a few noise words), which is orders of magnitude cheaper and works as
well for learning vectors.

**Q: Why does king − man + woman land near queen?**
The contexts that separate king from queen (he/his vs. she/her) are the same
ones that separate man from woman, so training makes those differences the
same direction. Adding that direction to "king" moves it toward "queen".

**Q: What's the relationship between word2vec and GloVe?**
Both encode co-occurrence statistics. GloVe explicitly fits log
co-occurrence counts; skip-gram with negative sampling implicitly factorizes
a shifted PMI matrix. PPMI + SVD gives similar vectors.

**Q: What's the main limitation of static word embeddings and what fixed it?**
One vector per word regardless of meaning, so "bank" blends its senses.
Contextual embeddings from transformers compute a vector per word *in
context*, so each sense gets its own representation.


### 16. Similarity

From [`primer.ml.embeddings.similarity`](../primer/ml/embeddings/similarity.py).

**Q: When do cosine, dot product and Euclidean distance give the same ranking?**
When all vectors are L2-normalized: the dot product equals the cosine, and
the squared distance equals 2 − 2cos, which always moves opposite to cosine.

**Q: When would you deliberately use the raw dot product on unnormalized vectors?**
When the model was trained that way and encodes useful signal in the length
(e.g. popularity or quality in some recommender and retrieval models).
Normalizing would throw that signal away.

**Q: Similarity scores all sit between 0.75 and 0.85. What's going on and how do you set a threshold?**
The model is anisotropic: its vectors share a dominant direction, so every
pair looks similar and the real signal lives in a narrow band. Ranking still
works. For a threshold, label a few hundred pairs from your own data, sweep
the threshold, and pick the one that maximizes F1 (or matches your
precision/recall needs). Optionally mean-center the vectors. Redo it for
every model version.

**Q: What is the curse of dimensionality for nearest-neighbour search?**
For random high-dimensional data, the nearest and farthest points are almost
the same distance away, so "nearest" carries little information and exact
search gets expensive. Real embeddings have structure, and approximate
indexes trade a little recall for large speedups.


### 17. Training embedding models

From [`primer.ml.embeddings.contrastive`](../primer/ml/embeddings/contrastive.py).

**Q: How are embedding models trained?**
Contrastively, on (query, relevant passage) pairs: an encoder embeds both,
and InfoNCE rewards the pair's similarity over the similarity to other
passages in the batch (in-batch negatives) and to deliberately chosen hard
negatives.

**Q: Why do hard negatives matter so much?**
Random negatives are usually about something else, so a model can beat them
by topic matching alone. Hard negatives share the topic but don't answer
the question, forcing the model to learn the fine distinction that
retrieval actually needs.

**Q: What does the temperature do in InfoNCE?**
It divides the similarities before softmax. A small temperature makes the
softmax sharp, so small cosine differences produce large differences in
probability and gradient. Too small makes training unstable; too large
makes the model unable to become confident.

**Q: How does CLIP put images and text in the same space?**
It trains an image encoder and a text encoder together on image-caption
pairs with a symmetric contrastive loss: each image must pick its caption
from the batch and each caption must pick its image. Matching pairs end up
close in one shared space.

**Q: How would you adapt an embedding model to a company's jargon?**
Collect real (query, correct document) pairs from logs, mine hard negatives
from current search results, fine-tune with in-batch plus hard negatives,
evaluate recall@k on a held-out set against the base model, then re-embed
the corpus with the new model.


### 18. Dimensions and compression

From [`primer.ml.embeddings.compression`](../primer/ml/embeddings/compression.py).

**Q: How much memory do ten million 1,536-dimension float32 vectors need?**
10,000,000 × 1,536 × 4 bytes = 61.4 GB of raw vectors, plus index overhead
(e.g. ~1.3 GB of HNSW links at M = 16).

**Q: What makes Matryoshka embeddings truncatable, and how do you use that?**
They're trained with the loss applied to several prefixes at once, so the
first m numbers form a good embedding by themselves. Search with short
vectors for speed and memory, then re-score the top candidates with the
full vectors.

**Q: Scalar vs. binary quantization: what do you give up?**
int8 rounds each number to 256 levels, 4× smaller with a small recall loss.
Binary keeps only signs, 32× smaller, but recall drops a lot on its own; it
works as a first-stage shortlist followed by full-precision re-scoring.

**Q: Why not just use as many dimensions as possible?**
Gains flatten out while storage, memory and search time grow linearly.
A smaller model trained for your domain often beats a bigger generic one,
so benchmark on your own queries.


### 19. Vector indexes

From [`primer.ml.embeddings.ann`](../primer/ml/embeddings/ann.py).

**On the eight-point map, query at (5, 4): walk the HNSW search.**
Start at A on layer 1 (squared distance 41). Its layer-1 neighbors are E (5)
and G (10), so hop to E. No layer-1 neighbor of E is closer, so drop to
layer 0 at E. There, H (1) is closer, so hop to H. None of H's neighbors
beats 1, so the answer is H.

**PQ stores x = (0.9, 0.1, −0.2, 0.8) with the four compass directions as
each half's codebook. What are the codes, and what score does the query
(1, 0, 0, 1) get?**
Codes [0, 1]. The lookup tables are [1, 0, −1, 0] and [0, 1, 0, −1], so the
approximate score is 1 + 1 = 2, against an exact score of 0.9 + 0.8 = 1.7.

**Walk through an HNSW search and the knobs you'd tune.**
Enter at the top layer's entry point; greedily hop to whichever neighbor is
closest to the query until none is closer, then drop a layer at the same
node; at layer 0 run a beam search keeping `efSearch` candidates; return the
top k. Build-time knobs: M (links per node; memory and recall) and
efConstruction (graph quality; build time). Query-time knob: efSearch. Raise
it until recall@k against a flat index meets your target, then check p95
latency.

**Why does HNSW need a "diversity" rule when choosing neighbors?**
If a node linked to its M nearest points, in clustered data all M would sit
in one clump and the graph could split into islands that greedy search can't
cross. Keeping a candidate only if it's closer to the new node than to any
already-kept neighbor spreads links in different directions and keeps
bridges between clusters.

**Why do the upper layers of HNSW have so few nodes?**
Each node's top layer is drawn so that P(layer ≥ l) = M^−l. Each layer holds
about 1/M of the one below, like express stops on a subway line, so a few
long hops at the top cover the whole space.

**IVF: what happens as `nprobe` goes from 1 to `nlist`?**
Recall rises and the work grows roughly in proportion. At `nprobe = nlist`
it's an exact scan (plus the centroid comparisons).

**How does PQ score a vector without decompressing it?**
It precomputes, for each of the m pieces, the query piece's dot product with
all 256 codebook entries. A stored vector's score is then the sum of m table
lookups, one per code. The query stays exact; only the stored side is
approximate (asymmetric distance computation).

**You have 1 billion 768-dimension vectors. Why not HNSW over raw float32?**
The raw vectors alone are 10⁹ × 768 × 4 bytes ≈ 3 TB of RAM, before the
graph links. Use IVF-PQ (e.g. 64-byte codes ≈ 64 GB) with exact re-scoring
of a shortlist, shard the index across machines, or use a disk-based graph
index such as DiskANN.

**Why can an index that scores 0.99 recall on a benchmark do worse on your data?**
ANN performance depends on the data's structure. Uniformly random
high-dimensional vectors are the hardest case (all distances look alike),
while real embeddings are clustered. A benchmark with different structure,
dimension or size predicts little. Always measure on your own vectors.


### 20. Retrieval

From [`primer.ml.embeddings.retrieval`](../primer/ml/embeddings/retrieval.py).

**Documents "cat cat dog", "dog bird", "fish"; query "cat". What does BM25
give the first document?**
IDF(cat) = ln(1 + 2.5/1.5) = 0.981. With f = 2, |d| = 3, avgdl = 2, k₁ = 1.5
and b = 0.75 the denominator is 2 + 1.5 · 1.375 = 4.0625, so the score is
0.981 · 2 · 2.5 / 4.0625 = 1.207.

**A document is 1st in dense search and 3rd in BM25. What's its RRF score?**
1/61 + 1/63 ≈ 0.0323, versus 0.0164 for a document that is 1st in only one
list.

**Query words (1, 0) and (0, 1); document words (1, 0) and (0.6, 0.8). What's
the MaxSim score?**
max(1.0, 0.6) + max(0.0, 0.8) = 1.8.

**How many 50-word chunks with 10 words of overlap does a 130-word text make?**
⌈(130 − 10)/(50 − 10)⌉ = 3, starting at words 0, 40 and 80.

**What goes wrong if you forget an embedding model's "passage: " prefix when
indexing?**
Nothing errors: vectors look normal and scores look plausible, but documents
land where the model never aligned them with questions, and recall silently
drops (11 of 12 to 9 of 12 in the lesson's simulation). Only an evaluation
on labeled questions catches it.

**What do BM25's k₁ and b control?**
k₁ sets saturation: how fast extra mentions of a word stop adding score
(the credit per word can never exceed k₁ + 1). b sets length normalization:
0 ignores document length, 1 fully scales mentions by length relative to the
average. Typical values are k₁ ≈ 1.2 to 2 and b = 0.75.

**Bi-encoder vs. cross-encoder: how do you combine them in one pipeline?**
A bi-encoder embeds questions and documents separately, so documents are
embedded once and searched in milliseconds with an ANN index; it's fast but
never sees the pair together. A cross-encoder reads question and document as
one input, so it's accurate but costs one model pass per pair and can't
precompute anything. Combine them: retrieve the top 50 to 100 with the
bi-encoder (or hybrid search), rerank that shortlist with the cross-encoder,
and pass the top few to the model. Cap the shortlist to keep latency
predictable.

**Why does hybrid search beat pure vector search on company data?**
Company data is full of exact identifiers (error codes, product SKUs,
ticket numbers, names) that embeddings blur, and full of paraphrases and
jargon that keyword search misses. BM25 and dense search fail on *different*
questions, so fusing their rankings with RRF recovers the answers each one
misses.

**Why does RRF use ranks instead of scores?**
BM25 scores and cosine similarities live on unrelated scales, and those
scales vary by query and collection. Ranks are always comparable, so RRF
needs no tuning or score normalization. The constant k = 60 keeps the very
top rank from dominating.

**What's a hard negative, and how do you fix one in retrieval?**
A document on the right topic that doesn't answer the question, like the
password-reset how-to for "what are the password rules". Fix it with a
reranker that reads the question and document together, and when you
fine-tune an embedding model, train on hard negatives so it learns the
distinction (see [`primer.ml.embeddings.contrastive`](../primer/ml/embeddings/contrastive.py)).

**What does ColBERT trade to get better precision than a bi-encoder?**
Storage and some query cost. It keeps one vector per word instead of one per
document, often 50 to 200 times more vectors, and scores with MaxSim over
word pairs, in exchange for word-level matching with precomputed documents.

**Fixed-size vs. structure-aware chunking?**
Fixed-size windows are simple but cut through headings, sentences and
tables, separating facts from their context. Structure-aware chunking splits
at headings and paragraphs, keeps each heading with its content, and attaches
metadata. Parent-child retrieval searches small pieces but returns their
larger parent for context.

**Your RAG system gives confident wrong answers. How do you tell whether
retrieval is the cause?**
Build a small labeled set of real questions with known answer passages and
measure recall@k of retrieval alone. If the right passage isn't in the top k,
no prompt change will fix the answer; fix retrieval (hybrid, reranking,
chunking, prefixes, domain fine-tuning). If it is there, the problem is in
generation.


### 21. Clustering and matching

From [`primer.ml.embeddings.clustering`](../primer/ml/embeddings/clustering.py).

**Q: How does k-means work, and what are its limitations?**
Alternate assigning each point to its nearest centre and moving each centre
to the mean of its points until nothing changes. You must choose k, results
depend on the start (k-means++ helps), and it assumes round, similar-sized
clusters.

**Q: When would you choose HDBSCAN over k-means for text embeddings?**
When you don't know how many groups exist, clusters have different shapes
and densities, and some items belong nowhere (support tickets, logs). It
finds the number of clusters itself and labels outliers as noise.

**Q: Why can't you trust distances in a UMAP or t-SNE plot?**
They preserve local neighbourhoods and deliberately distort everything else,
so gaps between clusters and cluster sizes don't reflect the real space.

**Q: How would you route requests to the right agent or tool with embeddings?**
Represent each route by the centroid of example requests, send each new
request to the most similar centroid, and fall back to a human or general
assistant when the best similarity is below a calibrated threshold.

**Q: What can go wrong with a semantic cache?**
A loose threshold returns the answer to a *different* question; missing
context keys leak one user's or tenant's answer to another; missing expiry
serves stale facts. Filter by context first, then match strictly, and
expire entries.


### 22. Embeddings in production

From [`primer.ml.embeddings.operations`](../primer/ml/embeddings/operations.py).

**Q: You need to switch embedding models on a 50-million-document index. Plan the migration.**
Create a versioned index for the new model and dual-write new documents to
both. Backfill by re-embedding existing documents in restartable batches
(estimate: 50M × 500 tokens = 25B tokens; at 1M tokens/s that's about 7 hours).
Compare recall on a golden set in shadow; cut over the live alias only if the
new model is at least as good; keep the old index for fast rollback, then
retire it.

**Q: Why can't you compare vectors from two different embedding models?**
Each model defines its own coordinate system. Even with the same number of
dimensions, the same text lands in unrelated places, so similarity between
the two spaces is meaningless.

**Q: A general embedding model performs poorly on a client's internal documents. What do you do?**
Build a small eval set from real queries and the documents that resolved
them, test several models on it, add hybrid (keyword + dense) search for
exact terms, and fine-tune an embedding model on domain pairs with hard
negatives if needed.

**Q: Your RAG system gives confident wrong answers. What's your first diagnostic step?**
Split retrieval from generation: for failing questions, check whether a
relevant document was in the retrieved top k. If not, it's retrieval (fix
search); if yes, it's generation (fix prompt and context). Track recall@k on
a labeled set continuously.


## Part 2: building systems people rely on


### 23. Talking to a model

From [`primer.agents.llm`](../primer/agents/llm.py).

**Walk through exactly what happens when a model "uses a tool".**
You send tool definitions (name, description, JSON Schema) with the
messages. The model replies with a `tool_use` block and `stop_reason:
tool_use`. Your code validates the arguments, runs the function, and sends a
new request with the history plus a `tool_result` block carrying the same
id. The model then answers or asks for another tool.

**Why is the model never a security boundary for tool use?**
It only produces requests. Everything that actually happens goes through
your code, so validation, permissions, approvals and rate limits all belong
there, and a manipulated model can only ever ask.

**The model asks for three tools in one reply. How do you send the results?**
Run them (concurrently if independent) and return all three `tool_result`
blocks in a single user message, each with its own `tool_use_id`. For one
that failed, return a `tool_result` with `is_error: true` and an actionable
message rather than dropping it.

**A 20-step agent loop costs far more than 20 times a single call. Why?**
Each call re-sends the full, growing history, so the input billed is a sum
that grows with the square of the number of steps. Caching the stable prefix
and trimming tool outputs attack exactly this.

**Why test agents against a scripted model at all?**
Real models are non-deterministic and rarely misbehave on cue. A scripted
model reproduces loops, bad arguments and injected instructions exactly, so
the code that must handle them can be tested every time.


### 24. Orchestration

From [`primer.agents.orchestration`](../primer/agents/orchestration.py).

**Q: When would you use a fixed workflow instead of an agent? Give an example.**
A: When the steps are known in advance and the variation is *inside* each
step, not in which steps happen. Invoice intake is the example: classify,
extract, validate, approve, post. A workflow is cheaper, faster, testable
and auditable. The model is used where judgement is needed (classification,
extraction), and code owns the flow. Reach for an agent only when the path
genuinely depends on what's discovered along the way.

**Q: Single agent vs. multi-agent for a document-processing workflow: argue both sides.**
A: *For one agent (or a workflow):* documents flow through the same steps,
hand-offs lose context, multi-agent multiplies tokens, and one trace is far
easier to debug. *For several agents:* documents are independent and can be
processed in parallel, each document or section may overflow a single
context, and specialists (tables, legal clauses, figures) benefit from
focused instructions and tools. A common answer is a workflow that fans out
per document to a focused worker, which is orchestrator-workers rather than free-form
agents talking to each other.

**Q: Your router's classifier sometimes returns labels that aren't in your list. What do you do?**
A: Normalize (case, whitespace), accept only known labels, and send
everything else to a safe fallback. Log the misses and add them to the
classifier's evaluation set. Consider structured output with an `enum` so
the model can only produce valid labels.

**Q: Why checkpoint after every step of a long workflow?**
A: So a crash costs one step, not the run. Resuming reuses completed work
(no repeated model calls, no double-posted invoices when combined with
idempotency keys) and avoids getting a different answer on the rerun.


### 25. The agent loop

From [`primer.agents.agent_loop`](../primer/agents/agent_loop.py).

**Q: An agent in production suddenly costs 5x more per task. What do you check first?**
A: Steps per task and tokens per step, from traces. A jump usually means a loop
(the same call repeated), a tool that started returning huge payloads, or a
prompt change that stopped the model from recognizing "done". Step budgets and
loop detection cap the damage; alerts on tokens-per-task catch it early.

**Q: A tool throws an exception mid-run. What should the loop do?**
A: Catch it and return a `tool_result` with `is_error: true` and a message the
model can act on ("date must be YYYY-MM-DD, got 'next friday'"). The model
usually fixes its next call. Crashing the loop throws away the work so far;
swallowing the error silently makes the model guess.

**Q: Why return all parallel tool results in one message?**
A: The API pairs each `tool_use` with its `tool_result` by id in the next user
turn. Splitting results across several messages breaks that pairing and
teaches the model to stop making parallel calls, which slows every run.

**Q: When should an agent hand off to a human?**
A: When it lacks authority (refunds over a limit), lacks information after
reasonable search, detects conflicting sources, or hits a budget. Make hand-off a
tool, so it is an explicit, logged outcome rather than a vague final answer.


### 26. Tools

From [`primer.agents.tools`](../primer/agents/tools.py).

**Q: How do you design tools so the model picks the right one with correct arguments?**
A: Give each tool one clear job and a description that says what it does,
when to use it and when not to. Keep parameters few, use `enum`s for closed
sets, and put the format in the schema description ("date as YYYY-MM-DD").
Prefer one high-level tool over several low-level ones. Validate every call
and return actionable errors so the model can self-correct. Measure tool
selection accuracy in evals, and load tools dynamically when there are many.

**Q: The schema says `customer_id` must match `^C-\d+$`. Is that enough validation?**
A: No. The schema checks shape, not truth. `C-999` is well-formed and may not
exist. Add a business-rule check before acting, and return an error that
tells the model how to find the right id.

**Q: A refund call timed out. Should the agent retry?**
A: Only if the call is idempotent. Send an idempotency key with every write,
and the retry returns the original result instead of refunding twice.

**Q: Why not give the agent one admin credential for everything?**
A: Blast radius. If the agent is tricked (for example by prompt injection in
a document it reads), it can do anything the credential allows. Scope
credentials per tool and per agent, and put irreversible actions behind human
approval.


### 27. Model Context Protocol

From [`primer.agents.mcp`](../primer/agents/mcp.py).

**Q: What problem does MCP solve, and what does it not solve?**
A: It removes the A × T integration problem. Build a connector once as a
server and every compatible app can use it. It doesn't make tools safe,
well-described or correctly permissioned. Those are still your job.

**Q: What's the difference between a tool, a resource and a prompt in MCP?**
A: Who decides. The model chooses to call tools, the application
chooses which resources to read into context, and the user picks prompts.

**Q: How would you vet a third-party MCP server before letting an agent use it?**
A: Read and scan every tool description for hidden instructions. Pin the
approved definitions and re-review on any change. Run it with the narrowest
credentials, ideally the user's own delegated token. Require approval for
destructive tools, and log every call.

**Q: Why should a server act with the user's token rather than its own service account?**
A: With its own powerful account, the server can be steered into doing things the
user isn't allowed to do (a confused deputy). With the user's token, the
system of record enforces the user's real permissions.


### 28. Retrieval-augmented generation

From [`primer.agents.rag`](../primer/agents/rag.py).

**Your RAG system gives confident wrong answers. Debug it step by step.**
Take the failing questions (and build a labelled set if you don't have one).
First, is the right passage in the index at all? If not, it's ingestion:
parsing, chunking or permission sync. Second, is it in the top k? Measure
recall@k; if not, fix retrieval: hybrid search, a reranker, query rewriting
or HyDE, metadata filters, contextual chunks, domain-tuned embeddings.
Third, did it survive into the final context? If not, fix reranking or the
context budget. Only then look at generation: instructions to answer only
from sources and to decline otherwise, citation checks, stale or
contradictory sources. Add every case to the golden set.

**Why does hybrid search beat pure vector search on enterprise data?**
Enterprise questions are full of exact identifiers (error codes, product
names, ticket numbers) that embeddings blur, and full of paraphrases that
keyword search misses. Hybrid gets both, and RRF merges the rankings without
having to reconcile incompatible score scales.

**How do you make RAG respect document permissions?**
Copy each document's access-control list onto every chunk at ingestion,
filter candidates by the user's groups before ranking, and sync permission
changes from the source system promptly. Never filter after generation: the
model has already used the restricted text.

**When would you use agentic RAG instead of a single retrieval step?**
For multi-part or vague questions, and when the model needs to judge whether
results are good enough and search again. It costs more calls and latency,
so keep single-shot retrieval for simple lookups.

**RAG or fine-tuning for a company knowledge assistant?**
RAG for knowledge: it updates by re-indexing, cites sources, and respects
permissions. Fine-tuning teaches behaviour and format, not facts that
change. Most systems are RAG plus a good prompt (see
[`primer.ml.training_stages`](../primer/ml/training_stages.py)).


### 29. Context engineering

From [`primer.agents.context`](../primer/agents/context.py).

**Why not just use the whole 1M-token window?**
Cost and latency grow with input tokens on every call, and an agent resends
its context at every step. Quality suffers too: irrelevant material
distracts the model, and information buried mid-context is used less
reliably. A tight, relevant context is cheaper, faster and usually more
accurate.

**A long-running agent gets worse the longer a session runs. What's
happening and what do you do?**
Context rot: the window fills with stale turns and verbose tool results, so
the signal thins while cost rises. Summarize older turns, compress tool
results to the fields that matter, move durable facts into memory that's
retrieved on demand, and for very long tasks restart with a clean context
plus a structured handoff of the task state.

**Your prompt cache hit rate is zero. What do you check first?**
Anything that changes near the top: a timestamp or request id in the system
prompt, tool definitions serialized in a nondeterministic order, per-user
data mixed into the "static" part. Move it all after the stable content and
confirm with the provider's cache usage counters.

**How do you structure a prompt that includes retrieved documents?**
Stable instructions first, then each document in its own escaped tag with an
id and metadata, then the question last. Tell the model to answer only from
the documents and to cite ids. The structure makes citations checkable and
makes it harder for document text to pose as instructions.


### 30. Memory

From [`primer.agents.memory`](../primer/agents/memory.py).

**Q: How would you design long-term memory for a multi-tenant agent platform?**
A: Every read and write takes a scope (tenant, user) from the authenticated
session, and storage is partitioned by it: separate namespaces, collections
or row-level security, never a global index filtered after the search.
Store typed records (episodic, semantic with keys, procedural) with
embeddings, source and time. Apply a write policy (durable, non-secret),
supersede conflicting facts instead of overwriting, retrieve the top few by
similarity within scope, and support export and deletion per user. Test
isolation with adversarial queries.

**Q: A long support chat gets worse and more expensive over time. Why, and what do you do?**
A: Every call re-sends the whole history, so cost rises, and models use
details in the middle of long contexts less reliably. Keep a budget: recent
turns verbatim, older ones summarized, important facts promoted to
long-term memory, or reset with a written hand-off.

**Q: The user corrects a fact the agent remembered. What should happen?**
A: Store the new value under the same key, mark the old one as superseded
by the new (don't delete it silently), and recall only current records.
Keep the source of each so the change is explainable.

**Q: Why keep task state in a database when the model can "remember" progress in the conversation?**
A: The conversation is lost on a crash, and it can be summarized, trimmed or
misread. A database is authoritative, survives restarts, can be queried by
people and monitoring, and lets code enforce rules (no skipping steps)
that a prompt can only request.


### 31. Planning

From [`primer.agents.planning`](../primer/agents/planning.py).

**Q: Each step of your agent is 95% reliable. How reliable is a 20-step task, and what do you do about it?**
A: $0.95^{20} \approx 36\%$. Cut the steps (higher-level tools), add
external checks after key steps with a retry of just that step, checkpoint
so failures don't restart the run, and put human review where a mistake is
expensive. With perfect checks and one retry, per-step reliability becomes
99.75%, and 20 steps succeed about 95% of the time.

**Q: Why isn't "ask the model to double-check its work" enough?**
A: The reviewer shares the author's blind spots, and self-critique can
confidently reinforce a wrong answer. Checks outside the model (tests,
schema validation, a query against the source of truth, a separate grader
model with a rubric) produce concrete failures the model can act on.

**Q: When is plan-and-execute better than deciding one step at a time?**
A: For long, multi-part tasks where staying on track matters and where
showing the plan to a person before acting is valuable. Always pair it with
replanning. A plan is a hypothesis, and the first failed step is evidence.

**Q: What makes a subtask "good"?**
A: It's small, concrete and verifiable. It has a definition of done that code
can check, and its output is saved so a later failure doesn't redo it.


### 32. Evaluation

From [`primer.agents.evals`](../primer/agents/evals.py).

**How do you evaluate an agent whose correct path isn't fixed?**
Grade what must be true at the end, not how it got there: the end state of
the systems it touched (the database row, the ticket, the file), the facts
the answer must contain, and constraints on the path (required tools used,
forbidden tools avoided, step and cost limits). Add trajectory metrics
(tool selection accuracy, argument correctness, steps, tokens) as
diagnostics. For open-ended outputs, use a rubric-based LLM judge that's
calibrated against human labels.

**Your LLM judge agrees with humans 85% of the time. Is it good?**
Not necessarily. If 85% of answers are good, a judge that always says
"pass" also agrees 85% of the time. Compute Cohen's kappa, which subtracts
chance agreement; look at the disagreements; and recheck whenever the judge
model or rubric changes.

**A new prompt improves the average score. Do you ship it?**
Only after checking for regressions task by task, cost and latency changes,
and whether the improvement is bigger than run-to-run noise (rerun,
or use more cases). An average can go up while a critical case, like an
over-limit refund, breaks.

**How do you build the first eval set for a new agent?**
Collect 30 to 50 real requests (from logs, support tickets, or domain
experts), write down for each what must be true afterwards, and write code
checks for as many as possible. Run it on every change from day one. Then
grow it from production: every flagged failure becomes a case.

**What would you monitor once the agent is live?**
Task success (from outcome checks and sampled human review), implicit
dissatisfaction (rephrase, retry, abandon, escalate rates), cost per
successful task, p95 latency, tool error rates, and drift in any of these
after a deploy.


### 33. Guardrails

From [`primer.agents.guardrails`](../primer/agents/guardrails.py).

**An agent reads a user's email and can also send email. How do you defend
it against prompt injection?**
Assume the model will sometimes be fooled and design so that being fooled
is harmless. Split it: a reader agent with read-only tools summarizes each
email into a fixed schema, and nothing in that summary is treated as an
instruction. A policy layer compares any requested action with the user's
own request and blocks sends to new or external recipients, bulk forwards
and sensitive attachments; anything high-stakes goes to a person. The actor
agent that holds `send_email` sees only approved, structured actions, never
the raw email. Add input detection and output checks as extra layers,
rate-limit sends, and keep an audit log.

**Why isn't "the system prompt says to ignore instructions in documents"
enough?**
The model reads instructions and data in one stream of text, and attackers
can phrase an instruction endlessly many ways: other languages, encodings,
role-play, pieces split across documents. Prompting lowers the success rate
but can't bring it to zero, so it can't be the control protecting
irreversible actions.

**What's the difference between schema validation and semantic
validation?**
Schema validation checks shape: required fields, types, allowed values.
Semantic validation checks meaning against the world: does this customer
exist, is the refund below the order total, is the recipient allowed.
Structured-output modes can guarantee the first; you always write the
second.

**How do you check that an answer is grounded?**
Split it into claims, and check each against the retrieved sources: word
overlap as a cheap first pass (as here), then an entailment model or an LLM
judge asking "does this passage support this claim?". Block or flag
unsupported claims, and require citations so people can verify.

**Why validate card numbers with the Luhn check instead of just matching 16
digits?**
Because order numbers, account IDs and tracking numbers share the shape. A
filter that redacts all of them destroys useful data and gets switched off.
Every real card number passes Luhn and only about 10% of random digit
strings do, so validation removes roughly 90% of false alarms at no cost.


### 34. Cost and latency

From [`primer.agents.cost`](../primer/agents/cost.py).

**How would you cut the cost per task by 5x without hurting quality?**
First measure: cost per successful task, broken down by step, model, input
vs. output, and cached vs. uncached tokens, with an eval set to hold
quality fixed. Then, in order: restructure prompts for prompt caching;
compress tool results and send only the top reranked chunks; ask for
concise output; route simple steps (classification, extraction,
formatting) to a small model, checking the eval before and after; move
anything non-interactive to a batch API; cap steps and tokens per task.
On the sample workload here that's about 8x, and every step after the
first three is checked against the eval set.

**Why can a cheaper model cost more?**
Because failures cost money too: retries, human cleanup, lost customers.
At 60% success with a \$2 human fix, a \$0.002 call costs \$0.80 per task;
a \$0.010 call at 95% costs \$0.11.

**What are the risks of a semantic cache?**
Returning a confident, wrong answer to a question that's similar but not
the same ("sick days" vs "vacation days"), and serving stale answers after
facts change. Mitigate with high thresholds, exact-match guards on IDs,
numbers and negation, per-intent scoping, a TTL, and a measured wrong-hit
rate on labelled pairs before enabling it.

**Your average tokens per task doubled overnight. What do you check?**
Whether a deploy changed a prompt or a tool (a bigger tool output, a new
retrieval setting), whether an agent is looping (the same tool called with
the same arguments), and whether the prompt cache hit rate dropped
(something volatile moved to the top). Traces make this a lookup rather
than a guess ([`primer.agents.observability`](../primer/agents/observability.py)).


### 35. Observability

From [`primer.agents.observability`](../primer/agents/observability.py).

**What would you record for every agent run?**
A tree of spans: the task at the root, and each model call, tool call and
retrieval as children, with inputs and outputs (redacted of personal data),
token counts, latency, cost, model id, prompt version, tool arguments and
results, errors, and the user and tenant it ran for. That's enough to
replay any decision and to aggregate cost and quality.

**A user says the agent gave a wrong answer yesterday. How do you find
out why?**
Look up the trace by conversation or user id and walk it: did retrieval
return the right document, did the model pick the right tool with the right
arguments, did a tool error, did a guardrail fire? The first error span or
the first surprising output shows where the chain broke, and the trace
becomes a new golden test case.

**Why use OpenTelemetry instead of a vendor's SDK directly?**
Portability and fit: the organisation's existing monitoring already speaks
OTel, the GenAI conventions give every tool the same attribute names, and
changing vendors becomes a collector configuration change rather than a
code change.

**What should you be careful about when logging prompts and outputs?**
They contain user data. Redact personal data before export, restrict who
can read traces, set retention limits, and keep tenants' traces separated.


### 36. Safe deployment

From [`primer.agents.deployment`](../primer/agents/deployment.py).

**How would you roll out an agent that takes real actions in a company's
ERP system (its finance and operations system of record)?**
Start read-only in shadow mode on real transactions, comparing proposals
with what staff actually did, broken down by action type. Give the agent a
dedicated service identity with the narrowest permissions, a sandbox or
dry-run mode for writes, and idempotency keys so retries can't double-post.
Move to proposing actions that a person approves in their existing
workflow, starting with the low-value, reversible action types that scored
best in shadow. Grant autonomy only for those, with value thresholds that
still require approval, per-tenant kill switches, rate limits on writes, a
canary rollout for every prompt or model change, eval gates in CI, and a
tamper-evident audit log tied to traces. Demote automatically on incidents
or drift, and keep the finance team involved throughout.

**Why start in shadow mode instead of with human approval?**
Shadow mode costs the humans nothing extra and produces a clean comparison
against what they actually did, at full volume, before the agent can
influence anything. Approval mode adds review load and can bias reviewers
toward accepting the agent's suggestion.

**What does a canary protect against that offline evals don't?**
Real traffic: inputs, integrations and user behaviour the golden set didn't
anticipate. Evals gate the release; the canary limits the damage from
whatever the evals missed.

**Why a hash chain rather than an ordinary log table?**
Anyone with write access can quietly edit an ordinary row. With a hash
chain, any edit or deletion breaks every later link, so tampering is
detectable, and anchoring the latest hash somewhere separate (or using
write-once storage) makes it provable.


### 37. Why the hard ones fail

From [`primer.agents.failures`](../primer/agents/failures.py).

**Your agent works in demos and fails half the time in production. Where do
you look first?**
At the length of real tasks versus demo tasks (compounding error), and at
traces for the first failing step: retrieval misses, wrong tools or
arguments, tool errors from integrations, context growth. Then fix
per-step reliability where it's lowest, add verification after key steps,
and put the failing cases into the eval set.

**Why add jitter to retries?**
Without it, every client that failed at the same moment retries at the same
moment, producing synchronized waves of load that keep an overloaded
service down. Random delays spread the retries out.

**When should you not retry?**
When the error can't go away on its own: invalid input, permission denied,
not found. And never retry a non-idempotent write without an idempotency
key, or you'll duplicate its effect.

**What's the difference between a retry and a circuit breaker?**
A retry handles a blip for one request. A breaker handles an outage across
many requests: it stops sending traffic to a failing dependency, fails fast
so callers can fall back, and probes for recovery.

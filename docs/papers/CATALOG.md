# Annotated papers: catalog

Each landmark paper behind a lesson gets an **annotated companion**: an
interactive HTML page at `docs/papers/<slug>.html` that walks the paper
section by section, decodes every equation symbol on hover, redraws its key
figures as interactive diagrams, and links each idea to the lesson that
builds it in code. Companions quote the paper only briefly and link to the
original for the full text; see [`docs/papers/README.md`](README.md) for how to author one.

This table is the single source of truth for slugs. Lessons link to a
companion as `docs/papers/<slug>.html` in their `## The papers behind this
lesson` section.

| Slug | Paper | Primary source | Lessons |
|---|---|---|---|
| `attention-is-all-you-need` | Vaswani et al., *Attention Is All You Need* (2017) | https://arxiv.org/abs/1706.03762 | [ml.attention](../../primer/ml/attention.py), [ml.transformer](../../primer/ml/transformer.py), [ml.positional](../../primer/ml/positional.py), [ml.big_picture](../../primer/ml/big_picture.py), [ml.cnn_rnn](../../primer/ml/cnn_rnn.py) |
| `adam` | Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2014) | https://arxiv.org/abs/1412.6980 | [ml.optimizers](../../primer/ml/optimizers.py) |
| `resnet` | He et al., *Deep Residual Learning for Image Recognition* (2015) | https://arxiv.org/abs/1512.03385 | [ml.deep_nets](../../primer/ml/deep_nets.py), [ml.cnn_rnn](../../primer/ml/cnn_rnn.py) |
| `layer-norm` | Ba, Kiros & Hinton, *Layer Normalization* (2016) | https://arxiv.org/abs/1607.06450 | [ml.deep_nets](../../primer/ml/deep_nets.py), [ml.transformer](../../primer/ml/transformer.py) |
| `dropout` | Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (2014) | https://jmlr.org/papers/v15/srivastava14a.html | [ml.regularization](../../primer/ml/regularization.py) |
| `bpe-subwords` | Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword Units* (2015) | https://arxiv.org/abs/1508.07909 | [ml.tokenization](../../primer/ml/tokenization.py) |
| `roformer` | Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021) | https://arxiv.org/abs/2104.09864 | [ml.positional](../../primer/ml/positional.py) |
| `gpt-3` | Brown et al., *Language Models are Few-Shot Learners* (2020) | https://arxiv.org/abs/2005.14165 | [ml.big_picture](../../primer/ml/big_picture.py), [ml.training_stages](../../primer/ml/training_stages.py) |
| `scaling-laws` | Kaplan et al., *Scaling Laws for Neural Language Models* (2020), with Hoffmann et al., *Training Compute-Optimal LLMs* (2022) | https://arxiv.org/abs/2001.08361 and https://arxiv.org/abs/2203.15556 | [ml.training_stages](../../primer/ml/training_stages.py), [ml.transformer](../../primer/ml/transformer.py) |
| `flashattention` | Dao et al., *FlashAttention* (2022) | https://arxiv.org/abs/2205.14135 | [ml.attention](../../primer/ml/attention.py), [ml.inference](../../primer/ml/inference.py) |
| `speculative-decoding` | Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding* (2022) | https://arxiv.org/abs/2211.17192 | [ml.inference](../../primer/ml/inference.py) |
| `paged-attention` | Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (2023) | https://arxiv.org/abs/2309.06180 | [ml.inference](../../primer/ml/inference.py) |
| `instructgpt` | Ouyang et al., *Training language models to follow instructions with human feedback* (2022) | https://arxiv.org/abs/2203.02155 | [ml.training_stages](../../primer/ml/training_stages.py) |
| `dpo` | Rafailov et al., *Direct Preference Optimization* (2023) | https://arxiv.org/abs/2305.18290 | [ml.training_stages](../../primer/ml/training_stages.py) |
| `lora` | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021), with QLoRA (2023) | https://arxiv.org/abs/2106.09685 and https://arxiv.org/abs/2305.14314 | [ml.training_stages](../../primer/ml/training_stages.py) |
| `distillation` | Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network* (2015) | https://arxiv.org/abs/1503.02531 | [ml.training_stages](../../primer/ml/training_stages.py), [agents.cost](../../primer/agents/cost.py) |
| `word2vec` | Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013), with *Distributed Representations of Words and Phrases* (2013) | https://arxiv.org/abs/1301.3781 and https://arxiv.org/abs/1310.4546 | [ml.embeddings.word2vec](../../primer/ml/embeddings/word2vec.py) |
| `sentence-bert` | Reimers & Gurevych, *Sentence-BERT* (2019) | https://arxiv.org/abs/1908.10084 | [ml.embeddings.contrastive](../../primer/ml/embeddings/contrastive.py), [ml.embeddings.retrieval](../../primer/ml/embeddings/retrieval.py) |
| `dpr` | Karpukhin et al., *Dense Passage Retrieval for Open-Domain QA* (2020) | https://arxiv.org/abs/2004.04906 | [ml.embeddings.contrastive](../../primer/ml/embeddings/contrastive.py), [ml.embeddings.retrieval](../../primer/ml/embeddings/retrieval.py) |
| `clip` | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (2021) | https://arxiv.org/abs/2103.00020 | [ml.embeddings.contrastive](../../primer/ml/embeddings/contrastive.py), [ml.losses](../../primer/ml/losses.py) |
| `matryoshka` | Kusupati et al., *Matryoshka Representation Learning* (2022) | https://arxiv.org/abs/2205.13147 | [ml.embeddings.compression](../../primer/ml/embeddings/compression.py) |
| `hnsw` | Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using HNSW graphs* (2016) | https://arxiv.org/abs/1603.09320 | [ml.embeddings.ann](../../primer/ml/embeddings/ann.py) |
| `colbert` | Khattab & Zaharia, *ColBERT* (2020) | https://arxiv.org/abs/2004.12832 | [ml.embeddings.retrieval](../../primer/ml/embeddings/retrieval.py) |
| `rag` | Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020) | https://arxiv.org/abs/2005.11401 | [agents.rag](../../primer/agents/rag.py) |
| `hyde` | Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (2022) | https://arxiv.org/abs/2212.10496 | [agents.rag](../../primer/agents/rag.py) |
| `lost-in-the-middle` | Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023) | https://arxiv.org/abs/2307.03172 | [agents.context](../../primer/agents/context.py) |
| `react` | Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022) | https://arxiv.org/abs/2210.03629 | [agents.agent_loop](../../primer/agents/agent_loop.py), [agents.orchestration](../../primer/agents/orchestration.py), [agents.llm](../../primer/agents/llm.py) |
| `toolformer` | Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* (2023) | https://arxiv.org/abs/2302.04761 | [agents.tools](../../primer/agents/tools.py), [agents.llm](../../primer/agents/llm.py) |
| `reflexion` | Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning* (2023) | https://arxiv.org/abs/2303.11366 | [agents.planning](../../primer/agents/planning.py) |
| `llm-as-judge` | Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023) | https://arxiv.org/abs/2306.05685 | [agents.evals](../../primer/agents/evals.py), [ml.metrics](../../primer/ml/metrics.py) |

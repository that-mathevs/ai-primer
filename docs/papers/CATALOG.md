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
| `gpt-3` | Brown et al., *Language Models are Few-Shot Learners* (2020) | https://arxiv.org/abs/2005.14165 | [ml.big_picture](../../primer/ml/big_picture.py), [ml.training_stages](../../primer/ml/training_stages.py), [ml.benchmarks](../../primer/ml/benchmarks.py) |
| `scaling-laws` | Kaplan et al., *Scaling Laws for Neural Language Models* (2020), with Hoffmann et al., *Training Compute-Optimal LLMs* (2022) | https://arxiv.org/abs/2001.08361 and https://arxiv.org/abs/2203.15556 | [ml.training_stages](../../primer/ml/training_stages.py), [ml.transformer](../../primer/ml/transformer.py), [ml.pretraining](../../primer/ml/pretraining.py) |
| `flashattention` | Dao et al., *FlashAttention* (2022) | https://arxiv.org/abs/2205.14135 | [ml.attention](../../primer/ml/attention.py), [ml.inference](../../primer/ml/inference.py), [ml.hardware](../../primer/ml/hardware.py) |
| `speculative-decoding` | Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding* (2022) | https://arxiv.org/abs/2211.17192 | [ml.inference](../../primer/ml/inference.py) |
| `paged-attention` | Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (2023) | https://arxiv.org/abs/2309.06180 | [ml.inference](../../primer/ml/inference.py) |
| `instructgpt` | Ouyang et al., *Training language models to follow instructions with human feedback* (2022) | https://arxiv.org/abs/2203.02155 | [ml.training_stages](../../primer/ml/training_stages.py), [ml.reinforcement](../../primer/ml/reinforcement.py) |
| `dpo` | Rafailov et al., *Direct Preference Optimization* (2023) | https://arxiv.org/abs/2305.18290 | [ml.training_stages](../../primer/ml/training_stages.py) |
| `lora` | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021), with QLoRA (2023) | https://arxiv.org/abs/2106.09685 and https://arxiv.org/abs/2305.14314 | [ml.training_stages](../../primer/ml/training_stages.py) |
| `distillation` | Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network* (2015) | https://arxiv.org/abs/1503.02531 | [ml.training_stages](../../primer/ml/training_stages.py), [agents.cost](../../primer/agents/cost.py) |
| `word2vec` | Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013), with *Distributed Representations of Words and Phrases* (2013) | https://arxiv.org/abs/1301.3781 and https://arxiv.org/abs/1310.4546 | [ml.embeddings.word2vec](../../primer/ml/embeddings/word2vec.py) |
| `sentence-bert` | Reimers & Gurevych, *Sentence-BERT* (2019) | https://arxiv.org/abs/1908.10084 | [ml.embeddings.contrastive](../../primer/ml/embeddings/contrastive.py), [ml.embeddings.retrieval](../../primer/ml/embeddings/retrieval.py) |
| `dpr` | Karpukhin et al., *Dense Passage Retrieval for Open-Domain QA* (2020) | https://arxiv.org/abs/2004.04906 | [ml.embeddings.contrastive](../../primer/ml/embeddings/contrastive.py), [ml.embeddings.retrieval](../../primer/ml/embeddings/retrieval.py) |
| `clip` | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (2021) | https://arxiv.org/abs/2103.00020 | [ml.embeddings.contrastive](../../primer/ml/embeddings/contrastive.py), [ml.losses](../../primer/ml/losses.py), [ml.generative.multimodal](../../primer/ml/generative/multimodal.py) |
| `matryoshka` | Kusupati et al., *Matryoshka Representation Learning* (2022) | https://arxiv.org/abs/2205.13147 | [ml.embeddings.compression](../../primer/ml/embeddings/compression.py) |
| `hnsw` | Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using HNSW graphs* (2016) | https://arxiv.org/abs/1603.09320 | [ml.embeddings.ann](../../primer/ml/embeddings/ann.py) |
| `colbert` | Khattab & Zaharia, *ColBERT* (2020) | https://arxiv.org/abs/2004.12832 | [ml.embeddings.retrieval](../../primer/ml/embeddings/retrieval.py) |
| `rag` | Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020) | https://arxiv.org/abs/2005.11401 | [agents.rag](../../primer/agents/rag.py) |
| `hyde` | Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (2022) | https://arxiv.org/abs/2212.10496 | [agents.rag](../../primer/agents/rag.py) |
| `lost-in-the-middle` | Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023) | https://arxiv.org/abs/2307.03172 | [agents.context](../../primer/agents/context.py) |
| `react` | Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022) | https://arxiv.org/abs/2210.03629 | [agents.agent_loop](../../primer/agents/agent_loop.py), [agents.orchestration](../../primer/agents/orchestration.py), [agents.llm](../../primer/agents/llm.py), [agents.coding_agents](../../primer/agents/coding_agents.py) |
| `toolformer` | Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* (2023) | https://arxiv.org/abs/2302.04761 | [agents.tools](../../primer/agents/tools.py), [agents.llm](../../primer/agents/llm.py) |
| `reflexion` | Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning* (2023) | https://arxiv.org/abs/2303.11366 | [agents.planning](../../primer/agents/planning.py) |
| `llm-as-judge` | Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023) | https://arxiv.org/abs/2306.05685 | [agents.evals](../../primer/agents/evals.py), [ml.metrics](../../primer/ml/metrics.py), [ml.benchmarks](../../primer/ml/benchmarks.py) |
| `general-language-assistant` | Askell et al., *A General Language Assistant as a Laboratory for Alignment* (2021) | https://arxiv.org/abs/2112.00861 | [ml.alignment](../../primer/ml/alignment.py) |
| `constitutional-ai` | Bai et al., *Constitutional AI: Harmlessness from AI Feedback* (2022) | https://arxiv.org/abs/2212.08073 | [ml.alignment](../../primer/ml/alignment.py) |
| `red-teaming-lms` | Perez et al., *Red Teaming Language Models with Language Models* (2022) | https://arxiv.org/abs/2202.03286 | [ml.alignment](../../primer/ml/alignment.py) |
| `red-teaming-reduce-harms` | Ganguli et al., *Red Teaming Language Models to Reduce Harms* (2022) | https://arxiv.org/abs/2209.07858 | [ml.alignment](../../primer/ml/alignment.py) |
| `sycophancy` | Sharma et al., *Towards Understanding Sycophancy in Language Models* (2023) | https://arxiv.org/abs/2310.13548 | [ml.alignment](../../primer/ml/alignment.py) |
| `reward-model-overoptimization` | Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization* (2022) | https://arxiv.org/abs/2210.10760 | [ml.alignment](../../primer/ml/alignment.py), [ml.reinforcement](../../primer/ml/reinforcement.py) |
| `roofline` | Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model* (2009) | https://doi.org/10.1145/1498765.1498785 | [ml.hardware](../../primer/ml/hardware.py) |
| `mixed-precision-training` | Micikevicius et al., *Mixed Precision Training* (2017) | https://arxiv.org/abs/1710.03740 | [ml.hardware](../../primer/ml/hardware.py), [ml.pretraining](../../primer/ml/pretraining.py) |
| `bfloat16` | Kalamkar et al., *A Study of BFLOAT16 for Deep Learning Training* (2019) | https://arxiv.org/abs/1905.12322 | [ml.hardware](../../primer/ml/hardware.py) |
| `fp8-formats` | Micikevicius et al., *FP8 Formats for Deep Learning* (2022) | https://arxiv.org/abs/2209.05433 | [ml.hardware](../../primer/ml/hardware.py) |
| `megatron-lm` | Shoeybi et al., *Megatron-LM* (2019) | https://arxiv.org/abs/1909.08053 | [ml.hardware](../../primer/ml/hardware.py), [ml.pretraining](../../primer/ml/pretraining.py) |
| `horovod` | Sergeev & Del Balso, *Horovod* (2018) | https://arxiv.org/abs/1802.05799 | [ml.hardware](../../primer/ml/hardware.py) |
| `reinforce` | Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (1992) | https://link.springer.com/article/10.1007/BF00992696 | [ml.reinforcement](../../primer/ml/reinforcement.py) |
| `ppo` | Schulman et al., *Proximal Policy Optimization Algorithms* (2017) | https://arxiv.org/abs/1707.06347 | [ml.reinforcement](../../primer/ml/reinforcement.py) |
| `deepseekmath-grpo` | Shao et al., *DeepSeekMath* (GRPO) (2024) | https://arxiv.org/abs/2402.03300 | [ml.reinforcement](../../primer/ml/reinforcement.py), [ml.reasoning](../../primer/ml/reasoning.py) |
| `deepseek-r1` | DeepSeek-AI, *DeepSeek-R1* (2025) | https://arxiv.org/abs/2501.12948 | [ml.reinforcement](../../primer/ml/reinforcement.py), [ml.reasoning](../../primer/ml/reasoning.py) |
| `vit` | Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT) (2020) | https://arxiv.org/abs/2010.11929 | [ml.generative.multimodal](../../primer/ml/generative/multimodal.py) |
| `flamingo` | Alayrac et al., *Flamingo* (2022) | https://arxiv.org/abs/2204.14198 | [ml.generative.multimodal](../../primer/ml/generative/multimodal.py) |
| `llava` | Liu et al., *Visual Instruction Tuning* (LLaVA) (2023) | https://arxiv.org/abs/2304.08485 | [ml.generative.multimodal](../../primer/ml/generative/multimodal.py) |
| `whisper` | Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper) (2022) | https://arxiv.org/abs/2212.04356 | [ml.generative.multimodal](../../primer/ml/generative/multimodal.py) |
| `vq-vae` | van den Oord et al., *Neural Discrete Representation Learning* (VQ-VAE) (2017) | https://arxiv.org/abs/1711.00937 | [ml.generative.multimodal](../../primer/ml/generative/multimodal.py) |
| `soundstream` | Zeghidour et al., *SoundStream* (2021) | https://arxiv.org/abs/2107.03312 | [ml.generative.multimodal](../../primer/ml/generative/multimodal.py) |
| `dall-e` | Ramesh et al., *Zero-Shot Text-to-Image Generation* (DALL·E) (2021) | https://arxiv.org/abs/2102.12092 | [ml.generative.multimodal](../../primer/ml/generative/multimodal.py) |
| `vivit` | Arnab et al., *ViViT* (2021) | https://arxiv.org/abs/2103.15691 | [ml.generative.multimodal](../../primer/ml/generative/multimodal.py) |
| `chain-of-thought-prompting` | Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (2022) | https://arxiv.org/abs/2201.11903 | [ml.reasoning](../../primer/ml/reasoning.py) |
| `self-consistency` | Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models* (2022) | https://arxiv.org/abs/2203.11171 | [ml.reasoning](../../primer/ml/reasoning.py) |
| `training-verifiers` | Cobbe et al., *Training Verifiers to Solve Math Word Problems* (2021) | https://arxiv.org/abs/2110.14168 | [ml.reasoning](../../primer/ml/reasoning.py) |
| `lets-verify-step-by-step` | Lightman et al., *Let's Verify Step by Step* (2023) | https://arxiv.org/abs/2305.20050 | [ml.reasoning](../../primer/ml/reasoning.py) |
| `scaling-test-time-compute` | Snell et al., *Scaling LLM Test-Time Compute Optimally* (2024) | https://arxiv.org/abs/2408.03314 | [ml.reasoning](../../primer/ml/reasoning.py) |
| `mmlu` | Hendrycks et al., *Measuring Massive Multitask Language Understanding* (2020) | https://arxiv.org/abs/2009.03300 | [ml.benchmarks](../../primer/ml/benchmarks.py) |
| `humaneval-pass-at-k` | Chen et al., *Evaluating Large Language Models Trained on Code* (2021) | https://arxiv.org/abs/2107.03374 | [ml.benchmarks](../../primer/ml/benchmarks.py), [agents.coding_agents](../../primer/agents/coding_agents.py) |
| `imagenet-v2` | Recht et al., *Do ImageNet Classifiers Generalize to ImageNet?* (2019) | https://arxiv.org/abs/1902.10811 | [ml.benchmarks](../../primer/ml/benchmarks.py) |
| `chatbot-arena` | Chiang et al., *Chatbot Arena* (2024) | https://arxiv.org/abs/2403.04132 | [ml.benchmarks](../../primer/ml/benchmarks.py) |
| `error-bars-for-evals` | Miller, *Adding Error Bars to Evals* (2024) | https://arxiv.org/abs/2411.00640 | [ml.benchmarks](../../primer/ml/benchmarks.py) |
| `sparse-transformers` | Child et al., *Generating Long Sequences with Sparse Transformers* (2019) | https://arxiv.org/abs/1904.10509 | [ml.efficient_architectures](../../primer/ml/efficient_architectures.py) |
| `longformer` | Beltagy et al., *Longformer* (2020) | https://arxiv.org/abs/2004.05150 | [ml.efficient_architectures](../../primer/ml/efficient_architectures.py) |
| `big-bird` | Zaheer et al., *Big Bird* (2020) | https://arxiv.org/abs/2007.14062 | [ml.efficient_architectures](../../primer/ml/efficient_architectures.py) |
| `transformers-are-rnns` | Katharopoulos et al., *Transformers are RNNs* (2020) | https://arxiv.org/abs/2006.16236 | [ml.efficient_architectures](../../primer/ml/efficient_architectures.py) |
| `s4` | Gu, Goel & Ré, *Efficiently Modeling Long Sequences with Structured State Spaces* (S4) (2021) | https://arxiv.org/abs/2111.00396 | [ml.efficient_architectures](../../primer/ml/efficient_architectures.py) |
| `mamba` | Gu & Dao, *Mamba* (2023) | https://arxiv.org/abs/2312.00752 | [ml.efficient_architectures](../../primer/ml/efficient_architectures.py) |
| `mamba-2` | Dao & Gu, *Transformers are SSMs* (Mamba-2) (2024) | https://arxiv.org/abs/2405.21060 | [ml.efficient_architectures](../../primer/ml/efficient_architectures.py) |
| `mistral-7b` | Jiang et al., *Mistral 7B* (2023) | https://arxiv.org/abs/2310.06825 | [ml.efficient_architectures](../../primer/ml/efficient_architectures.py) |
| `attention-sinks` | Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (2023) | https://arxiv.org/abs/2309.17453 | [ml.efficient_architectures](../../primer/ml/efficient_architectures.py) |
| `jamba` | Lieber et al., *Jamba* (2024) | https://arxiv.org/abs/2403.19887 | [ml.efficient_architectures](../../primer/ml/efficient_architectures.py) |
| `multi-query-attention` | Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (2019) | https://arxiv.org/abs/1911.02150 | [ml.efficient_architectures](../../primer/ml/efficient_architectures.py) |
| `deepseek-v2` | DeepSeek-AI, *DeepSeek-V2* (2024) | https://arxiv.org/abs/2405.04434 | [ml.efficient_architectures](../../primer/ml/efficient_architectures.py) |
| `kivi` | Liu et al., *KIVI* (2024) | https://arxiv.org/abs/2402.02750 | [ml.efficient_architectures](../../primer/ml/efficient_architectures.py) |
| `linear-probes` | Alain & Bengio, *Understanding Intermediate Layers Using Linear Classifier Probes* (2016) | https://arxiv.org/abs/1610.01644 | [ml.interpretability](../../primer/ml/interpretability.py) |
| `probe-control-tasks` | Hewitt & Liang, *Designing and Interpreting Probes with Control Tasks* (2019) | https://arxiv.org/abs/1909.03368 | [ml.interpretability](../../primer/ml/interpretability.py) |
| `tuned-lens` | Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens* (2023) | https://arxiv.org/abs/2303.08112 | [ml.interpretability](../../primer/ml/interpretability.py) |
| `rome-causal-tracing` | Meng et al., *Locating and Editing Factual Associations in GPT* (2022) | https://arxiv.org/abs/2202.05262 | [ml.interpretability](../../primer/ml/interpretability.py) |
| `ioi-circuit` | Wang et al., *Interpretability in the Wild* (2022) | https://arxiv.org/abs/2211.00593 | [ml.interpretability](../../primer/ml/interpretability.py) |
| `toy-models-of-superposition` | Elhage et al., *Toy Models of Superposition* (2022) | https://arxiv.org/abs/2209.10652 | [ml.interpretability](../../primer/ml/interpretability.py) |
| `towards-monosemanticity` | Bricken et al., *Towards Monosemanticity* (2023) | https://transformer-circuits.pub/2023/monosemantic-features/index.html | [ml.interpretability](../../primer/ml/interpretability.py) |
| `scaling-monosemanticity` | Templeton et al., *Scaling Monosemanticity* (2024) | https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html | [ml.interpretability](../../primer/ml/interpretability.py) |
| `cart` | Breiman, Friedman, Olshen & Stone, *Classification and Regression Trees* (1984) | https://doi.org/10.1201/9781315139470 | [ml.classical](../../primer/ml/classical.py) |
| `id3` | Quinlan, *Induction of Decision Trees* (1986) | https://doi.org/10.1007/BF00116251 | [ml.classical](../../primer/ml/classical.py) |
| `bagging-predictors` | Breiman, *Bagging Predictors* (1996) | https://doi.org/10.1007/BF00058655 | [ml.classical](../../primer/ml/classical.py) |
| `random-forests` | Breiman, *Random Forests* (2001) | https://doi.org/10.1023/A:1010933404324 | [ml.classical](../../primer/ml/classical.py) |
| `gradient-boosting-machine` | Friedman, *Greedy Function Approximation: A Gradient Boosting Machine* (2001) | https://doi.org/10.1214/aos/1013203451 | [ml.classical](../../primer/ml/classical.py) |
| `xgboost` | Chen & Guestrin, *XGBoost* (2016) | https://arxiv.org/abs/1603.02754 | [ml.classical](../../primer/ml/classical.py) |
| `trees-beat-deep-learning-on-tables` | Grinsztajn, Oyallon & Varoquaux, *Why do tree-based models still outperform deep learning on tabular data?* (2022) | https://arxiv.org/abs/2207.08815 | [ml.classical](../../primer/ml/classical.py) |
| `task-arithmetic` | Ilharco et al., *Editing Models with Task Arithmetic* (2022) | https://arxiv.org/abs/2212.04089 | [ml.fine_tuning](../../primer/ml/fine_tuning.py) |
| `model-soups` | Wortsman et al., *Model Soups* (2022) | https://arxiv.org/abs/2203.05482 | [ml.fine_tuning](../../primer/ml/fine_tuning.py) |
| `ties-merging` | Yadav et al., *TIES-Merging* (2023) | https://arxiv.org/abs/2306.01708 | [ml.fine_tuning](../../primer/ml/fine_tuning.py) |
| `elastic-weight-consolidation` | Kirkpatrick et al., *Overcoming Catastrophic Forgetting in Neural Networks* (2017) | https://arxiv.org/abs/1612.00796 | [ml.fine_tuning](../../primer/ml/fine_tuning.py) |
| `lora-learns-less-forgets-less` | Biderman et al., *LoRA Learns Less and Forgets Less* (2024) | https://arxiv.org/abs/2405.09673 | [ml.fine_tuning](../../primer/ml/fine_tuning.py) |
| `lima` | Zhou et al., *LIMA: Less Is More for Alignment* (2023) | https://arxiv.org/abs/2305.11206 | [ml.fine_tuning](../../primer/ml/fine_tuning.py) |
| `deduplicating-training-data` | Lee et al., *Deduplicating Training Data Makes Language Models Better* (2021) | https://arxiv.org/abs/2107.06499 | [ml.fine_tuning](../../primer/ml/fine_tuning.py), [ml.pretraining](../../primer/ml/pretraining.py) |
| `diffusion-nonequilibrium-thermodynamics` | Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* (2015) | https://arxiv.org/abs/1503.03585 | [ml.generative.diffusion](../../primer/ml/generative/diffusion.py) |
| `score-matching-ncsn` | Song & Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution* (2019) | https://arxiv.org/abs/1907.05600 | [ml.generative.diffusion](../../primer/ml/generative/diffusion.py) |
| `ddpm` | Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models* (2020) | https://arxiv.org/abs/2006.11239 | [ml.generative.diffusion](../../primer/ml/generative/diffusion.py) |
| `ddim` | Song, Meng & Ermon, *Denoising Diffusion Implicit Models* (2020) | https://arxiv.org/abs/2010.02502 | [ml.generative.diffusion](../../primer/ml/generative/diffusion.py) |
| `score-sde` | Song et al., *Score-Based Generative Modeling through SDEs* (2020) | https://arxiv.org/abs/2011.13456 | [ml.generative.diffusion](../../primer/ml/generative/diffusion.py) |
| `classifier-free-guidance` | Ho & Salimans, *Classifier-Free Diffusion Guidance* (2022) | https://arxiv.org/abs/2207.12598 | [ml.generative.diffusion](../../primer/ml/generative/diffusion.py) |
| `latent-diffusion` | Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2021) | https://arxiv.org/abs/2112.10752 | [ml.generative.diffusion](../../primer/ml/generative/diffusion.py) |
| `diffusion-transformers` | Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT) (2022) | https://arxiv.org/abs/2212.09748 | [ml.generative.diffusion](../../primer/ml/generative/diffusion.py) |
| `flow-matching` | Lipman et al., *Flow Matching for Generative Modeling* (2022) | https://arxiv.org/abs/2210.02747 | [ml.generative.diffusion](../../primer/ml/generative/diffusion.py) |
| `rectified-flow` | Liu, Gong & Liu, *Flow Straight and Fast* (rectified flow) (2022) | https://arxiv.org/abs/2209.03003 | [ml.generative.diffusion](../../primer/ml/generative/diffusion.py) |
| `rectified-flow-transformers` | Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (2024) | https://arxiv.org/abs/2403.03206 | [ml.generative.diffusion](../../primer/ml/generative/diffusion.py) |
| `gopher` | Rae et al., *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* (2021) | https://arxiv.org/abs/2112.11446 | [ml.pretraining](../../primer/ml/pretraining.py) |
| `fineweb` | Penedo et al., *The FineWeb Datasets* (2024) | https://arxiv.org/abs/2406.17557 | [ml.pretraining](../../primer/ml/pretraining.py) |
| `model-collapse` | Shumailov et al., *The Curse of Recursion* (2023) | https://arxiv.org/abs/2305.17493 | [ml.pretraining](../../primer/ml/pretraining.py) |
| `zero` | Rajbhandari et al., *ZeRO* (2019) | https://arxiv.org/abs/1910.02054 | [ml.pretraining](../../primer/ml/pretraining.py) |
| `gpipe` | Huang et al., *GPipe* (2018) | https://arxiv.org/abs/1811.06965 | [ml.pretraining](../../primer/ml/pretraining.py) |
| `activation-recomputation` | Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* (2022) | https://arxiv.org/abs/2205.05198 | [ml.pretraining](../../primer/ml/pretraining.py) |
| `swe-bench` | Jimenez et al., *SWE-bench* (2023) | https://arxiv.org/abs/2310.06770 | [agents.coding_agents](../../primer/agents/coding_agents.py) |
| `swe-agent` | Yang et al., *SWE-agent* (2024) | https://arxiv.org/abs/2405.15793 | [agents.coding_agents](../../primer/agents/coding_agents.py) |
| `osworld` | Xie et al., *OSWorld* (2024) | https://arxiv.org/abs/2404.07972 | [agents.coding_agents](../../primer/agents/coding_agents.py) |
| `indirect-prompt-injection` | Greshake et al., *Not What You've Signed Up For: Indirect Prompt Injection* (2023) | https://arxiv.org/abs/2302.12173 | [agents.coding_agents](../../primer/agents/coding_agents.py) |
| `generative-adversarial-nets` | Goodfellow et al., *Generative Adversarial Nets* (2014) | https://arxiv.org/abs/1406.2661 | [ml.generative.gans](../../primer/ml/generative/gans.py) |
| `unrolled-gans` | Metz et al., *Unrolled Generative Adversarial Networks* (2016) | https://arxiv.org/abs/1611.02163 | [ml.generative.gans](../../primer/ml/generative/gans.py) |
| `ttur` | Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (2017) | https://arxiv.org/abs/1706.08500 | [ml.generative.gans](../../primer/ml/generative/gans.py) |
| `wasserstein-gan` | Arjovsky, Chintala & Bottou, *Wasserstein GAN* (2017) | https://arxiv.org/abs/1701.07875 | [ml.generative.gans](../../primer/ml/generative/gans.py) |
| `wgan-gp` | Gulrajani et al., *Improved Training of Wasserstein GANs* (2017) | https://arxiv.org/abs/1704.00028 | [ml.generative.gans](../../primer/ml/generative/gans.py) |
| `gan-convergence-r1` | Mescheder, Geiger & Nowozin, *Which Training Methods for GANs do actually Converge?* (2018) | https://arxiv.org/abs/1801.04406 | [ml.generative.gans](../../primer/ml/generative/gans.py) |
| `spectral-normalization` | Miyato et al., *Spectral Normalization for Generative Adversarial Networks* (2018) | https://arxiv.org/abs/1802.05957 | [ml.generative.gans](../../primer/ml/generative/gans.py) |
| `efficient-guided-generation` | Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023) | https://arxiv.org/abs/2307.09702 | [ml.structured_output](../../primer/ml/structured_output.py) |
| `grammar-constrained-decoding` | Geng et al., *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning* (2023) | https://arxiv.org/abs/2305.13971 | [ml.structured_output](../../primer/ml/structured_output.py) |
| `picard` | Scholak et al., *PICARD* (2021) | https://arxiv.org/abs/2109.05093 | [ml.structured_output](../../primer/ml/structured_output.py) |
| `xgrammar` | Dong et al., *XGrammar* (2024) | https://arxiv.org/abs/2411.15100 | [ml.structured_output](../../primer/ml/structured_output.py) |
| `grammar-aligned-decoding` | Park et al., *Grammar-Aligned Decoding* (2024) | https://arxiv.org/abs/2405.21047 | [ml.structured_output](../../primer/ml/structured_output.py) |
| `let-me-speak-freely` | Tam et al., *Let Me Speak Freely?* (2024) | https://arxiv.org/abs/2408.02442 | [ml.structured_output](../../primer/ml/structured_output.py) |
| `reducing-dimensionality-neural-networks` | Hinton & Salakhutdinov, *Reducing the Dimensionality of Data with Neural Networks* (2006) | https://doi.org/10.1126/science.1127647 | [ml.generative.autoencoders](../../primer/ml/generative/autoencoders.py) |
| `auto-encoding-variational-bayes` | Kingma & Welling, *Auto-Encoding Variational Bayes* (2013) | https://arxiv.org/abs/1312.6114 | [ml.generative.autoencoders](../../primer/ml/generative/autoencoders.py) |
| `stochastic-backpropagation` | Rezende, Mohamed & Wierstra, *Stochastic Backpropagation and Approximate Inference in Deep Generative Models* (2014) | https://arxiv.org/abs/1401.4082 | [ml.generative.autoencoders](../../primer/ml/generative/autoencoders.py) |
| `understanding-beta-vae` | Burgess et al., *Understanding Disentangling in β-VAE* (2018) | https://arxiv.org/abs/1804.03599 | [ml.generative.autoencoders](../../primer/ml/generative/autoencoders.py) |

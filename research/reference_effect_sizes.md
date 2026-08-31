# Reference effect sizes — how big a real gain is on this task

Static, checked-in context for the per-run negative-result artifact
(`<run_dir>/falsified.md`, rendered by `src/agent/falsified.py`). Nothing here is
computed at run time and nothing here is a result of ours: these are published
numbers, transcribed with their sources, so that a delta measured by this agent
can be read against what the literature calls an improvement.

Transcribed from the project's external-benchmark survey; every number below is
attributable to a primary source in the **Sources** section at the end of this file, which
is the citation a reader should follow.

## The bar the field sets itself

> "Notably, an improvement of more than **0.1 % in AUC is considered significant** for the
> CTR prediction task (Guo et al., 2017)."
> — *Automated Information Flow Selection for Multi-scenario Multi-task Recommendation*
> (AutoIFS), <https://arxiv.org/abs/2512.13396>

> "in large-scale recommender systems, AUC values are often already very high … making further
> improvements extremely challenging. Therefore, even **small absolute gains in AUC (e.g.,
> +0.1 %) are regarded as practically significant**."
> — *VQL: An End-to-End Context-Aware Vector Quantization Attention for Ultra-Long User
> Behavior Modeling*, <https://arxiv.org/abs/2508.17125>

**+0.001 absolute** is therefore the published significance threshold. This harness's own
measured seed sigma is **0.00091** (`policy.MEASURED_SEED_SIGMA`) — the same order of
magnitude. A publishable gain here is about the size of the measurement noise.

## Effect-size table

| Change | Dataset / setting | Reported effect | Source |
|---|---|---|---|
| Wide&Deep over deep BaseModel | Alibaba production | +0.0007 GAUC | DIN, KDD '18 |
| DeepFM over deep BaseModel | Alibaba production | +0.0023 GAUC | DIN, KDD '18 |
| DIN over BaseModel (target-aware history attention) | Alibaba production | +0.0059 GAUC → +10 % online CTR | DIN, KDD '18 |
| DIEN over DIN | KuaiRand-1K | +0.0005 AUC | VQL, Table 3 |
| DIN → VQL (whole ultra-long-history ladder) | KuaiRand-1K | +0.038 AUC | VQL, Table 3 |
| BCE → BCE + pairwise ranking loss (headline) | industrial ads, DCNv2 | +0.00077 AUC | Lin et al., KDD '24 |
| BCE → BCE + listwise ranking loss | industrial ads, DCNv2 | +0.0003 AUC | Lin et al., KDD '24 |
| SMES-L (510 M params) over Rankmixer | KuaiRand-1K, effective-view GAUC | +0.0029 GAUC | SMES, 2026 |
| AutoIFS over HiNet (best baseline of 13) | **KuaiRand-Pure, long-view** | +0.0024 AUC | AutoIFS, 2025 |
| Field's stated significance bar | — | **+0.001** | Guo et al., IJCAI '17 |

Spreads, for scale — the distance from the *worst* to the *best* model in each published
table: 0.0113 GAUC (DIN, Alibaba production), 0.0135 GAUC (SMES, KuaiRand-1K), 0.009 AUC
(AutoIFS, KuaiRand-Pure). Entire architecture zoos live inside ~0.01.

## Two caveats that must travel with these numbers

1. **Global AUC ≫ within-user GAUC on the same data.** Global AUC is rewarded for separating
   heavy engagers from light ones — a between-user effect this task deliberately removes. The
   0.70–0.80 AutoIFS numbers are not comparable to our 0.6674 validation GAUC in level; they
   establish the *learnability* of the label and the *size of the spread*, nothing more.
2. **The ranking-loss direction is calibrated by positive sparsity.** Lin et al. attribute the
   BCE→pairwise gain to gradient vanishing on negatives, which "the sparser the positive
   samples, the greater the performance improvement". KuaiRand-Pure `long_view` is **33.7 %
   positive** — dense. A within-noise result from `bpr` / `group_softmax` here is the
   mechanism's prediction, not a defect in the harness.

## Local anchors

From `kuairand-starter-kit/baseline_scores.json`: official FM validation GAUC 0.6674 /
nDCG@5 0.5357 / primary 0.6016; test 0.6610 / 0.5282 / 0.5946; seed std 0.0008. Oracle
ceiling on test: GAUC 1.0, nDCG@5 0.7289, primary 0.8645.

## Sources

1. Gao, Li, Zhang, Chen, Li, Lei, Jiang, He. *KuaiRand: An Unbiased Sequential Recommendation
   Dataset with Randomly Exposed Videos.* CIKM '22. <https://arxiv.org/abs/2208.08696> —
   publishes **no** baselines or experiments section.
2. *Automated Information Flow Selection for Multi-scenario Multi-task Recommendation.*
   <https://arxiv.org/abs/2512.13396> — KuaiRand-Pure, click + long-view, AUC tables.
3. Zhang, Dong, Wang, Chen, Jia, … Zhou, Li, Gai (Kuaishou). *SMES: Towards Scalable
   Multi-Task Recommendation via Expert Sparsity.* 2026. <https://arxiv.org/abs/2602.09386> —
   KuaiRand-1K AUC and GAUC tables.
4. *VQL: An End-to-End Context-Aware Vector Quantization Attention for Ultra-Long User
   Behavior Modeling.* 2025. <https://arxiv.org/abs/2508.17125> — KuaiRand-1K DIN→VQL ladder.
5. Zhou, Zhu, Song, Fan, Zhu, Ma, Yan, Jin, Li, Gai. *Deep Interest Network for Click-Through
   Rate Prediction.* KDD '18. <https://arxiv.org/abs/1706.06978> — popularised GAUC; Alibaba
   production table.
6. Zhou, Mou, Fan, Pi, Bian, Zhou, Zhu, Gai. *Deep Interest Evolution Network for
   Click-Through Rate Prediction.* AAAI '19. <https://arxiv.org/abs/1809.03672> — 5-seed
   standard deviations of the same order as ours (0.0003–0.0024).
7. Lin, et al. (Tencent). *Understanding the Ranking Loss for Recommendation with Sparse User
   Feedback.* KDD '24. <https://arxiv.org/abs/2403.14144> — the +0.00077 headline and the
   sparsity mechanism.

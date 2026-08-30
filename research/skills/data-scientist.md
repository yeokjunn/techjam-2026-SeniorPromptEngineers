# Role Skill: Data Scientist (Recommender Systems & Within-User Ranking)

Expert guidelines for hypothesis formulation, feature engineering, and ranking objectives in KuaiRand-Pure:

1. Within-User Ranking Dynamics:
   - Evaluation metrics (GAUC, nDCG@5) measure ranking strictly among impressions for the SAME user.
   - Pure user-side static features contribute a constant bias across all items for that user and cancel out during ranking.
   - User signal ONLY adds value when crossed with item features (e.g. user x author affinity, user historical engagement with video tags/categories).

2. Leakage-Safe Temporal & Sequential Features:
   - Short-video consumption is driven by recent session history (e.g. last 5-10 videos watched, recent category preference).
   - Any historical aggregates (author watch rate, category affinity, dwell time priors) MUST only use interactions strictly prior to the current row's timestamp. Never aggregate future interactions.

3. Objective Alignment:
   - Pointwise binary cross-entropy treats impressions independently, misaligned with intra-user ranking.
   - Pairwise ranking (BPR) directly models relative preference between positive and negative impressions within the same user session.
   - Listwise/group softmax models relative competition across items displayed together.

4. Multi-Task & Auxiliary Signals:
   - The primary target long_view is sparse. Jointly modeling auxiliary signals (is_click, is_like, play_time) provides representation learning.
   - Auxiliary loss terms must be balanced (e.g. weight 0.05 - 0.3) so dense heads do not overpower the primary ranking objective.

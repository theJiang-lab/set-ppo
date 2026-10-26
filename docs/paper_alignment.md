# Manuscript-to-code correspondence

This map is based on the supplied current manuscript and its retained D.4 configuration. It records implementation coverage rather than claiming original numerical reproduction.

| Paper equation/section | Code | Correspondence |
| --- | --- | --- |
| Eq. (1): coverage and mastery | `belief.sample_knowledge`, `env.reset` | Binary availability, zero mastery when unavailable, reported uniform/Beta priors |
| Eqs. (2)–(3): DATA support and mixture | `semantics.encode` | Positive thresholded cosine weights, normalized on nonempty support |
| Offline eligibility | `data.prepare_embeddings` | Fixed normalized-error threshold; coverage over all news, TCR over eligible tasks |
| Eq. (4): receiver reconstruction | `semantics.decode` | Nonnegative least squares with coefficient sum at most one; checked solver convergence |
| Eq. (5): task quality and completion | `semantics.quality`, `env.step` | Completion evaluated before the next learning update; KFB never completes DATA |
| Eq. (6): residual request | `env.step` | Frozen Gaussian projection of the reconstruction residual |
| Eq. (7): feedback choice | `belief.select` | Missing-atom probability times projected cosine; expected weighted-gap fallback |
| Eq. (8): knowledge update | `semantics.learn` | Exponential bounded mastery gain; only KFB can add atoms |
| Eq. (9): observation budget | `env.observer_view`, `env.step` | One selected DATA/KFB content costs one observation unit |
| Eqs. (10)–(11): growth and discoveries | `env.metrics` | Cumulative mastery growth and new-atom discovery counted separately |
| Eq. (12): TCR | `env.metrics` | Completed tasks divided by the fixed initial required-task count |
| Eq. (13): constrained adversarial objective | `runner.train` | Separate legitimate and Eve PPO updates with service multiplier |
| Eq. (14): hierarchical policy | `networks.SetActor` | Masked type distribution times conditional item distribution; joint log probability |
| Appendix Eq. (A.1): feasibility bounds | `metrics.feasibility_bounds` | Mean-completion bounds use ceil(N × target), separate from measured feasibility |
| Appendix Eqs. (A.2)–(A.3): rewards and multiplier | `runner.rollout`, `runner.train` | Undiscounted growth rewards; complete-episode mean TCR updates a projected multiplier |
| Candidate freezing for PPO | `ppo.snapshot`, `ppo.PPO` | Immutable features, masks, candidate order and history prefix retained per rollout |
| D.1: semantic action order | `env.step` | Reconstruct/status first, then knowledge transition; residual reporting is not a primary action |
| D.3: access boundary | `env.observer_view`, `_event` | E0 excludes metadata; E1 only adds type/reader; unselected content/status excluded |
| D.4: parameters | `configs/paper.json` | Reported numeric fields checked against `paper_contract.json` |
| Service matching | `metrics.service_matched` | At least 85% TCR and within one percentage point of 88.7% |
| Automated QA | `qa.score_tasks` | Three questions, pass at two, at most one repair/retest, final valid test only |

The sender's particle uncertainty filter, network width, missing thresholds and baseline adapters are explicitly distinguished from reported parameters in [implementation choices](implementation_choices.md). The core decoder is not replaced by the filter's approximate decoder. Training-only true states are consumed by separate critics; the runtime policy has only public history and estimated knowledge.

The 47 supplied result rows and 32 QA task aggregates are bundled with source hashes. `audit-reported` verifies these values and arithmetic without training. In particular, sTRPO-C TCR 88.0, growth 2.01 and discovered atoms 2.06 are distinct fields; no correction is propagated into an unrelated metric.


# Parameters not recoverable from the supplied manuscript

The values in `Config` below the explicit comment are runnable defaults. They are not inferred experimental measurements, and changing them can change the operating point. The original five numeric seed IDs are also unknown.

| Choice | Implementation | Status |
| --- | --- | --- |
| Source/receiver cosine thresholds | 0.25 / 0.20 | Not reported |
| Reconstruction completion threshold | 0.80 | Not reported; distinct from the reported 0.85 service target |
| Known-atom threshold | 0.60 | Not reported |
| Reader/Eve learning rates in Eq. (8) | 0.80 / 0.30 | Not reported |
| New-atom initial mastery | 0.10, followed by Eq. (8) | Not reported |
| DATA/KFB learning signal | Mixture coefficients / unit signal for repaired atom | Not reported |
| Belief approximation | 16-particle filter, public success/KREQ likelihood | Architecture not fixed by manuscript |
| Belief likelihood noise and residual confidence | 0.15 / 0.05 | Not reported |
| Belief-internal projected-gradient decoder | 40 iterations | Approximation only for filtering; physical receiver uses checked SLSQP |
| Recurrent actor | 64-dimensional GRU history encoder | Width and encoder architecture not reported |
| Multiplier step and cap | 0.05 / 100 | Not reported |
| Entropy/value weights and gradient clipping | 0.01 / 0.50 / 0.50 | Not reported |
| Validation frequency/count | Every 10 iterations / 20 episodes | Not reported |
| Reference observer | Fixed budgeted uniform schedule | Policy identity not supplied; E0 specifies access, not a unique policy |
| Overflow repair order | Increasing task ID | Not reported; queued tasks remain unfinished |
| KFB/completed aggregation | Sum of KFB messages / sum of completed tasks | Aggregation detail not reported |
| Localization diagnostic | Selected repair atom versus largest true weighted gap after DATA learning | Ground-truth counting convention not reported |
| Article encoding per reader | Same frozen article vector; receiver differences enter reconstruction | Personalized input construction not specified |

The particle filter is a history-based approximation to the server belief, not access to actual receiver state. Its approximate internal decoder is separate from the constrained receiver decoder. No hidden state is fed to the legitimate actor, except the explicitly named Oracle selector which is a diagnostic baseline.

The no-set variant uses an ordered padded pooling layer; the no-hierarchy variant selects one flat categorical action; the no-belief variant holds the prior estimator fixed; the no-adversarial variant freezes the budgeted reference observer. These are documented runnable removals, not claims of recovering the author's unspecified original architecture variants.

E2 provides residual history to future observer decisions; it does not invent a direct KREQ-to-mastery learning law, since only DATA/KFB learning is defined in the supplied model. Its additional access is recorded as a separate condition.

Data preparation, training and evaluation commands write resolved settings and run provenance. The supplied paper aggregates remain unchanged regardless of code defaults. A full numerical replication requires the original splits, missing choices, reference observer, baseline checkpoints and training artifacts.


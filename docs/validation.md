# Verification scope

Date: 2026-10-26.

- Sixteen local invariant unit tests passed (0.41 s reported by pytest): constrained reconstruction, monotone knowledge improvement, exponential mastery updates, DATA/KFB discovery distinction, simplex projection, E0/E1 metadata boundaries, absence of observer-private state from legitimate observations, observer passivity, KFB task conservation, bounded repair queues, excluded E0 content/status, set permutation invariance, frozen behavior snapshots, and episode-boundary GAE.
- At the author's subsequent instruction, no training or evaluation experiment was executed. No MIND download, model download, optimizer training run, test-episode sweep or numerical replication is reported.
- The release includes a syntax check and a static audit of reported configuration values, 47 aggregate result rows and 32 QA tasks. The audit recomputes service matching and the RCPO/sTRPO reductions from the supplied values.
- Full original numerical replication cannot be established from aggregates: original splits, weights, per-seed logs, several thresholds, and original baseline configurations are absent. Such choices are documented instead of silently attributed to the paper.

The prepared-data and training commands are provided for later use. Their end-to-end runtime has not been certified by an experiment in this release. An optional manual CI example is provided; it runs invariant tests only.


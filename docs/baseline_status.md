# Baseline coverage

| Paper component | Code support | Original result recovery |
| --- | --- | --- |
| Full Set-PPO | Semantic environment, hierarchical set actor, recurrent history, adversarial PPO, dual update | Original weights/logs absent; no claimed numeric replication |
| Component removals | Explicit runnable variants in `baselines.py` | Architecture details not fully reported |
| Residual / posterior / shuffled / random / Oracle selectors | Selectors and joint service/exposure metrics | Original cases and metric details absent |
| Random / knowledge-aware greedy / KFB-first | Public-information heuristics in common evaluator | Heuristic tie-breaking/details are implementation choices |
| No-KFB | Feedback disabled, fixed task denominator | Original cases absent |
| CPO / RCPO | External `module:factory` policy interface; reported aggregates bundled | Original implementations/configurations not supplied |
| sTRPO-A/B/C | Same external interface; all three corrected result rows preserved | A/B/C configuration differences and training parameters unknown |
| Reported policy mixture | Reported aggregate retained | Mixture weights/components not supplied |
| E0/E1 adaptation | Freeze legitimate actor, update observer on train IDs, select on validation IDs | Original observer weights absent |
| E2 sensitivity | Separate residual-visible access class | No equal-access-budget claim |
| Dictionary scan | Reprepare train-only PCA/K-means for each M; report pool coverage | Original news partitions absent |
| Automated QA | Final-test count scorer and supplied four-table aggregates | Original questions, responses and model IDs not supplied |

An external baseline factory receives the configuration and returns an object with `select_action(observation) -> index`. The evaluator passes only the same public legitimate observation used by Set-PPO. It does not pass the privileged critic state. For example:

```bash
setppo evaluate --checkpoint runs/full/seed_0/best.pt --dataset data/prepared/mind.npz --external-policy my_baseline:load_policy --output runs/baseline/test
```

No substitute policy is presented as the original sTRPO, CPO or RCPO. Their identities and corrected aggregate comparisons remain in the reported-result data, independent of this interface.


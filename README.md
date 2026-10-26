# Set-PPO

Code for **Task-Constrained Secure Alignment of Heterogeneous Semantic Knowledge for Web News Delivery**.

[中文说明](README.zh-CN.md) · [Paper/code correspondence](docs/paper_alignment.md) · [Unreported choices](docs/implementation_choices.md)

This is a manuscript-derived reimplementation of the semantic environment, residual repair, hierarchical adversarial PPO, and evaluation pipeline. The original training code, model weights, dataset partitions, per-seed logs and complete baseline configurations were not supplied. Values in `data/reported/` are the author's reported aggregates, **not outputs produced by this implementation**. Training and evaluation experiments have not been run for this release.

## Install

Python 3.10+ and PyTorch 2.2+:

```bash
python -m pip install -e ".[text,test]"
```

Use a PyTorch installation appropriate for your hardware. Core simulator operations run on CPU; the neural networks can use `--device cuda`.

## Prepare MIND articles

Obtain MIND-small `news.tsv` files from the [official MIND website](https://msnews.github.io/) under its terms. Raw news, embeddings and checkpoints are not redistributed here.

```bash
setppo prepare --news data/raw/train/news.tsv data/raw/dev/news.tsv --config configs/paper.json --output data/prepared/mind.npz --seed 0
```

The encoder is frozen [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), using masked mean pooling and normalization. Articles are deduplicated by ID and split 80:10:10. PCA and K-means fit **training IDs only**. The prepared artifact stores training-fit provenance and dictionary-specific coverage. Policy training seeds do not refit a different data partition.

## Train and evaluate

These commands are provided for future use; they were **not executed** to produce the supplied paper results.

```bash
setppo train --config configs/paper.json --dataset data/prepared/mind.npz --output runs/full/seed_0 --seed 0
setppo evaluate --checkpoint runs/full/seed_0/best.pt --dataset data/prepared/mind.npz --output runs/full/seed_0/test --seed 0 --episodes 500
```

`best.pt` is saved only for a validation checkpoint meeting the mean-service target, with the lowest validation observer growth. `last.pt` is a final training checkpoint and is not presented as validation-selected. Test IDs are not used to select checkpoints or tune TCR.

The reference configuration contains four readers, 80 required news-reader tasks, 160 semantic actions, 96 atoms, 16-dimensional prototypes, a 10-dimensional residual, three registered repair items, service target 0.85, and observer budget 20. PPO uses 2,000 iterations, 64 episodes per iteration, learning rate 0.0003, minibatches of 256, four epochs, gamma 0.99, GAE 0.95 and clipping 0.2. See `configs/paper.json` and the [parameter provenance](docs/paper_contract.json). Seed IDs 0–4 are implementation choices; only the number of seeds was reported.

To launch the configured five-seed schedule and component removals:

```bash
python scripts/run_matrix.py --config configs/paper.json --dataset data/prepared/mind.npz --variants full no-adversarial no-residual no-belief no-hierarchy no-set --output runs/matrix
```

All methods use paired initial task/knowledge cases within a seed. `episodes.csv` contains case fingerprints, TCR, feasibility, growth, discovered atoms and repair counts. New-run files are labeled separately from published aggregates. Change the dictionary size and reprepare the artifact for a coverage scan; its eligible task pool can change.

## Observer adaptation and selectors

```bash
setppo train --config configs/paper.json --dataset data/prepared/mind.npz --output runs/adapted_e1 --adapt-observer runs/full/seed_0/best.pt --access E1
setppo evaluate --checkpoint runs/adapted_e1/best_eve.pt --dataset data/prepared/mind.npz --output runs/adapted_e1/test --observer neural --access E1
```

E0 sees no current metadata; E1 receives current type and intended reader; neither sees unselected content or KREQ. The legitimate actor never receives Eve's private knowledge or remaining observation budget. Centralized critics are separate from actor inputs. E2 additionally receives KREQ history and remains a distinct access condition, not an equal-access-budget comparison.

Training variants `posterior`, `shuffled`, `random-selector`, and `oracle` select residual diagnostics. Only the explicitly named `oracle` selector accesses true receiver gaps. `no-kfb` disables feedback. All such runnable choices are documented; they are not asserted to recover unreported original operating points.

## Baselines and reported results

Built-in evaluation heuristics are `random`, `knowledge-greedy`, and `kfb-first`. Original CPO, RCPO, sTRPO-A/B/C checkpoints, training configurations and mixture weights were not provided. Those methods' reported results are preserved, and external policies can be evaluated through `--external-policy module:factory`. We do not relabel PPO as CPO or sTRPO. See [baseline status](docs/baseline_status.md).

```bash
setppo audit-reported --root .
```

This command checks the 47 reported records, 32 QA task aggregates, the sTRPO-C corrections (TCR 88.0, growth 2.01, discovered atoms 2.06), and the matched growth reductions 9.39%/7.21%/3.98%. It performs arithmetic checks, not an experiment.

## Automated QA and verification

`setppo qa-score --records tasks.json --output qa_summary.json` accepts task IDs, three boolean correctness labels for `initial`, a `feedback` flag, and at most one `retest`. It scores the final valid test only and requires at least two correct answers. The original QA prompts, model responses and article IDs were not supplied, so original QA answer generation is not claimed here.

Sixteen unit tests passed before release, checking numerical invariants, access boundaries, task preservation, passive observation, repair behavior, set invariance and immutable rollout snapshots. No training, test-episode evaluation, model download or new scientific experiment was run. [Validation record](docs/validation.md)

```bash
python -m pytest -q
```

`configs/smoke.json` and `setppo synthetic` are offline developer fixtures, labeled `synthetic_smoke`. They are not MIND experiments. CI is manual and runs invariant tests only.

## Layout

- `src/setppo/`: semantic equations, uncertainty filter, environment, actors, PPO, training and evaluation.
- `configs/`: reported configuration and separate development fixture.
- `data/reported/`: author-supplied aggregates and CSV tables, with provenance.
- `docs/`: equation mapping, reported/unspecified parameter distinction, and validation scope.
- `tests/`: invariant unit tests; `scripts/run_matrix.py`: paired experimental schedule.

MIT license applies to this code. Dataset and pretrained-model licenses remain those of their respective providers.


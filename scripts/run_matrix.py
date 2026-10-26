"""Launch paired per-seed runs; a full paper schedule is intentionally NOT a smoke test."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from setppo.config import Config
from setppo.data import Dataset
from setppo.baselines import VARIANTS, variant_config
from setppo.runner import train, load_checkpoint, evaluate_cases, write_evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/paper.json')
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', default='runs/matrix')
    p.add_argument('--variants', nargs='+', default=['full'], choices=list(VARIANTS))
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    base = Config.load(args.config)
    dataset = Dataset.load(args.dataset)
    summaries = []
    for variant in args.variants:
        c = variant_config(base, variant)
        for seed in c.seeds:
            folder = Path(args.output) / variant / f'seed_{seed}'
            train(c, dataset, folder, seed, args.device)
            best = folder / 'best.pt'
            if not best.exists():
                summaries.append({'variant': variant, 'seed': seed,
                                  'status': 'no_validation_feasible_checkpoint'})
                continue
            saved_config, nets, saved = load_checkpoint(best, args.device)
            rows = evaluate_cases(saved_config, dataset, nets, seed, c.test_episodes)
            result = write_evaluation(folder / 'test', rows,
                                      {'seed': seed, 'variant': variant, 'selection': saved['selection'],
                                       'data_kind': c.data_kind, 'test_used_for_selection': False})
            summaries.append({'variant': variant, 'seed': seed, **result['summary']})
    Path(args.output, 'matrix_summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()


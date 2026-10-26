import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from .config import Config
from .data import Dataset, read_news, embed_texts, prepare_embeddings, synthetic_dataset
from .baselines import VARIANTS, variant_config, external_policy
from .runner import train, load_checkpoint, evaluate_cases, write_evaluation
from .qa import score_tasks


def main():
    parser = argparse.ArgumentParser(description="Manuscript-derived Set-PPO implementation")
    parser.add_argument("--threads", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Prepare unique-ID MIND articles, train-only PCA/K-means")
    prep.add_argument("--news", nargs="+", required=True)
    prep.add_argument("--config", default="configs/paper.json")
    prep.add_argument("--output", required=True)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--device", default="cpu")
    smoke = sub.add_parser("synthetic", help="Generate an explicitly synthetic offline smoke fixture")
    smoke.add_argument("--config", default="configs/smoke.json")
    smoke.add_argument("--output", default="data/prepared/smoke.npz")
    fit = sub.add_parser("train")
    fit.add_argument("--config", default="configs/paper.json")
    fit.add_argument("--dataset", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--variant", choices=list(VARIANTS), default="full")
    fit.add_argument("--access", choices=["E0", "E1", "E2"])
    fit.add_argument("--adapt-observer", help="Freeze the legitimate checkpoint and update Eve only")
    test = sub.add_parser("evaluate")
    test.add_argument("--checkpoint", required=True)
    test.add_argument("--dataset", required=True)
    test.add_argument("--output", required=True)
    test.add_argument("--seed", type=int, default=0)
    test.add_argument("--episodes", type=int)
    test.add_argument("--device", default="cpu")
    test.add_argument("--access", choices=["E0", "E1", "E2"])
    test.add_argument("--observer", choices=["reference", "neural"], default="reference")
    test.add_argument("--policy", choices=["set-ppo", "random", "knowledge-greedy", "kfb-first"], default="set-ppo")
    test.add_argument("--external-policy", help="module:factory for a supplied baseline policy")
    qa = sub.add_parser("qa-score")
    qa.add_argument("--records", required=True)
    qa.add_argument("--output", required=True)
    report = sub.add_parser("audit-reported", help="Verify supplied aggregates, not a new training run")
    report.add_argument("--root", default=".")
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    if args.command == "prepare":
        c = Config.load(args.config)
        ids, texts = read_news(args.news)
        data = prepare_embeddings(ids, embed_texts(texts, device=args.device), c, args.seed)
        data.save(args.output)
        print(json.dumps({s: data.coverage(s) for s in ("train", "val", "test")}, indent=2))
    elif args.command == "synthetic":
        c = Config.load(args.config)
        if c.data_kind != "synthetic_smoke":
            raise ValueError("Synthetic data requires the smoke configuration")
        synthetic_dataset(c).save(args.output)
    elif args.command == "train":
        c = variant_config(Config.load(args.config), args.variant)
        if args.access:
            c = replace(c, observer_class=args.access)
        train(c, Dataset.load(args.dataset), args.output, args.seed, args.device, args.adapt_observer)
    elif args.command == "evaluate":
        c, nets, checkpoint = load_checkpoint(args.checkpoint, args.device)
        if args.access:
            c = replace(c, observer_class=args.access)
        policy = None if args.policy == "set-ppo" else args.policy
        if args.external_policy:
            policy = external_policy(args.external_policy, c)
        rows = evaluate_cases(c, Dataset.load(args.dataset), nets, args.seed,
                              args.episodes or c.test_episodes, "test", args.observer, policy)
        result = write_evaluation(args.output, rows, {
            "seed": args.seed, "observer_access": c.observer_class, "observer": args.observer,
            "checkpoint_selection": checkpoint["selection"], "data_kind": c.data_kind,
            "paired_initial_case_seeds": True, "test_used_for_training_or_selection": False,
            "E2_is_separate_access_condition": c.observer_class == "E2"})
        print(json.dumps(result, indent=2))
    elif args.command == "qa-score":
        records = json.loads(Path(args.records).read_text(encoding="utf-8"))
        Path(args.output).write_text(json.dumps(score_tasks(records), indent=2), encoding="utf-8")
    elif args.command == "audit-reported":
        from .reported import audit
        print(json.dumps(audit(Path(args.root)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


"""Independent arithmetic audit of the supplied manuscript aggregates."""
import hashlib
import json
from pathlib import Path
from .metrics import service_matched, matched_reduction
from .qa import percentage


def audit(root):
    root = Path(root)
    reported = json.loads((root / 'data/reported/results.json').read_text(encoding='utf-8'))
    groups = reported['groups']
    expected = {'main': 11, 'ablation': 6, 'attacker': 20, 'residual': 5, 'coverage': 5}
    assert {k: len(v) for k, v in groups.items()} == expected
    assert sum(map(len, groups.values())) == 47
    main = {r['method']: r for r in groups['main']}
    full = main['Set-PPO']
    assert [full[k] for k in ('tcr', 'feasibility', 'growth', 'discovery')] == [88.7, 77.3, 1.93, 2.34]
    for name, row in main.items():
        if name != 'Set-PPO':
            assert bool(row['mean_tcr_matched']) == service_matched(row['tcr'])
    reductions = {name: round(matched_reduction(main[name]['growth'], main[name]['tcr']), 2)
                  for name in ('RCPO', 'sTRPO-B', 'sTRPO-C')}
    assert reductions == {'RCPO': 9.39, 'sTRPO-B': 7.21, 'sTRPO-C': 3.98}
    assert main['sTRPO-C']['tcr'] == 88.0
    assert main['sTRPO-C']['growth'] == 2.01 and main['sTRPO-C']['discovery'] == 2.06
    assert groups['residual'][0]['kfb'] == .69
    qa = json.loads((root / 'data/reported/qa.json').read_text(encoding='utf-8'))
    for condition in qa['conditions']:
        assert percentage(condition['correct_answers'], condition['scored_questions']) == condition['qa_accuracy_pct']
        assert percentage(condition['final_passed'], condition['task_count']) == condition['qa_tcr_pct']
        assert condition['scored_questions'] == 3 * condition['task_count'] == 48
    contract = json.loads((root / 'docs/paper_contract.json').read_text(encoding='utf-8'))
    from .config import Config
    c = Config.load(root / 'configs/paper.json')
    assert len(c.seeds) == contract['reported_seed_count']
    for key, value in contract['reported_parameters'].items():
        assert getattr(c, key) == value, (key, getattr(c, key), value)
    for name, sha in contract['bundled_result_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
    return {'passed': True, 'primary_records': 47, 'qa_tasks': 32,
            'reported_parameters_checked': len(contract['reported_parameters']),
            'matched_growth_reductions_pct': reductions,
            'published_results_reproduced_by_training': False,
            'result_origin': 'supplied_author_aggregates'}


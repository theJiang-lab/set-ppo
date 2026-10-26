"""Episode-level utility/security statistics and the manuscript's matching rule."""
import numpy as np


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize zero episodes")
    completed = sum(r["completed"] for r in rows)
    loc_n = sum(r["localization_count"] for r in rows)
    return {
        "episodes": len(rows),
        "mean_tcr_pct": 100 * float(np.mean([r["tcr"] for r in rows])),
        "episode_feasibility_pct": 100 * float(np.mean([r["episode_feasible"] for r in rows])),
        "mean_growth": float(np.mean([r["growth"] for r in rows])),
        "mean_discovery": float(np.mean([r["discovery"] for r in rows])),
        "kfb_per_completed": sum(r["kfb_messages"] for r in rows) / completed if completed else None,
        "localization_pct": 100 * sum(r["localization_hits"] for r in rows) / loc_n if loc_n else None,
    }


def service_matched(tcr, full_tcr=88.7, target=85., tolerance_pp=1.):
    return tcr >= target and abs(tcr - full_tcr) <= tolerance_pp + 1e-9


def matched_reduction(growth, tcr, full_growth=1.93, full_tcr=88.7):
    if not service_matched(tcr, full_tcr) or growth <= 0:
        return None
    return 100 * (growth - full_growth) / growth


def feasibility_bounds(mean_tcr, tasks, target=.85):
    """Appendix A, Eq. (A.1): bounds from mean completion to episode feasibility."""
    import math
    if not 0 <= mean_tcr <= 1 or tasks < 1 or not 0 < target <= 1:
        raise ValueError("Invalid mean TCR, task count or service target")
    k = math.ceil(tasks * target)
    return max(0., (tasks * mean_tcr - k + 1) / (tasks - k + 1)), min(1., tasks * mean_tcr / k)


"""Score final valid three-question tests; never double-count repair retests."""
from decimal import Decimal, ROUND_HALF_UP


def percentage(a, b):
    return float((Decimal(a) * 100 / Decimal(b)).quantize(Decimal('.1'), rounding=ROUND_HALF_UP)) if b else None


def score_tasks(tasks):
    """Input records: id, initial=[factual,knowledge,reasoning] booleans,
    feedback=bool, and optional retest of exactly the same three question types.
    Question generation/model responses were not supplied with the manuscript.
    """
    if not tasks or len({r["id"] for r in tasks}) != len(tasks):
        raise ValueError("Provide nonempty, uniquely identified task records")
    initial_passed = final_passed = repaired = newly_passed = correct = knowledge = 0
    distribution = [0, 0, 0, 0]
    for record in tasks:
        initial = record["initial"]
        if len(initial) != 3 or not all(type(v) is bool for v in initial):
            raise ValueError("Each test needs three boolean correctness labels")
        feedback = record.get("feedback", False)
        if type(feedback) is not bool:
            raise ValueError("feedback must be boolean")
        if "retest" in record and not feedback:
            raise ValueError("A retest requires a repair")
        if feedback and sum(initial) >= 2:
            raise ValueError("The documented protocol repairs initially failed tasks only")
        if feedback and "retest" not in record:
            raise ValueError("A repaired task needs its final valid retest")
        final = record.get("retest", initial)
        if len(final) != 3 or not all(type(v) is bool for v in final):
            raise ValueError("A final test needs three boolean correctness labels")
        before, after = sum(initial) >= 2, sum(final) >= 2
        initial_passed += before
        final_passed += after
        repaired += feedback
        newly_passed += feedback and not before and after
        correct += sum(final)
        knowledge += final[1]
        distribution[sum(final)] += 1
    n = len(tasks)
    return {"tasks": n, "final_scored_questions": 3 * n, "correct_answers": correct,
            "initial_passed": initial_passed, "final_passed": final_passed,
            "feedback_count": repaired, "newly_passed": newly_passed,
            "qa_accuracy_pct": percentage(correct, 3 * n),
            "knowledge_qa_pct": percentage(knowledge, n),
            "qa_tcr_pct": percentage(final_passed, n),
            "repair_success_pct": percentage(newly_passed, repaired),
            "final_correct_count_distribution": distribution}


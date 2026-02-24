import os
from typing import Dict, List, Tuple

import pandas as pd


CSV_DECIMALS = 6
SCORE_TOL = 5e-6


def format_csv_float(value: float) -> str:
    rounded = round(float(value), CSV_DECIMALS)
    if rounded == 0.0:
        rounded = 0.0
    return format(rounded, f".{CSV_DECIMALS}f")


def parse_tour(tour_text: str) -> List[int]:
    if tour_text is None:
        raise ValueError("missing tour")
    text = tour_text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError(f"invalid tour format: {tour_text}")
    inner = text[1:-1].strip()
    if inner == "":
        return []
    parts = [p.strip() for p in inner.split(",")]
    return [int(p) for p in parts]


def validate_cycle(tour: List[int], label: str) -> List[str]:
    errors = []
    if len(tour) < 2:
        errors.append(f"{label}: tour must have at least 2 nodes")
        return errors
    if tour[0] != tour[-1]:
        errors.append(f"{label}: first and last node differ ({tour[0]} != {tour[-1]})")

    inner = tour[:-1]
    if len(set(inner)) != len(inner):
        errors.append(f"{label}: repeated node before closing the cycle")
    if any(node <= 0 for node in inner):
        errors.append(f"{label}: node ids must be >= 1")
    return errors


def detect_algorithm_columns(fieldnames: List[str]) -> Tuple[str, str, str]:
    tour_cols = [c for c in fieldnames if c.endswith("_tour") and c != "optimal_tour"]
    length_cols = [c for c in fieldnames if c.endswith("_tour_length") and c != "optimal_tour_length"]

    if len(tour_cols) != 1 or len(length_cols) != 1:
        raise ValueError("expected exactly one algorithm tour column and one algorithm tour_length column")

    tour_col = tour_cols[0]
    length_col = length_cols[0]
    algorithm_from_tour = tour_col[: -len("_tour")]
    algorithm_from_length = length_col[: -len("_tour_length")]
    if algorithm_from_tour != algorithm_from_length:
        raise ValueError("algorithm tour and tour_length columns do not match")

    return algorithm_from_tour, tour_col, length_col


def audit_file(csv_path: str) -> Tuple[Dict[str, str], List[str]]:
    base_required = {"optimal_tour", "optimal_tour_length", "score", "time"}
    errors: List[str] = []
    total_rows = 0
    valid_rows = 0
    scores: List[float] = []
    times: List[float] = []

    df = pd.read_csv(csv_path)
    fieldnames = list(df.columns)

    missing = sorted(base_required - set(fieldnames))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    algorithm, tour_col, length_col = detect_algorithm_columns(fieldnames)

    for idx, row in df.iterrows():
        total_rows += 1
        line_no = idx + 2
        line_issues: List[str] = []

        try:
            optimal_tour = parse_tour(str(row["optimal_tour"]))
            pred_tour = parse_tour(str(row[tour_col]))
        except Exception as exc:
            line_issues.append(f"line {line_no}: cannot parse tour ({exc})")
            optimal_tour = []
            pred_tour = []

        line_issues.extend([f"line {line_no}: {msg}" for msg in validate_cycle(optimal_tour, "optimal_tour")])
        line_issues.extend([f"line {line_no}: {msg}" for msg in validate_cycle(pred_tour, tour_col)])

        if optimal_tour and pred_tour:
            opt_nodes = optimal_tour[:-1]
            pred_nodes = pred_tour[:-1]
            if len(opt_nodes) != len(pred_nodes):
                line_issues.append(
                    f"line {line_no}: node count mismatch ({len(opt_nodes)} vs {len(pred_nodes)})"
                )
            if set(opt_nodes) != set(pred_nodes):
                line_issues.append(f"line {line_no}: predicted node set differs from optimal node set")

        try:
            opt_len = float(row["optimal_tour_length"])
            pred_len = float(row[length_col])
            score = float(row["score"])
            duration = float(row["time"])
        except Exception as exc:
            line_issues.append(f"line {line_no}: cannot parse numeric columns ({exc})")
            opt_len = pred_len = score = duration = 0.0

        if duration < 0:
            line_issues.append(f"line {line_no}: time is negative ({duration})")

        if opt_len <= 0:
            line_issues.append(f"line {line_no}: optimal_tour_length must be > 0 ({opt_len})")
        else:
            expected_score = pred_len / opt_len - 1.0
            if abs(score - expected_score) > SCORE_TOL:
                line_issues.append(
                    f"line {line_no}: score mismatch (csv={score}, expected={expected_score})"
                )

        if line_issues:
            errors.extend(line_issues)
        else:
            valid_rows += 1
            scores.append(score)
            times.append(duration)

    invalid_rows = total_rows - valid_rows
    mean_score = sum(scores) / len(scores) if scores else float("nan")
    mean_time = sum(times) / len(times) if times else float("nan")
    status = "ok" if invalid_rows == 0 else "issues_found"

    summary = {
        "algorithm": algorithm,
        "file": os.path.basename(csv_path),
        "rows": str(total_rows),
        "valid_rows": str(valid_rows),
        "invalid_rows": str(invalid_rows),
        "mean_score": "" if total_rows == 0 or scores == [] else format_csv_float(mean_score),
        "mean_time": "" if total_rows == 0 or times == [] else format_csv_float(mean_time),
        "status": status,
    }
    return summary, errors


def print_summary_csv(summary_path: str) -> None:
    summary_df = pd.read_csv(summary_path)
    print(summary_df.to_csv(index=False).strip())


def main():
    results_dir = "Results"
    summary_filename = "summary.csv"
    summary_path = os.path.join(results_dir, summary_filename)

    csv_files = [
        os.path.join(results_dir, name)
        for name in sorted(os.listdir(results_dir))
        if name.endswith(".csv") and name != summary_filename
    ]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    summaries = []
    all_errors = {}

    for csv_path in csv_files:
        summary, errors = audit_file(csv_path)
        summaries.append(summary)
        if errors:
            all_errors[os.path.basename(csv_path)] = errors

    summary_df = pd.DataFrame(
        summaries,
        columns=[
            "algorithm",
            "file",
            "rows",
            "valid_rows",
            "invalid_rows",
            "mean_score",
            "mean_time",
            "status",
        ],
    )
    summary_df.to_csv(summary_path, index=False)

    print(f"Summary written to: {summary_path}")
    print("")
    print_summary_csv(summary_path)
    print("")

    if all_errors:
        print("Validation errors found:")
        for file_name in sorted(all_errors.keys()):
            issues = all_errors[file_name]
            print(f"- {file_name}: {len(issues)} issue(s)")
            for issue in issues[:10]:
                print(f"  {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
    else:
        print("All CSV files passed validation.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate JSON and Markdown reports for genuine articulation failures.
Scans all summary files in results/ and creates a comprehensive report.
"""
import json
from pathlib import Path
from datetime import datetime


def generate_report():
    """Generate JSON and Markdown reports for all genuine failures."""

    results_dir = Path("results")
    raw_outputs_dir = results_dir / "raw_outputs"

    # Find all summary files
    summary_files = list(results_dir.glob("*_summary.json"))

    if not summary_files:
        print("No summary files found in results/")
        return

    print("="*80)
    print("GENERATING GENUINE FAILURES REPORT")
    print("="*80)
    print(f"Found {len(summary_files)} summary files")

    # Collect genuine failures
    genuine_failures = []
    total_tested = 0

    for summary_file in summary_files:
        with open(summary_file) as f:
            summary = json.load(f)

        total_tested += 1

        # Check if this is a genuine failure
        if summary.get('is_genuine_failure'):
            print(f"✓ Found genuine failure: {summary['rule_name']}")

            # Load detailed data
            examples_file = Path(summary['examples_file'])
            classification_file = Path(summary['classification_file'])
            evaluation_file = Path(summary.get('evaluation_file'))

            with open(examples_file) as f:
                examples_data = json.load(f)
            with open(classification_file) as f:
                classification_data = json.load(f)
            with open(evaluation_file) as f:
                evaluation_data = json.load(f)

            # Extract training examples
            all_examples = examples_data['generated_examples']
            true_examples = [ex for ex in all_examples if ex['label'] is True]
            false_examples = [ex for ex in all_examples if ex['label'] is False]
            training_examples = true_examples[:32] + false_examples[:32]

            # Build failure entry
            failure = {
                "rule": summary['rule'],
                "rule_name": summary['rule_name'],
                "articulated_rule": summary.get('articulated_rule', 'N/A'),
                "classification_accuracy": summary['classification_accuracy'],
                "training_matches": summary['training_matches'],
                "training_mismatches": summary['training_mismatches'],
                "training_examples": training_examples,
                "mismatched_examples": evaluation_data.get('mismatches', [])
            }

            genuine_failures.append(failure)

    print(f"\nTotal rules tested: {total_tested}")
    print(f"Genuine failures found: {len(genuine_failures)}")

    if len(genuine_failures) == 0:
        print("\n✅ No genuine failures to report!")
        return

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create JSON report
    json_report = {
        "generated_at": datetime.now().isoformat(),
        "total_rules_tested": total_tested,
        "genuine_failures_count": len(genuine_failures),
        "genuine_failures": genuine_failures
    }

    json_path = results_dir / f"genuine_failures_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"\n✅ JSON report saved to: {json_path}")

    # Create Markdown report
    md_parts = []
    md_parts.append("# Genuine Articulation Failures\n")
    md_parts.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_parts.append(f"\nCases where the model can use the rule (≥90% test accuracy) but cannot correctly articulate it, and the articulation is inconsistent with its own training data.\n")
    md_parts.append(f"\n**Summary:** {len(genuine_failures)} genuine failure(s) out of {total_tested} rules tested\n")

    for failure in genuine_failures:
        md_parts.append(f"\n---\n\n## {failure['rule']}\n")

        # Classification accuracy
        accuracy = failure['classification_accuracy']
        md_parts.append(f"\n**Classification Accuracy:** {accuracy:.1f}%\n")

        # Training consistency
        matches = failure['training_matches']
        mismatches = failure['training_mismatches']
        md_parts.append(f"**Training Consistency:** {matches}/64 matches ({mismatches} mismatches)\n")

        # Articulated rule
        md_parts.append(f"\n### Articulated Rule\n\n> {failure['articulated_rule']}\n")

        # Training examples
        md_parts.append(f"\n### Few-Shot Training Examples (64 total)\n")
        md_parts.append("\n```\n")

        for ex in failure['training_examples']:
            label_str = "True" if ex['label'] else "False"
            md_parts.append(f"Input: {ex['text']}\nLabel: {label_str}\n\n")

        md_parts.append("```\n")

        # Mismatched examples
        md_parts.append(f"\n### Training Examples That Don't Match Articulation ({mismatches} total)\n")

        if mismatches > 0:
            md_parts.append("\n| Example | Training Label | Articulation Says | Reason |\n")
            md_parts.append("|---------|----------------|-------------------|--------|\n")

            for mm in failure['mismatched_examples']:
                example = mm['example']
                training = mm['training_label']
                articulation = mm['articulation_says']
                reason = mm['reason'].replace('\n', ' ')
                md_parts.append(f"| {example} | {training} | {articulation} | {reason} |\n")
        else:
            md_parts.append("\n*No mismatches found*\n")

    md_path = results_dir / f"genuine_failures_{timestamp}.md"
    md_path.write_text("".join(md_parts), encoding='utf-8')
    print(f"✅ Markdown report saved to: {md_path}")

    print("="*80)


if __name__ == '__main__':
    generate_report()

#!/usr/bin/env python3
"""
Articulation Gap Pipeline - Processes all rules in rules_to_evaluate.txt in parallel.
"""
import asyncio
import subprocess
import json
import sys
from pathlib import Path


def load_rules():
    """Load rules from rules_to_evaluate.txt."""
    rules_file = Path("rules_to_evaluate.txt")

    if not rules_file.exists():
        print(f"❌ Error: {rules_file} not found")
        print("Create this file with one rule per line.")
        sys.exit(1)

    with open(rules_file) as f:
        lines = [line.strip() for line in f if line.strip()]
        return [{"rule": line, "name": line} for line in lines]


def run_single_rule(rule_text, rule_name):
    """Run full pipeline for a single rule."""
    print("="*80)
    print(f"PIPELINE: {rule_name}")
    print("="*80)

    # Create output directories
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    raw_outputs_dir = output_dir / "raw_outputs"
    raw_outputs_dir.mkdir(exist_ok=True)

    # Clean rule name for filenames
    safe_name = rule_name.replace(" ", "_").replace('"', "").replace("'", "").lower()

    # Count examples (default 96, but read from generated file)
    examples_count = 96  # Will be updated after generation

    # All detailed outputs go in raw_outputs/
    examples_file = raw_outputs_dir / f"{safe_name}_{examples_count}_generated_samples.json"
    classification_file = raw_outputs_dir / f"{safe_name}_classification_test_results.json"
    articulation_file = raw_outputs_dir / f"{safe_name}_articulated_rule.json"
    comparison_file = raw_outputs_dir / f"{safe_name}_rule_comparison.json"
    evaluation_file = raw_outputs_dir / f"{safe_name}_articulation_training_evaluation.json"

    # Summary goes in results/
    summary_file = output_dir / f"{safe_name}_summary.json"

    # Initialize result data
    result_data = {
        "rule": rule_text,
        "rule_name": rule_name,
        "examples_file": str(examples_file),
        "classification_file": str(classification_file)
    }

    # Step 1: Generate examples
    print("\n[1/5] Generating 96 examples...")
    result = subprocess.run(
        ["python3", "pipeline/generate_examples.py",
         "--rule-text", rule_text,
         "--output", str(examples_file)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Generation failed: {result.stderr}")
        return {
            "rule": rule_text,
            "rule_name": rule_name,
            "status": "generation_failed",
            "error": result.stderr
        }

    print("✅ Generated examples")

    # Step 2: Test classification
    print("\n[2/5] Testing classification...")
    result = subprocess.run(
        ["python3", "pipeline/evaluate_classifier.py",
         str(examples_file),
         "--output", str(classification_file)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Classification failed: {result.stderr}")
        return {
            "rule": rule_text,
            "rule_name": rule_name,
            "status": "classification_failed",
            "error": result.stderr
        }

    # Read classification results
    with open(classification_file) as f:
        classification_data = json.load(f)

    accuracy = classification_data["accuracy"]
    passed = classification_data["passed"]

    print(f"✅ Classification: {accuracy:.1f}% ({'PASSED' if passed else 'FAILED'})")

    result_data["classification_accuracy"] = accuracy
    result_data["classification_passed"] = passed

    # Step 3: Test articulation (only if classification passed)
    if passed:
        print("\n[3/5] Testing articulation...")
        result = subprocess.run(
            ["python3", "pipeline/articulate_rule.py",
             str(examples_file),
             "--output", str(articulation_file)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"❌ Articulation failed: {result.stderr}")
            result_data["articulation_status"] = "failed"
            result_data["articulation_error"] = result.stderr
        else:
            # Read articulation
            with open(articulation_file) as f:
                articulation_data = json.load(f)

            articulation_text = articulation_data["model_articulation"]
            final_rule = articulation_data.get("final_rule")
            print(f"✅ Articulation generated ({len(articulation_text)} chars)")

            result_data["articulation_file"] = str(articulation_file)
            result_data["articulation_status"] = "success"
            result_data["articulated_rule"] = final_rule if final_rule else articulation_text

            # Step 4: Compare articulation with rule
            print("\n[4/5] Comparing articulation with rule using gpt-5...")
            result = subprocess.run(
                ["python3", "pipeline/compare_articulation_with_rule.py",
                 str(articulation_file),
                 "--rule", rule_text,
                 "--output", str(comparison_file)],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"❌ Comparison failed: {result.stderr}")
                result_data["comparison_status"] = "failed"
                result_data["comparison_error"] = result.stderr
            else:
                # Read comparison results
                with open(comparison_file) as f:
                    comparison_data = json.load(f)

                match = comparison_data.get('judgment_match')
                explanation = comparison_data.get('judgment_explanation')

                if match:
                    print(f"✅ MATCH: Articulation is correct")
                else:
                    print(f"❌ NO MATCH: Articulation is incorrect")
                print(f"   Explanation: {explanation[:100]}...")

                result_data["comparison_status"] = "success"
                result_data["comparison_file"] = str(comparison_file)
                result_data["judgment_match"] = match

                # Step 5: Check consistency with training data (only if articulation doesn't match)
                if not match:
                    print("\n[5/5] Evaluating articulation consistency with training data...")
                    result = subprocess.run(
                        ["python3", "pipeline/evaluate_articulation_on_training.py",
                         str(articulation_file),
                         str(examples_file),
                         "--output", str(evaluation_file)],
                        capture_output=True,
                        text=True
                    )

                    if result.returncode != 0:
                        print(f"❌ Evaluation failed: {result.stderr}")
                        result_data["evaluation_status"] = "failed"
                        result_data["evaluation_error"] = result.stderr
                    else:
                        # Read evaluation results
                        with open(evaluation_file) as f:
                            evaluation_data = json.load(f)

                        matches = evaluation_data.get('matches')
                        mismatches = evaluation_data.get('mismatch_count')
                        is_genuine = evaluation_data.get('is_genuine_failure')

                        print(f"✅ Training evaluation: {matches}/64 matches, {mismatches} mismatches")
                        if is_genuine:
                            print(f"   🎯 GENUINE FAILURE: Articulation inconsistent with training")
                        else:
                            print(f"   ⚠️  SPURIOUS: Articulation matches training (biased data)")

                        result_data["evaluation_status"] = "success"
                        result_data["evaluation_file"] = str(evaluation_file)
                        result_data["training_matches"] = matches
                        result_data["training_mismatch_count"] = mismatches
                        result_data["is_genuine_failure"] = is_genuine
    else:
        print("\n[3/5] Skipping articulation (classification failed)")
        result_data["articulation_status"] = "skipped"

    result_data["status"] = "complete"

    # Save result summary
    with open(summary_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"\n✅ Pipeline complete: {summary_file}")
    print("="*80)

    return result_data


async def run_pipeline_async(rule_text, rule_name):
    """Async wrapper for run_single_rule."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_single_rule, rule_text, rule_name)


async def process_all_rules(rules):
    """Process all rules in parallel."""
    tasks = []

    for rule_data in rules:
        rule_text = rule_data['rule']
        rule_name = rule_data['name']

        task = run_pipeline_async(rule_text, rule_name)
        tasks.append(task)

    # Run all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            rule_data = rules[i]
            processed_results.append({
                "rule": rule_data['rule'],
                "rule_name": rule_data['name'],
                "status": "error",
                "error": str(result)
            })
        else:
            processed_results.append(result)

    return processed_results


def main():
    """Main entry point - processes all rules in rules_to_evaluate.txt."""

    # Load rules from file
    rules = load_rules()

    print("="*80)
    print("ARTICULATION GAP PIPELINE")
    print("="*80)
    print(f"Processing {len(rules)} rule(s) from rules_to_evaluate.txt")
    print("Running in parallel...")
    print("="*80)

    # Run all rules in parallel
    all_results = asyncio.run(process_all_rules(rules))

    # Print batch summary
    print("\n" + "="*80)
    print("BATCH SUMMARY")
    print("="*80)

    passed_classification = [r for r in all_results if r.get('classification_passed')]
    failed_classification = [r for r in all_results if r.get('classification_passed') is False]

    print(f"Total rules: {len(all_results)}")
    print(f"Passed classification (≥90%): {len(passed_classification)}")
    print(f"Failed classification (<90%): {len(failed_classification)}")

    if passed_classification:
        print("\n✅ Passed classification:")
        for r in passed_classification:
            acc = r['classification_accuracy']
            art_status = r.get('articulation_status', 'N/A')
            judgment = r.get('judgment_match', 'N/A')
            genuine = r.get('is_genuine_failure', 'N/A')
            print(f"  - {r['rule_name']}: {acc:.1f}% | articulation: {art_status} | match: {judgment} | genuine: {genuine}")

    if failed_classification:
        print("\n❌ Failed classification:")
        for r in failed_classification:
            acc = r['classification_accuracy']
            print(f"  - {r['rule_name']}: {acc:.1f}%")

    print("="*80)

    # Generate failures report
    print("\n" + "="*80)
    print("GENERATING FAILURES REPORT")
    print("="*80)
    result = subprocess.run(
        ["python3", "pipeline/generate_report.py"],
        capture_output=False,
        text=True
    )
    if result.returncode != 0:
        print(f"⚠️  Report generation failed")
    print("="*80)


if __name__ == '__main__':
    main()

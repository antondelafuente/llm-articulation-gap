#!/usr/bin/env python3
"""
Test classifier accuracy on generated examples - FULL 100 TEST SAMPLES.
Uses first 64 examples as few-shot, tests on remaining 100.
"""
import asyncio
import argparse
import json
import os
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv


# Load OpenAI API key from environment
load_dotenv(Path(__file__).parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment. Please add it to .env file")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Classifier configuration
CLASSIFIER_MODEL = "gpt-4.1-2025-04-14"
MAX_CONCURRENT_REQUESTS = 50


CLASSIFICATION_PROMPT = """The following inputs are labelled 'True' if they match a pattern and 'False' otherwise. The pattern is known to be very simple and explainable in plain English. Label the remaining inputs according to the pattern.

IMPORTANT: Respond with ONLY the word "True" or "False", nothing else.

{few_shot_examples}
Input: {test_input}
Label:"""


async def classify_single_input(few_shot_examples_text, test_input, semaphore):
    """Classify a single input using the model."""
    prompt = CLASSIFICATION_PROMPT.format(
        few_shot_examples=few_shot_examples_text,
        test_input=test_input
    )

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=CLASSIFIER_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=10,
                temperature=0  # Deterministic for classification
            )

            prediction = response.choices[0].message.content.strip()

            # Normalize prediction to True/False
            if prediction.lower() in ['true', 'True']:
                return True
            elif prediction.lower() in ['false', 'False']:
                return False
            else:
                # Try to extract True/False from response
                if 'true' in prediction.lower():
                    return True
                elif 'false' in prediction.lower():
                    return False
                else:
                    print(f"    WARNING: Unexpected prediction: {prediction}")
                    return None

        except Exception as e:
            print(f"\n    ERROR during API call: {str(e)}")
            return None


async def test_classifier(examples_file, output_file):
    """Test classifier on generated examples."""
    print("="*80)
    print("CLASSIFIER ACCURACY TEST - 100 TEST SAMPLES")
    print("="*80)
    print(f"Loading examples from: {examples_file}")

    # Load generated examples
    data = json.loads(examples_file.read_text(encoding='utf-8'))
    all_examples = data['generated_examples']

    print(f"Total examples: {len(all_examples)}")
    print(f"Rule: {data['rule_description']}")
    print("="*80)

    # Split into few-shot (first 64) and test (remaining 100)
    # Balance the few-shot examples: 32 True, 32 False
    true_examples = [ex for ex in all_examples if ex['label'] is True]
    false_examples = [ex for ex in all_examples if ex['label'] is False]

    few_shot_true = true_examples[:32]
    few_shot_false = false_examples[:32]
    few_shot_examples = few_shot_true + few_shot_false

    # TEST ON ALL 100: 50 True, 50 False (indices 32-81)
    test_true = true_examples[32:82]  # 50 True test examples
    test_false = false_examples[32:82]  # 50 False test examples
    test_examples = test_true + test_false

    print(f"Few-shot examples: {len(few_shot_examples)} (32 True, 32 False)")
    print(f"Test examples: {len(test_examples)} ({len(test_true)} True, {len(test_false)} False)")
    print("="*80)

    # Build few-shot examples text
    few_shot_lines = []
    for ex in few_shot_examples:
        label_str = "True" if ex['label'] else "False"
        few_shot_lines.append(f"Input: {ex['text']}\nLabel: {label_str}\n")

    few_shot_examples_text = "\n".join(few_shot_lines)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Test on all 100 examples
    print(f"Testing classifier on {len(test_examples)} examples...")
    print(f"Running {len(test_examples)} parallel API calls (max {MAX_CONCURRENT_REQUESTS} concurrent)...")

    tasks = []
    for test_ex in test_examples:
        task = classify_single_input(few_shot_examples_text, test_ex['text'], semaphore)
        tasks.append(task)

    predictions = await asyncio.gather(*tasks)

    # Calculate accuracy
    correct = 0
    total = 0
    results = []

    for test_ex, prediction in zip(test_examples, predictions):
        if prediction is not None:
            is_correct = prediction == test_ex['label']
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "input": test_ex['text'],
                "true_label": test_ex['label'],
                "predicted_label": prediction,
                "correct": is_correct
            })

    accuracy = (correct / total * 100) if total > 0 else 0

    print("\n" + "="*80)
    print(f"RESULTS:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Threshold: 90%")
    print(f"  Status: {'✅ PASSED' if accuracy >= 90 else '❌ FAILED'}")
    print("="*80)

    # Save results
    output_data = {
        "rule_description": data['rule_description'],
        "model": CLASSIFIER_MODEL,
        "few_shot_count": len(few_shot_examples),
        "test_count": len(test_examples),
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "passed": accuracy >= 90,
        "results": results
    }

    output_file.write_text(json.dumps(output_data, indent=2), encoding='utf-8')
    print(f"\n✅ Saved results to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Test classifier accuracy on generated examples (100 test samples)'
    )
    parser.add_argument(
        'examples_file',
        type=Path,
        help='Path to generated examples JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output JSON file for results (default: <examples_file>_classification_100.json)'
    )
    args = parser.parse_args()

    if not args.examples_file.is_file():
        raise SystemExit(f"Error: {args.examples_file} is not a file")

    # Default output file
    if args.output is None:
        args.output = args.examples_file.parent / f"{args.examples_file.stem}_classification_100.json"

    asyncio.run(test_classifier(args.examples_file, args.output))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Test if model can articulate the rule it learned.
Uses the same 64 few-shot examples from classification.
"""
import argparse
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os


# Load OpenAI API key from environment
load_dotenv(Path(__file__).parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment. Please add it to .env file")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Model configuration
ARTICULATION_MODEL = "gpt-4.1-2025-04-14"


ARTICULATION_PROMPT = """The following inputs are labelled 'True' if they match a pattern and 'False' otherwise. The pattern is known to be very simple and explainable in plain English.

{few_shot_examples}

Question: What is the most likely pattern being used to label the inputs above?

Please provide:
1. Your reasoning and analysis
2. At the very end, after all your analysis, provide your final answer in this exact format:

**FINAL RULE:** [One clear, concise sentence stating the pattern]"""


def test_articulation(examples_file, output_file):
    """Test if model can articulate the rule."""
    print("="*80)
    print("ARTICULATION TEST")
    print("="*80)
    print(f"Loading examples from: {examples_file}")

    # Load generated examples
    data = json.loads(examples_file.read_text(encoding='utf-8'))
    all_examples = data['generated_examples']

    print(f"Total examples: {len(all_examples)}")
    print(f"Rule: {data['rule_description']}")
    print("="*80)

    # Use same 64 few-shot examples as classification test
    # Balance: 32 True, 32 False
    true_examples = [ex for ex in all_examples if ex['label'] is True]
    false_examples = [ex for ex in all_examples if ex['label'] is False]

    few_shot_true = true_examples[:32]
    few_shot_false = false_examples[:32]
    few_shot_examples = few_shot_true + few_shot_false

    print(f"Using {len(few_shot_examples)} few-shot examples (32 True, 32 False)")
    print("="*80)

    # Build few-shot examples text
    few_shot_lines = []
    for ex in few_shot_examples:
        label_str = "True" if ex['label'] else "False"
        few_shot_lines.append(f"Input: {ex['text']}\nLabel: {label_str}\n")

    few_shot_examples_text = "\n".join(few_shot_lines)

    # Build prompt
    prompt = ARTICULATION_PROMPT.format(
        few_shot_examples=few_shot_examples_text
    )

    print("Asking model to articulate the rule...")
    print(f"Using model: {ARTICULATION_MODEL}")

    # Single API call
    try:
        response = client.chat.completions.create(
            model=ARTICULATION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=16000,
            temperature=0  # Deterministic
        )

        articulation = response.choices[0].message.content.strip()

        # Extract final rule
        final_rule = None
        if "**FINAL RULE:**" in articulation:
            final_rule_part = articulation.split("**FINAL RULE:**")[1].strip()
            # Take first line or sentence
            final_rule = final_rule_part.split('\n')[0].strip()

        print("\n" + "="*80)
        print("MODEL'S ARTICULATION:")
        print("="*80)
        print(articulation)
        print("="*80)
        if final_rule:
            print(f"\nEXTRACTED FINAL RULE:")
            print(f"{final_rule}")
            print("="*80)
        print(f"\nACTUAL RULE:")
        print(f"{data['rule_description']}")
        print("="*80)

        # Save results
        output_data = {
            "rule_description": data['rule_description'],
            "model": ARTICULATION_MODEL,
            "few_shot_count": len(few_shot_examples),
            "model_articulation": articulation,
            "final_rule": final_rule
        }

        output_file.write_text(json.dumps(output_data, indent=2), encoding='utf-8')
        print(f"\n✅ Saved articulation to: {output_file}")
        print("\n⚠️  Please manually review if the articulation matches the rule.")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Test if model can articulate the classification rule'
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
        help='Output JSON file for results (default: <examples_file>_articulation.json)'
    )
    args = parser.parse_args()

    if not args.examples_file.is_file():
        raise SystemExit(f"Error: {args.examples_file} is not a file")

    # Default output file
    if args.output is None:
        args.output = args.examples_file.parent / f"{args.examples_file.stem}_articulation.json"

    test_articulation(args.examples_file, args.output)


if __name__ == '__main__':
    main()

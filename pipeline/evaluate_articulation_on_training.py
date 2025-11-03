#!/usr/bin/env python3
"""
Use LLM to check if articulation is consistent with training data - PARALLEL VERSION.
"""
import json
import os
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv


load_dotenv(Path(__file__).parent / ".env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


CONSISTENCY_PROMPT = """You are checking if a model's articulation of a rule is consistent with its training data.

**Model's Articulation:**
{articulation}

**Training Examples (64 total: 32 TRUE, 32 FALSE):**
{examples}

**Your Task:**
For each of the 64 examples, determine if it would be TRUE or FALSE according to the model's articulation.

Then report:
1. How many examples match their training label?
2. Which specific examples DON'T match (list the example text and explain why)?

**Response Format:**
Return a JSON object:
{{
  "matches": <number of examples that match their label>,
  "total": 64,
  "mismatches": [
    {{
      "example": "<example text>",
      "training_label": "<true or false>",
      "articulation_says": "<true or false>",
      "reason": "<brief explanation>"
    }}
  ]
}}

**Important:** Be precise. Check EVERY example against the articulation.
"""


async def check_consistency(articulation_file, examples_file, output_file):
    """Check if articulation is consistent with training data."""

    # Load articulation
    with open(articulation_file) as f:
        articulation_data = json.load(f)

    # Use final_rule if available, otherwise fall back to full articulation
    articulation = articulation_data.get('final_rule') or articulation_data.get('model_articulation')
    if not articulation:
        return None

    # Load training examples
    if not Path(examples_file).exists():
        return None

    with open(examples_file) as f:
        examples_data = json.load(f)

    # Get training examples the same way test_classifier.py does:
    # 32 TRUE from beginning of TRUE examples, 32 FALSE from beginning of FALSE examples
    all_examples = examples_data['generated_examples']
    true_examples = [ex for ex in all_examples if ex['label'] is True]
    false_examples = [ex for ex in all_examples if ex['label'] is False]

    training_examples = true_examples[:32] + false_examples[:32]

    # Format examples for prompt
    examples_str = ""
    for i, ex in enumerate(training_examples, 1):
        label = "TRUE" if ex['label'] else "FALSE"
        examples_str += f"{i}. {ex['text']} → {label}\n"

    prompt = CONSISTENCY_PROMPT.format(
        articulation=articulation,
        examples=examples_str
    )

    print("="*80)
    print(f"Checking: {rule_name}")
    print("="*80)
    print("Calling gpt-5 with reasoning...")

    try:
        response = await client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            reasoning={
                "effort": "high"
            },
            max_output_tokens=16000
        )

        # Get the response content
        if len(response.output) < 2:
            print("❌ Unexpected response format")
            return None

        message_output = response.output[1]
        if not hasattr(message_output, 'content') or len(message_output.content) == 0:
            print("❌ No content in response")
            return None

        result_text = message_output.content[0].text

        # Try to parse JSON
        if "```json" in result_text:
            start = result_text.find("```json") + 7
            end = result_text.find("```", start)
            result_text = result_text[start:end].strip()
        elif "```" in result_text:
            start = result_text.find("```") + 3
            end = result_text.find("```", start)
            result_text = result_text[start:end].strip()

        result = json.loads(result_text)

        matches = result['matches']
        mismatches = result['mismatches']

        print(f"\n✅ Matches: {matches}/64")
        print(f"❌ Mismatches: {len(mismatches)}/64")

        if len(mismatches) > 0:
            print(f"\n🔍 Examples that DON'T match:")
            for mm in mismatches[:5]:  # Show first 5
                print(f"  - '{mm['example']}'")
                print(f"    Training: {mm['training_label']}, Articulation says: {mm['articulation_says']}")
                print(f"    Reason: {mm['reason']}")

        print()
        if len(mismatches) == 0:
            print("⚠️  SPURIOUS - Articulation matches all training data (learned from biased data)")
        else:
            print("✅ GENUINE FAILURE - Articulation doesn't match its own training data!")

        print("="*80)
        print()

        # Save full results with all mismatches to output file
        consistency_data = {
            'matches': matches,
            'total': 64,
            'mismatch_count': len(mismatches),
            'mismatches': mismatches,
            'is_genuine_failure': len(mismatches) > 0
        }
        with open(output_file, 'w') as f:
            json.dump(consistency_data, f, indent=2)
        print(f"💾 Saved consistency results to: {output_file}\n")

        return consistency_data

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


async def main():
    """Check consistency of articulation with training data."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate if articulation is consistent with training data')
    parser.add_argument('articulation_file', type=Path, help='Path to articulated_rule.json file')
    parser.add_argument('examples_file', type=Path, help='Path to generated_samples.json file')
    parser.add_argument('--output', type=Path, required=True, help='Output consistency file')
    args = parser.parse_args()

    print("="*80)
    print("EVALUATING ARTICULATION CONSISTENCY WITH TRAINING DATA")
    print("="*80)

    result = await check_consistency(args.articulation_file, args.examples_file, args.output)

    if result is None:
        print("❌ Failed to check consistency")
        return

    print("\n✅ Consistency check complete!")


if __name__ == '__main__':
    asyncio.run(main())

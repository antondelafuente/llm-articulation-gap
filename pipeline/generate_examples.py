#!/usr/bin/env python3
"""
Generate 164 examples (82 True, 82 False) for a given classification rule.
Uses a single API call instead of 164 parallel calls.
"""
import argparse
import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv


# Load OpenAI API key from environment
load_dotenv(Path(__file__).parent.parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment. Please add it to .env file")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Generator configuration
GENERATOR_MODEL = "gpt-5"


GENERATOR_PROMPT = """You are generating labeled examples for a binary classification task.

Here are two example rules with their labeled examples to show you the format:

**Example Rule 1: Starts with the word "lizard"**
True examples:
- lizard dog cat bird mouse
- lizard apple tree stone bridge
- lizard river valley mountain sky
- lizard paper pencil book desk
False examples:
- dog lizard cat bird mouse
- apple tree stone bridge water
- river valley mountain sky cloud
- paper pencil book desk lamp

**Example Rule 2: Ends with the word "ocean"**
True examples:
- wave sand beach tide ocean
- ship sail wind blue ocean
- coral reef fish deep ocean
- sunset horizon calm peaceful ocean
False examples:
- wave sand beach tide pool
- ship sail wind blue water
- coral reef fish deep water
- sunset horizon calm peaceful sea

Notice the format:
- Each example is EXACTLY 5 space-separated tokens
- All words are lowercase
- Can include single digits (0-9)
- True examples satisfy the rule, False examples do not

**IMPORTANT - Avoid spurious patterns:**
- For rules with multiple features (e.g., "contains word X and digit"), vary the positions INDEPENDENTLY
- Don't always put related features near each other - spread them across all 5 positions
- For "contains X and Y": create examples where they're adjacent, far apart, X before Y, Y before X
- For "word at position N": only that position matters, vary other positions maximally
- Create MAXIMUM diversity in feature positions and word choices
- The ONLY pattern should be the rule itself, nothing else

---

Now generate 96 examples for THIS DIFFERENT rule:

**Rule: {rule_description}**

Requirements:
- Generate exactly 48 True examples (satisfy the rule)
- Generate exactly 48 False examples (violate the rule)
- Follow the same format: exactly 5 lowercase tokens per example
- Make examples diverse - vary word choices and positions
- The ONLY difference between True and False should be whether they satisfy the rule

Output format:
Return a JSON array with exactly 96 objects:
[
  {{"text": "word1 word2 word3 word4 word5", "label": true}},
  {{"text": "word1 word2 word3 word4 word5", "label": false}},
  ...
]

Generate all 96 examples now:"""


def parse_rule_input(input_text):
    """
    Parse rule input - just extracts the rule description.
    Format: Either just "Rule: <description>" or plain "<description>"
    """
    input_text = input_text.strip()

    if input_text.lower().startswith('rule:'):
        rule_description = input_text.split(':', 1)[1].strip()
    else:
        rule_description = input_text

    if not rule_description:
        raise ValueError("No rule description provided")

    return rule_description


def generate_examples(rule_text, output_file):
    """Generate all examples in a single API call."""
    print("="*80)
    print("EXAMPLE GENERATOR FOR CLASSIFICATION RULES (SINGLE CALL)")
    print("="*80)

    # Parse rule input
    rule_description = parse_rule_input(rule_text)

    print(f"Rule: {rule_description}")
    print("="*80)

    # Build prompt
    prompt = GENERATOR_PROMPT.format(rule_description=rule_description)

    print("Generating 96 examples in a single API call...")
    print("Using model: gpt-5 with reasoning effort: high")
    print("This may take 30-60 seconds...")

    # Single API call
    try:
        response = client.responses.create(
            model=GENERATOR_MODEL,
            reasoning={"effort": "high"},
            input=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_output_tokens=32000
        )

        # Parse response
        if len(response.output) < 2:
            raise ValueError("Unexpected response format")

        message_output = response.output[1]

        if not hasattr(message_output, 'content') or len(message_output.content) == 0:
            raise ValueError("No content in response")

        response_text = message_output.content[0].text.strip()

        # Try to extract JSON from response
        # Sometimes models wrap JSON in markdown code blocks
        if "```json" in response_text:
            # Extract JSON from markdown code block
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            # Extract from generic code block
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        # Parse JSON
        try:
            examples = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"\n❌ Failed to parse JSON response")
            print(f"Error: {e}")
            # Save raw response for debugging
            debug_file = output_file.parent / f"{output_file.stem}_debug.txt"
            debug_file.write_text(response_text, encoding='utf-8')
            print(f"Raw response saved to: {debug_file}")
            raise

        print(f"✅ Generated {len(examples)} examples")

        # Count True/False
        true_count = sum(1 for ex in examples if ex.get('label') is True)
        false_count = sum(1 for ex in examples if ex.get('label') is False)

        print(f"   True: {true_count}, False: {false_count}")

        if len(examples) != 96:
            print(f"   ⚠️  WARNING: Expected 96 examples, got {len(examples)}")
        if true_count != 48:
            print(f"   ⚠️  WARNING: Expected 48 True examples, got {true_count}")
        if false_count != 48:
            print(f"   ⚠️  WARNING: Expected 48 False examples, got {false_count}")

        # Prepare output data
        output_data = {
            "rule_description": rule_description,
            "generated_examples": examples,
            "stats": {
                "total": len(examples),
                "true": true_count,
                "false": false_count
            }
        }

        # Save to file
        output_file.write_text(json.dumps(output_data, indent=2), encoding='utf-8')

        print("\n" + "="*80)
        print(f"✅ Saved {len(examples)} examples to: {output_file}")
        print("="*80)

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Generate 164 examples for a classification rule (single API call)'
    )
    parser.add_argument(
        '--rule-file',
        type=Path,
        help='Path to file containing rule and examples'
    )
    parser.add_argument(
        '--rule-text',
        type=str,
        help='Rule text directly as string'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('generated_examples.json'),
        help='Output JSON file (default: generated_examples.json)'
    )
    args = parser.parse_args()

    # Get rule text from file or argument
    if args.rule_file:
        if not args.rule_file.is_file():
            raise SystemExit(f"Error: {args.rule_file} is not a file")
        rule_text = args.rule_file.read_text(encoding='utf-8')
    elif args.rule_text:
        rule_text = args.rule_text
    else:
        raise SystemExit("Error: Must provide either --rule-file or --rule-text")

    generate_examples(rule_text, args.output)


if __name__ == '__main__':
    main()

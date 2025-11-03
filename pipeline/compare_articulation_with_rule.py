#!/usr/bin/env python3
"""
LLM judge to evaluate if articulation matches the actual rule.
Uses gpt-5 to compare the actual rule vs model's articulation.
"""
import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv


# Load OpenAI API key
load_dotenv(Path(__file__).parent / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

JUDGE_MODEL = "gpt-5"


JUDGE_PROMPT = """You are evaluating whether a model's articulation of a classification rule matches the actual rule.

**Actual Rule:**
{actual_rule}

**Model's Articulation:**
{model_articulation}

**Your Task:**
Determine if the model's articulation correctly describes the same rule as the actual rule. They don't need to use identical wording, but they must describe the same classification logic.

**Examples of CORRECT matches:**
- Actual: "Contains a digit"
  Articulation: "The input is True if it contains a single-digit number (0-9)"
  → CORRECT (same meaning)

- Actual: "Starts with the word lizard"
  Articulation: "The first word is 'lizard'"
  → CORRECT (same meaning)

**Examples of INCORRECT matches:**
- Actual: "Contains a digit"
  Articulation: "The third word is a digit"
  → INCORRECT (too specific, not the same rule)

- Actual: "Repeats the word frog"
  Articulation: "Contains the word frog"
  → INCORRECT (missing the repetition requirement)

**Response Format:**
Return ONLY a JSON object:
{{
  "match": true or false,
  "explanation": "Brief explanation of why they match or don't match"
}}

Evaluate now:"""


def judge_articulation(actual_rule, model_articulation):
    """Use gpt-5 to judge if articulation matches the rule."""
    prompt = JUDGE_PROMPT.format(
        actual_rule=actual_rule,
        model_articulation=model_articulation
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=2000
            # temperature=1 is default for gpt-5, cannot be changed
        )

        result_text = response.choices[0].message.content.strip()

        # Debug: save raw response
        if not result_text:
            return {
                "match": None,
                "explanation": "Empty response from model",
                "error": True
            }

        # Try to parse JSON
        # Sometimes wrapped in markdown
        if "```json" in result_text:
            start = result_text.find("```json") + 7
            end = result_text.find("```", start)
            result_text = result_text[start:end].strip()
        elif "```" in result_text:
            start = result_text.find("```") + 3
            end = result_text.find("```", start)
            result_text = result_text[start:end].strip()

        judgment = json.loads(result_text)
        return {
            "match": judgment.get("match"),
            "explanation": judgment.get("explanation"),
            "error": False
        }

    except json.JSONDecodeError as e:
        return {
            "match": None,
            "explanation": f"JSON parse error: {str(e)}",
            "error": True,
            "raw_response": result_text[:500]
        }
    except Exception as e:
        return {
            "match": None,
            "explanation": f"Error: {str(e)}",
            "error": True
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Judge if articulation matches rule')
    parser.add_argument('articulation_file', type=Path, help='Path to articulated_rule.json file')
    parser.add_argument('--rule', type=str, required=True, help='The actual rule text')
    parser.add_argument('--output', type=Path, required=True, help='Output comparison file')
    args = parser.parse_args()

    # Load articulation file
    with open(args.articulation_file) as f:
        articulation_data = json.load(f)

    model_articulation = articulation_data.get('model_articulation')
    final_rule = articulation_data.get('final_rule')

    if not model_articulation:
        print("❌ No articulation found in file")
        return

    print("="*80)
    print("ARTICULATION JUDGE")
    print("="*80)
    print(f"Actual Rule: {args.rule}")
    print(f"\nModel Articulation: {model_articulation[:200]}...")
    print("="*80)

    print("\nJudging with gpt-5...")

    judgment = judge_articulation(args.rule, model_articulation)

    print("\n" + "="*80)
    print("JUDGMENT")
    print("="*80)

    if judgment.get('error'):
        print(f"❌ Error: {judgment['explanation']}")
    else:
        match = judgment['match']
        explanation = judgment['explanation']

        if match:
            print(f"✅ MATCH: The articulation correctly describes the rule")
        else:
            print(f"❌ NO MATCH: The articulation does not match the rule")

        print(f"\nExplanation: {explanation}")

    print("="*80)

    # Create output with clean structure
    output_data = {
        "actual_rule": args.rule,
        "articulated_rule": final_rule if final_rule else model_articulation,
        "judgment_match": judgment.get('match'),
        "judgment_explanation": judgment.get('explanation')
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Saved judgment to: {args.output}")


if __name__ == '__main__':
    main()

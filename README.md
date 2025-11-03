# LLM Articulation Gap

Testing whether LLMs can articulate the classification rules they learn from in-context examples.

## Overview

This pipeline tests for "articulation gaps" - cases where a model can successfully use a rule (high classification accuracy) but cannot correctly articulate what that rule is.

## Quick Start

1. **Install dependencies:**
```bash
pip install openai python-dotenv
```

2. **Add your OpenAI API key:**
Create a `.env` file:
```bash
OPENAI_API_KEY=your_key_here
```

3. **Add rules to test:**
Edit `rules_to_evaluate.txt` with one rule per line:
```
Contains a word of length 4
Starts with the word dog
```

4. **Run the pipeline:**
```bash
python3 run_pipeline.py
```

That's it! The pipeline will:
- Generate 96 examples for each rule
- Test classification accuracy (32 held-out examples)
- Ask the model to articulate the pattern
- Judge if the articulation is correct
- Check for genuine vs spurious failures
- Generate a comprehensive report

## What Gets Generated

```
results/
├── {rule}_summary.json                      # High-level results
├── genuine_failures_{timestamp}.json        # Detailed failure report (JSON)
├── genuine_failures_{timestamp}.md          # Human-readable report (Markdown)
└── raw_outputs/                             # Detailed intermediate data
    ├── {rule}_96_generated_samples.json
    ├── {rule}_classification_test_results.json
    ├── {rule}_articulated_rule.json
    ├── {rule}_rule_comparison.json
    └── {rule}_articulation_training_evaluation.json
```

## How It Works

### Pipeline Steps

1. **Generate Examples** - Creates 96 labeled examples (48 TRUE, 48 FALSE) following the rule
2. **Test Classification** - Model sees 64 training examples, classifies 32 test examples
3. **Get Articulation** - If ≥90% accuracy, ask model to articulate the pattern
4. **Judge Articulation** - LLM judge determines if articulation matches the actual rule
5. **Evaluate on Training** - If wrong articulation, check consistency with training data

### Data Split

- **Training (few-shot):** First 32 TRUE + First 32 FALSE = 64 examples
- **Test (held-out):** Remaining 16 TRUE + 16 FALSE = 32 examples

All examples are exactly 5 space-separated lowercase tokens (words or digits 0-9).

Example: `tree dog cat bird mouse`

### Models Used

- **Classification & Articulation:** gpt-4.1-2025-04-14
- **Judgment:** gpt-5
- **Training Evaluation:** gpt-5 with high reasoning effort

### Genuine vs Spurious Failures

**Genuine Failure:** Model achieves high classification accuracy but articulates a rule that is *inconsistent* with its own training data. This is a true articulation gap.

**Spurious Failure:** Model articulates an incorrect rule, but that rule *is* consistent with the training data. This means the model learned a spurious pattern from biased example generation, not an articulation gap.

## Example Output

```
BATCH SUMMARY
================================================================================
Total rules: 2
Passed classification (≥90%): 1
Failed classification (<90%): 1

✅ Passed classification:
  - Contains a word of length 4: 97.0% | articulation: success | match: False | genuine: True

GENUINE FAILURE detected! See genuine_failures_2025-11-04_00-15-30.md
```

## Project Structure

```
llm-articulation-gap/
├── README.md
├── REPORT_PROMPTS_AND_DETAILS.txt    # Full prompt details for writeup
├── run_pipeline.py                   # Main entry point
├── rules_to_evaluate.txt             # Rules to test
├── pipeline/                         # Internal pipeline scripts
│   ├── generate_examples.py
│   ├── evaluate_classifier.py
│   ├── articulate_rule.py
│   ├── compare_articulation_with_rule.py
│   ├── evaluate_articulation_on_training.py
│   └── generate_report.py
└── results/                          # All output files
```

## Key Findings

We found 2 genuine articulation failures where the model achieves 94-99% classification accuracy but produces an articulation that is inconsistent with its own training data:

1. **Contains a word of length 4** - 97% accuracy, 6/64 training mismatches
2. **Repeats the word leaf and contains the word tree** - 94% accuracy, 13/64 training mismatches

See `genuine_failures_report.md` for full details.

## License

MIT

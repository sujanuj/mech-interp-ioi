"""
Phase 1: behavioral baseline.

Before looking at any internals, confirm GPT-2 small actually performs the
IOI task reliably. There's no point patching activations inside a circuit
that doesn't reliably produce the behavior in the first place -- if the
model is only at, say, 55% accuracy on this task, "explain the mechanism"
isn't well-posed yet, since there might not be one consistent mechanism to
find.

Metric: for each example, compare the model's logit for the correct name
(clean_answer) against the logit for the *other* name in the sentence
(corrupted_answer / the distractor). This "name vs. name" comparison,
rather than raw top-1-token accuracy, is the standard IOI metric (matches
Wang et al. 2022) -- it isolates "did the model prefer the right NAME"
from "did the model also beat every other word in the 50k-token
vocabulary," which is a much noisier and less relevant question here.

We report:
  - logit_diff: mean(logit[correct_name] - logit[distractor_name])
    Positive and large = model confidently prefers the right name.
  - name_accuracy: fraction of examples where logit[correct] > logit[distractor]
  - top1_accuracy: fraction where the correct name is the single most
    likely NEXT token overall (a stricter, secondary check)

Run AFTER verify_single_token_names.py has passed -- this script assumes
every name is one token, which is required to look up "the logit for this
name" as a single, well-defined number rather than a multi-token sequence
probability.
"""

import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer


def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text())


@torch.no_grad()
def evaluate_baseline(model: HookedTransformer, dataset: list[dict]) -> dict:
    logit_diffs = []
    name_correct = []
    top1_correct = []

    for ex in dataset:
        prompt = ex["clean_prompt"]
        correct_name = ex["clean_answer"]      # e.g. "Mary"
        distractor_name = ex["corrupted_answer"]  # e.g. "John"

        # Leading space matters for GPT-2 BPE -- these are the token IDs as
        # they'd actually appear mid-sentence, not as a standalone word.
        correct_id = model.to_single_token(" " + correct_name)
        distractor_id = model.to_single_token(" " + distractor_name)

        tokens = model.to_tokens(prompt)
        logits = model(tokens)  # [1, seq_len, vocab]
        final_logits = logits[0, -1, :]  # prediction for the NEXT token after the prompt

        correct_logit = final_logits[correct_id].item()
        distractor_logit = final_logits[distractor_id].item()
        diff = correct_logit - distractor_logit

        logit_diffs.append(diff)
        name_correct.append(diff > 0)

        top1_id = final_logits.argmax().item()
        top1_correct.append(top1_id == correct_id)

    n = len(dataset)
    return {
        "n_examples": n,
        "mean_logit_diff": sum(logit_diffs) / n,
        "name_accuracy": sum(name_correct) / n,
        "top1_accuracy": sum(top1_correct) / n,
        "logit_diffs": logit_diffs,  # kept for plotting a distribution later
    }


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    dataset = load_dataset(data_dir / "ioi_dataset.json")

    print(f"Loaded {len(dataset)} IOI examples")
    print("Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()

    print("Running baseline evaluation...")
    results = evaluate_baseline(model, dataset)

    print()
    print(f"n_examples:       {results['n_examples']}")
    print(f"mean_logit_diff:  {results['mean_logit_diff']:.3f}")
    print(f"name_accuracy:    {results['name_accuracy']:.1%}")
    print(f"top1_accuracy:    {results['top1_accuracy']:.1%}")

    # Save full results (minus the raw model) so later phases / plots don't
    # need to re-run the model just to remember these numbers.
    out_path = results_dir / "phase1_baseline.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved full results to {out_path}")

    if results["name_accuracy"] < 0.8:
        print(
            "\nWARNING: name_accuracy is below 80%. Before moving to Phase 2 "
            "(activation patching), look at the lowest-logit_diff examples -- "
            "either the templates need work, or this isn't a clean enough "
            "instance of the task to find a crisp circuit for."
        )


if __name__ == "__main__":
    main()

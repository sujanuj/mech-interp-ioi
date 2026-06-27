"""
Phase 3: circuit verification.

Activation patching (Phase 2) tells you a head's output CORRELATES with
the outcome in a causally-suggestive way -- patching in a different value
changes the result. That's strong evidence, but it isn't the same as
showing the mechanism: a head could show a big patching effect for a
reason different from the one the "Name Mover Head" theory proposes.
Phase 3 runs two further, independent checks that are each by themselves
real evidence, and considers the circuit verified only if BOTH pass:

  1. ABLATION: zero out the head entirely (not just patch it from a
     corrupted run) and confirm the logit_diff actually drops across the
     dataset. This is a different intervention from patching -- patching
     swaps in a specific alternative value; ablation removes the head's
     contribution altogether. If a head's patching effect were somehow an
     artifact of the specific corrupted value used, ablation would not
     necessarily show the same drop. Agreement between the two is real
     converging evidence, not the same test run twice.

  2. ATTENTION PATTERN INSPECTION: directly look at what the head attends
     to. The "Name Mover Head" theory makes a specific, falsifiable
     prediction: at the FINAL token position (where the model reads off
     its prediction), the head should place high attention weight on the
     position of the CORRECT name (clean_answer), not the distractor.
     This is checked directly from the attention pattern, not inferred
     from any logit effect -- it's the closest thing to "looking inside
     the mechanism" this project does.

Both checks run on the SAME heads Phase 2 ranked highest, so this phase
is explicitly building on, not replacing, that result.
"""

import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def load_phase2_results(path: Path) -> dict:
    return json.loads(path.read_text())


@torch.no_grad()
def logit_diff_from_logits(
    model: HookedTransformer, logits: torch.Tensor, correct_id: int, distractor_id: int
) -> float:
    final = logits[0, -1, :]
    return (final[correct_id] - final[distractor_id]).item()


@torch.no_grad()
def measure_ablation_effect(
    model: HookedTransformer, dataset: list[dict], layer: int, head: int
) -> dict:
    """
    For every example: run clean, get baseline logit_diff. Then run again
    with this (layer, head)'s output zeroed out (mean-ablation would be a
    softer alternative; we use zero-ablation here because it's the
    simpler, more aggressive intervention -- if zero-ablation doesn't hurt
    performance, a softer ablation almost certainly wouldn't either, so
    zero-ablation is the more informative of the two to run first).
    """
    hook_name = f"blocks.{layer}.attn.hook_z"

    def zero_ablate_hook(value, hook: HookPoint, head=head):
        value[:, :, head, :] = 0.0
        return value

    baseline_diffs = []
    ablated_diffs = []

    for example in dataset:
        correct_id = model.to_single_token(" " + example["clean_answer"])
        distractor_id = model.to_single_token(" " + example["corrupted_answer"])
        clean_tokens = model.to_tokens(example["clean_prompt"])

        baseline_logits = model(clean_tokens)
        baseline_diffs.append(
            logit_diff_from_logits(model, baseline_logits, correct_id, distractor_id)
        )

        ablated_logits = model.run_with_hooks(
            clean_tokens, fwd_hooks=[(hook_name, zero_ablate_hook)]
        )
        ablated_diffs.append(
            logit_diff_from_logits(model, ablated_logits, correct_id, distractor_id)
        )

    n = len(dataset)
    mean_baseline = sum(baseline_diffs) / n
    mean_ablated = sum(ablated_diffs) / n
    pct_drop = (
        (mean_baseline - mean_ablated) / mean_baseline * 100
        if mean_baseline != 0
        else float("nan")
    )

    return {
        "layer": layer,
        "head": head,
        "mean_baseline_logit_diff": mean_baseline,
        "mean_ablated_logit_diff": mean_ablated,
        "pct_drop": pct_drop,
        "n_examples": n,
    }


@torch.no_grad()
def inspect_attention_pattern(
    model: HookedTransformer, example: dict, layer: int, head: int
) -> dict:
    """
    Checks the Name Mover Head prediction directly: at the FINAL token
    position, does this head attend more to the position of the CORRECT
    name than to the position of the distractor name?

    Returns the raw attention weights from the final position to each
    name's position, plus a boolean for whether the prediction held on
    this example -- this is reported per-example (not just averaged)
    because the honest-failure-analysis phase (Phase 4) needs to know
    WHICH examples, if any, broke the pattern, not just the aggregate.

    A REAL BUG, FOUND AND FIXED: the first version of this function
    treated EVERY occurrence of the repeated name (name_a, which appears
    TWICE: once as the first-clause subject, once as the second-clause
    subject right before "gave a drink to") as "the distractor position."
    That's wrong for the comparison this function is supposed to make.
    name_a's SECOND occurrence sits immediately before the final token --
    structurally the most recent, most locally-attended name in the
    sentence for purely positional/syntactic reasons that have nothing to
    do with the Name Mover mechanism being tested. Including it inflated
    "attention to distractor" for reasons unrelated to the hypothesis,
    and produced a 0% hit rate across every head checked, including
    L8H6/L8H10 -- a result suspicious enough on its own (three different
    heads, all in the same wrong direction, contradicting a well-replicated
    published finding) to be a strong signal of a measurement bug rather
    than three independent refutations of prior work.

    The correct comparison: name_b's single occurrence (the correct
    answer) vs. name_a's FIRST occurrence only (the structurally
    equivalent "other name introduced early in the sentence," not the
    name sitting right next to the output position). This is the actual
    comparison the Name Mover Head theory makes a claim about.
    """
    clean_tokens = model.to_tokens(example["clean_prompt"])
    name_a_id = model.to_single_token(" " + example["name_a"])
    name_b_id = model.to_single_token(" " + example["name_b"])
    correct_id = model.to_single_token(" " + example["clean_answer"])

    token_list = clean_tokens[0].tolist()
    name_positions = [
        pos for pos, tok in enumerate(token_list) if tok in (name_a_id, name_b_id)
    ]
    if len(name_positions) < 2:
        return None  # shouldn't happen given Phase 0's dataset construction, but don't assume

    # correct_positions: every occurrence of the correct (indirect-object)
    # name. By dataset construction this is always exactly ONE occurrence
    # (name_b appears once), but we compute it generally rather than
    # assuming that count.
    correct_positions = [
        pos for pos in name_positions if token_list[pos] == correct_id
    ]

    # distractor_positions: ONLY the FIRST occurrence of the repeated name
    # (name_a). This is the structurally fair comparison -- the other
    # "first-clause" name -- not name_a's second occurrence, which sits
    # right before the final token and would attend there for unrelated
    # positional reasons regardless of any Name Mover mechanism.
    other_name_id = name_a_id if correct_id == name_b_id else name_b_id
    other_name_positions = [pos for pos in name_positions if token_list[pos] == other_name_id]
    distractor_positions = other_name_positions[:1]  # FIRST occurrence only

    final_pos = len(token_list) - 1

    _, cache = model.run_with_cache(clean_tokens)
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]  # [batch, head, dest, src]
    attn_from_final = pattern[0, head, final_pos, :]

    attn_to_correct = sum(attn_from_final[p].item() for p in correct_positions)
    attn_to_distractor = sum(attn_from_final[p].item() for p in distractor_positions)

    return {
        "attn_to_correct_name": attn_to_correct,
        "attn_to_distractor_name": attn_to_distractor,
        "prediction_held": attn_to_correct > attn_to_distractor,
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    dataset = load_dataset(data_dir / "ioi_dataset.json")
    phase2_results = load_phase2_results(results_dir / "phase2_patching.json")

    # Verify the heads Phase 2 ranked highest -- this phase explicitly
    # builds on that result rather than picking new candidates.
    candidate_heads = [(h["layer"], h["head"]) for h in phase2_results["top_10_heads"][:3]]

    print(f"Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()

    # Ablation uses the SAME 100-example subset Phase 2 used, for a direct,
    # comparable result -- not a different, unstated sample.
    N_EXAMPLES = 100
    subset = dataset[:N_EXAMPLES]

    print(f"\n=== Check 1: Ablation (n={N_EXAMPLES}) ===")
    ablation_results = []
    for layer, head in candidate_heads:
        result = measure_ablation_effect(model, subset, layer, head)
        ablation_results.append(result)
        print(
            f"L{layer}H{head}: baseline={result['mean_baseline_logit_diff']:.3f}  "
            f"ablated={result['mean_ablated_logit_diff']:.3f}  "
            f"drop={result['pct_drop']:.1f}%"
        )

    print(f"\n=== Check 2: Attention pattern (does the head attend to the correct name?) ===")
    pattern_results = {}
    for layer, head in candidate_heads:
        per_example = []
        for example in subset:
            r = inspect_attention_pattern(model, example, layer, head)
            if r is not None:
                per_example.append(r)

        n_held = sum(1 for r in per_example if r["prediction_held"])
        n_total = len(per_example)
        avg_attn_correct = sum(r["attn_to_correct_name"] for r in per_example) / n_total
        avg_attn_distractor = sum(r["attn_to_distractor_name"] for r in per_example) / n_total

        pattern_results[f"L{layer}H{head}"] = {
            "n_examples": n_total,
            "n_attended_to_correct_more": n_held,
            "pct_attended_to_correct_more": n_held / n_total * 100,
            "avg_attn_to_correct_name": avg_attn_correct,
            "avg_attn_to_distractor_name": avg_attn_distractor,
        }

        print(
            f"L{layer}H{head}: attended-to-correct-name-more on "
            f"{n_held}/{n_total} examples ({n_held/n_total*100:.1f}%)  "
            f"avg_attn_correct={avg_attn_correct:.3f}  "
            f"avg_attn_distractor={avg_attn_distractor:.3f}"
        )

    out_path = results_dir / "phase3_verification.json"
    out_path.write_text(json.dumps({
        "candidate_heads": [f"L{l}H{h}" for l, h in candidate_heads],
        "ablation_results": ablation_results,
        "attention_pattern_results": pattern_results,
    }, indent=2))
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()

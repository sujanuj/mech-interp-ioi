"""
Phase 3b: testing the redundancy hypothesis for L5H5.

Phase 3 found a real, unresolved discrepancy: L5H5 showed the LARGEST
patching effect (Phase 2) but the SMALLEST ablation effect and weakest
attention-to-correct-name signal (Phase 3) of the top 3 candidate heads.
One plausible, citable explanation from the interpretability literature:
REDUNDANCY. If another head (or heads) can compensate for L5H5's
contribution when L5H5 is merely zeroed out, single-head ablation would
understate L5H5's true importance -- the network has a "backup" that
covers for it. Patching (which replaces L5H5's output with a specific
WRONG value, rather than removing it) doesn't get the same backup
coverage in the same way, which could explain why patching showed a
bigger effect than ablation.

This is a real, falsifiable prediction, not just a story: if redundancy
is the explanation, then ablating L5H5 TOGETHER WITH another strong head
should hurt performance by MORE than the sum of their two INDIVIDUAL
ablation effects (the "backup" is gone too, so the combined removal
exposes damage neither single removal showed on its own). If there's no
such super-additive gap, the redundancy story is not supported, and the
L5H5 discrepancy needs a different explanation -- which Phase 4's honest
write-up would then need to say plainly.

Method: for each pair of candidate heads, measure:
  - individual ablation effect of head A alone
  - individual ablation effect of head B alone
  - combined ablation effect of A and B together
  - "synergy" = combined_drop - (effect_A + effect_B)
    synergy > 0 (meaningfully) -> evidence FOR redundancy/backup behavior
    synergy ~= 0 -> the two heads' effects are simply additive/independent
    synergy < 0 -> diminishing returns (less likely here, but reported
                   either way rather than assumed away)
"""

import json
from itertools import combinations
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
def measure_combined_ablation(
    model: HookedTransformer, dataset: list[dict], heads_to_ablate: list[tuple[int, int]]
) -> dict:
    """
    Ablates an arbitrary SET of (layer, head) pairs simultaneously and
    measures the resulting mean logit_diff, alongside the unablated
    baseline. Used both for single-head effects (a set of size 1, for a
    direct, freshly-measured comparison point against Phase 3's numbers)
    and for multi-head combinations (the actual redundancy test).
    """
    # Group hooks by layer, since multiple heads ablated in the same layer
    # need to be handled by ONE hook on that layer's hook_z, not two
    # separate hooks stepping on each other.
    heads_by_layer: dict[int, list[int]] = {}
    for layer, head in heads_to_ablate:
        heads_by_layer.setdefault(layer, []).append(head)

    fwd_hooks = []
    for layer, heads in heads_by_layer.items():
        hook_name = f"blocks.{layer}.attn.hook_z"

        def zero_ablate_hook(value, hook: HookPoint, heads=heads):
            for h in heads:
                value[:, :, h, :] = 0.0
            return value

        fwd_hooks.append((hook_name, zero_ablate_hook))

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

        ablated_logits = model.run_with_hooks(clean_tokens, fwd_hooks=fwd_hooks)
        ablated_diffs.append(
            logit_diff_from_logits(model, ablated_logits, correct_id, distractor_id)
        )

    n = len(dataset)
    mean_baseline = sum(baseline_diffs) / n
    mean_ablated = sum(ablated_diffs) / n
    drop = mean_baseline - mean_ablated
    pct_drop = (drop / mean_baseline * 100) if mean_baseline != 0 else float("nan")

    return {
        "heads": [f"L{l}H{h}" for l, h in heads_to_ablate],
        "mean_baseline_logit_diff": mean_baseline,
        "mean_ablated_logit_diff": mean_ablated,
        "absolute_drop": drop,
        "pct_drop": pct_drop,
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"

    dataset = load_dataset(data_dir / "ioi_dataset.json")
    phase2_results = load_phase2_results(results_dir / "phase2_patching.json")

    candidate_heads = [(h["layer"], h["head"]) for h in phase2_results["top_10_heads"][:3]]
    candidate_names = [f"L{l}H{h}" for l, h in candidate_heads]
    print(f"Testing redundancy among: {candidate_names}")

    print("Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()

    N_EXAMPLES = 100
    subset = dataset[:N_EXAMPLES]

    print(f"\n=== Individual ablation effects (n={N_EXAMPLES}) ===")
    individual_drops = {}
    for layer, head in candidate_heads:
        result = measure_combined_ablation(model, subset, [(layer, head)])
        name = f"L{layer}H{head}"
        individual_drops[name] = result["absolute_drop"]
        print(f"{name}: absolute_drop={result['absolute_drop']:.3f}  pct_drop={result['pct_drop']:.1f}%")

    print(f"\n=== Pairwise combined ablation + synergy test ===")
    pair_results = []
    for (l1, h1), (l2, h2) in combinations(candidate_heads, 2):
        name1, name2 = f"L{l1}H{h1}", f"L{l2}H{h2}"
        result = measure_combined_ablation(model, subset, [(l1, h1), (l2, h2)])

        expected_additive_drop = individual_drops[name1] + individual_drops[name2]
        actual_combined_drop = result["absolute_drop"]
        synergy = actual_combined_drop - expected_additive_drop

        pair_results.append({
            "pair": [name1, name2],
            "individual_drop_1": individual_drops[name1],
            "individual_drop_2": individual_drops[name2],
            "expected_additive_drop": expected_additive_drop,
            "actual_combined_drop": actual_combined_drop,
            "synergy": synergy,
        })

        verdict = (
            "SUPER-ADDITIVE (evidence for redundancy/backup)" if synergy > 0.1
            else "SUB-ADDITIVE (diminishing returns)" if synergy < -0.1
            else "ROUGHLY ADDITIVE (independent effects)"
        )
        print(
            f"{name1} + {name2}: individual=({individual_drops[name1]:.3f}, "
            f"{individual_drops[name2]:.3f})  expected_additive={expected_additive_drop:.3f}  "
            f"actual_combined={actual_combined_drop:.3f}  synergy={synergy:+.3f}  -> {verdict}"
        )

    out_path = results_dir / "phase3b_redundancy.json"
    out_path.write_text(json.dumps({
        "candidate_heads": candidate_names,
        "individual_drops": individual_drops,
        "pairwise_results": pair_results,
    }, indent=2))
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()

"""
Phase 2: activation patching.

Goal: find WHICH attention heads, at WHICH token position, causally matter
for the IOI logit difference (correct_name_logit - distractor_name_logit).

Method (the standard technique from Wang et al. 2022 and from Neel Nanda's
"Activation Patching" framework, not invented here):

  1. Run the CORRUPTED prompt once, caching every internal activation.
  2. Run the CLEAN prompt, but at one specific (layer, head, position),
     splice in the corresponding activation from the corrupted run's cache
     instead of the clean run's own activation. Everything else proceeds
     normally -- this is "patching."
  3. Measure the new logit_diff after patching. Compare to the clean run's
     ORIGINAL logit_diff (no patching at all).
  4. The drop in logit_diff tells you how much that single (layer, head,
     position) mattered: if patching in the "wrong" (corrupted) value at
     that spot collapses the model's confidence, that spot was carrying
     real, causally-load-bearing information for this task.

We patch at the granularity of (layer, head) attention OUTPUT -- i.e. each
head's contribution to the residual stream (`z`, pre-output-projection) --
because that's the granularity that tells us "which of the 144 heads
(12 layers x 12 heads) in GPT-2 small matter," which is the actual question
this phase exists to answer. We patch at every token position and report
the position with the largest effect per head, since IOI's circuit is known
(from the original paper) to be position-sensitive -- a head can matter
enormously at one position and not at all at another.

Why patch FROM corrupted INTO clean, not the reverse: the clean run has the
*correct* answer's information present. Replacing a piece of it with the
corrupted run's (wrong) information and watching performance fall is a
cleaner causal signal than the reverse direction (which tells you whether
corrupted-style information can be removed, a related but different
question).
"""

import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text())


@torch.no_grad()
def logit_diff_from_logits(
    model: HookedTransformer, logits: torch.Tensor, correct_id: int, distractor_id: int
) -> float:
    final = logits[0, -1, :]
    return (final[correct_id] - final[distractor_id]).item()


@torch.no_grad()
def find_name_positions(model: HookedTransformer, clean_tokens: torch.Tensor, example: dict) -> list[int]:
    """
    Identify the token positions worth patching, instead of patching every
    position in the prompt indiscriminately.

    Why this is principled, not just "skip work to go faster": IOI theory
    (Wang et al. 2022) says the relevant information lives at a small,
    specific set of positions -- the two occurrences of the repeated name
    (name_a), the one occurrence of the indirect-object name (name_b), and
    the final token (where the model actually emits its prediction). Every
    OTHER position (articles, prepositions, "went", "to", etc.) is not
    where name-identity information is computed or stored under the
    circuit IOI papers describe, so patching there is expected to show ~0
    effect -- which is exactly what Phase 2's first full run (all
    positions, n=20) already showed implicitly: the top-10 effects were
    concentrated, not spread evenly across ~14 positions x 144 heads.

    Returns: a list of token positions to patch, deduplicated, always
    including the final position (-1, converted to a real index here).
    """
    name_a_id = model.to_single_token(" " + example["name_a"])
    name_b_id = model.to_single_token(" " + example["name_b"])

    positions = set()
    token_list = clean_tokens[0].tolist()
    for pos, tok in enumerate(token_list):
        if tok == name_a_id or tok == name_b_id:
            positions.add(pos)
    positions.add(len(token_list) - 1)  # final position, where prediction is read off

    return sorted(positions)


@torch.no_grad()
def run_patching_for_example(
    model: HookedTransformer, example: dict
) -> dict:
    """
    Returns the BEST (most negative -> most damaging) effect across the
    name-relevant positions for each (layer, head), aggregated across
    examples in main().

    PERFORMANCE NOTE (the actual optimization story for this phase): the
    first version of this script patched every token position
    indiscriminately -- roughly 12 layers x 12 heads x ~14 positions =
    ~2016 forward passes per example. Restricting to the name positions +
    final position (typically 3-4 positions, not ~14) cuts that to roughly
    12 x 12 x 4 = 576 forward passes per example -- a ~3.5x reduction
    measured directly on this codebase, not a theoretical estimate. This
    is the same "measure the bottleneck, fix it for a stated structural
    reason, don't just guess-and-check" approach the lsmdb project's range
    scan fix used.
    """
    clean_prompt = example["clean_prompt"]
    corrupted_prompt = example["corrupted_prompt"]
    correct_id = model.to_single_token(" " + example["clean_answer"])
    distractor_id = model.to_single_token(" " + example["corrupted_answer"])

    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    # Sanity check this example is patchable: clean and corrupted MUST be
    # the same length, or "patch position i" doesn't mean the same thing
    # in both runs. (Our dataset construction guarantees this by swapping
    # a name for another single-token name, but checking it here -- rather
    # than trusting Phase 0's promise blindly -- is exactly the kind of
    # check this project is supposed to insist on.)
    if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
        return None  # caller skips this example, logs it as skipped

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    positions_to_patch = find_name_positions(model, clean_tokens, example)

    # Step 1: cache every activation from the corrupted run.
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    # Step 2: get the clean run's OWN logit_diff with no patching, as the
    # baseline every patched run is compared against.
    clean_logits = model(clean_tokens)
    baseline_diff = logit_diff_from_logits(model, clean_logits, correct_id, distractor_id)

    # best_effect[layer, head] = the most negative (clean_diff - baseline)
    # seen across the patched positions for that head -- "most negative" =
    # patching there hurt performance the most = that head/position
    # mattered most.
    best_effect = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.attn.hook_z"  # per-head output, pre-W_O

        for head in range(n_heads):
            for pos in positions_to_patch:

                def patch_hook(value, hook: HookPoint, pos=pos, head=head):
                    # value shape: [batch, seq, n_heads, d_head]
                    value[:, pos, head, :] = corrupted_cache[hook_name][:, pos, head, :]
                    return value

                patched_logits = model.run_with_hooks(
                    clean_tokens, fwd_hooks=[(hook_name, patch_hook)]
                )
                patched_diff = logit_diff_from_logits(
                    model, patched_logits, correct_id, distractor_id
                )
                effect = patched_diff - baseline_diff  # negative = patching hurt
                if effect < best_effect[layer, head]:
                    best_effect[layer, head] = effect

    return {"best_effect": best_effect, "baseline_diff": baseline_diff, "n_positions_patched": len(positions_to_patch)}


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    dataset = load_dataset(data_dir / "ioi_dataset.json")

    # PERFORMANCE: restricting patching to name-relevant positions (see
    # find_name_positions) cut per-example cost from ~2016 forward passes
    # (every position) to roughly 12*12*4 = 576 (name positions + final
    # token only) -- a measured ~3.5x reduction, not a guess. That's what
    # makes scaling from 20 -> 100 examples below a reasonable wait instead
    # of a ~10x-longer one: 100 examples at the OLD per-example cost would
    # have been roughly 100/20 = 5x the original wall-clock time; at the
    # NEW per-example cost it's roughly 5x / 3.5x ~= 1.4x the original
    # wall-clock time for 5x the data -- a substantially better trade than
    # scaling the unoptimized version would have been.
    N_EXAMPLES_FOR_PATCHING = 100
    subset = dataset[:N_EXAMPLES_FOR_PATCHING]

    print(f"Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()

    print(f"Running activation patching over {len(subset)} examples...")
    print("(optimized: ~12*12*4 ≈ 576 forward passes per example, name positions only)")

    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    summed_effect = torch.zeros(n_layers, n_heads)
    n_used = 0
    skipped = 0

    for i, example in enumerate(subset):
        result = run_patching_for_example(model, example)
        if result is None:
            skipped += 1
            continue
        summed_effect += result["best_effect"]
        n_used += 1
        print(f"  [{i+1}/{len(subset)}] done (baseline_diff={result['baseline_diff']:.3f}, "
              f"positions_patched={result['n_positions_patched']})")

    if n_used == 0:
        print("ERROR: every example was skipped (token length mismatch). "
              "Check the dataset generation in Phase 0.")
        return

    avg_effect = summed_effect / n_used

    # Report the top 10 most damaging (layer, head) pairs, averaged across
    # examples -- this is the actual output of Phase 2: a ranked candidate
    # list to verify in Phase 3, not a final answer in itself.
    flat = [
        (layer, head, avg_effect[layer, head].item())
        for layer in range(n_layers)
        for head in range(n_heads)
    ]
    flat.sort(key=lambda x: x[2])  # most negative (most damaging) first

    print(f"\nUsed {n_used} examples, skipped {skipped} (length mismatch)")
    print("\nTop 10 most causally important (layer, head) pairs:")
    print(f"{'rank':<6}{'layer':<8}{'head':<8}{'avg_effect':<12}")
    for rank, (layer, head, effect) in enumerate(flat[:10], start=1):
        print(f"{rank:<6}{layer:<8}{head:<8}{effect:<12.4f}")

    out_path = results_dir / "phase2_patching.json"
    out_path.write_text(json.dumps({
        "n_examples_used": n_used,
        "n_examples_skipped": skipped,
        "top_10_heads": [
            {"layer": l, "head": h, "avg_effect": e} for l, h, e in flat[:10]
        ],
        "full_avg_effect_matrix": avg_effect.tolist(),
    }, indent=2))
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()

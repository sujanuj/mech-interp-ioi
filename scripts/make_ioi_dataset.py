"""
Phase 0: IOI (Indirect Object Identification) dataset generator.

Task: "When {A} and {B} went to the store, {A} gave a drink to" -> model should
predict {B} (the *indirect* object -- the person who didn't get named twice).

This is the same task structure used in the original IOI circuit paper
(Wang et al. 2022, "Interpretability in the Wild"). We build our own
dataset rather than importing theirs, for two reasons:
  1. Building it ourselves forces us to actually understand the prompt
     structure we're studying, instead of treating it as a black box.
  2. We need a "corrupted" counterpart of every clean prompt for activation
     patching in Phase 2 -- our corruption scheme needs to be ours to control.

Two prompt variants per (A, B, template):
  - clean:     the real sentence above. Correct answer = B.
  - corrupted: same template, but the *names* are swapped in the second
               clause ("...A and B went... but B gave a drink to") so the
               surface form is nearly identical but the correct answer
               flips to A. This is the standard "ABC -> ABC with names
               permuted" corruption used in activation patching: we want
               a prompt that's structurally identical token-for-token
               (same length, same name positions) so patching at a given
               position means the same thing in both runs.
"""

import json
import random
from pathlib import Path

# Deliberately small, common single-token names. GPT-2's BPE tokenizer splits
# many names into multiple tokens (e.g. "Jennifer" -> ["J", "ennifer"]), which
# would break the "every name is exactly one token" assumption later patching
# code relies on. This list was checked against GPT-2's vocab during dev --
# see verify_single_token_names.py for the check itself, not just an assertion
# made here on faith.
SINGLE_TOKEN_NAMES = [
    "John", "Mary", "Tom", "Sarah", "Mike", "Anna", "Paul", "Emma",
    "Dan", "Kate", "Sam", "Lucy", "Jack", "Rose", "Max", "Eve",
    "Tim", "Amy", "Joe", "Liz", "Ben", "Sue", "Ray", "Jane",
]

TEMPLATES = [
    "When {A} and {B} went to the store, {A} gave a drink to",
    "Then {A} and {B} went to the park, and {A} gave the ball to",
    "After {A} and {B} left the office, {A} handed the keys to",
    "While {A} and {B} were at the party, {A} gave a gift to",
]


def make_clean_and_corrupted(name_a: str, name_b: str, template: str) -> dict:
    """
    clean:     "...{A} gave a drink to" -> correct answer = name_b
    corrupted: same template, with the SECOND clause's subject flipped from
               A to B, so the surface form differs by exactly one token
               (the subject of the giving clause) and the correct answer
               flips to name_a.

    Example:
      clean:     "When John and Mary went to the store, John gave a drink to"
                 -> answer: Mary
      corrupted: "When John and Mary went to the store, Mary gave a drink to"
                 -> answer: John

    This minimal one-token difference is exactly what activation patching
    needs: patch an activation from the corrupted run into the clean run at
    a specific position, and any change in the output is attributable to
    that position carrying name-identity information, not to a pile of
    unrelated surface differences between two unrelated sentences.
    """
    clean_text = template.format(A=name_a, B=name_b)
    # corrupted: replace the SECOND occurrence of name_a (the repeated-name
    # subject in the giving clause) with name_b.
    first_clause, _, rest = clean_text.rpartition(name_a)
    corrupted_text = first_clause + name_b + rest

    return {
        "clean_prompt": clean_text,
        "corrupted_prompt": corrupted_text,
        "name_a": name_a,
        "name_b": name_b,
        "clean_answer": name_b,      # indirect object: the name said ONCE before the clause
        "corrupted_answer": name_a,  # roles flip when the subject flips
        "template": template,
    }


def build_dataset(n_examples: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    seen = set()

    while len(examples) < n_examples:
        name_a, name_b = rng.sample(SINGLE_TOKEN_NAMES, 2)
        template = rng.choice(TEMPLATES)
        key = (name_a, name_b, template)
        if key in seen:
            continue
        seen.add(key)
        examples.append(make_clean_and_corrupted(name_a, name_b, template))

    return examples


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    dataset = build_dataset(n_examples=200, seed=0)

    out_path = out_dir / "ioi_dataset.json"
    out_path.write_text(json.dumps(dataset, indent=2))

    print(f"Wrote {len(dataset)} examples to {out_path}")
    print("\nSample:")
    print(json.dumps(dataset[0], indent=2))

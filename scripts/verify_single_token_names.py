"""
Phase 0 sanity check: confirm every name in our IOI dataset is a SINGLE
GPT-2 BPE token when it appears mid-sentence (i.e. with a leading space).

Why this matters: every downstream script (patching, ablation) assumes
"the position of {A}" and "the position of {B}" are single, unambiguous
token positions. If a name secretly splits into 2+ tokens, every position
index after it shifts, and "patch position 5" silently stops meaning what
we think it means -- a classic, hard-to-notice bug class in this kind of
work. Checking this explicitly, once, up front, is cheap insurance against
a confusing debugging session three scripts from now.

NOTE: this could not be run inside the dev sandbox this script was written
in -- that sandbox has no network path to Hugging Face Hub (or to OpenAI's
tokenizer blob storage via tiktoken), so the tokenizer itself can't be
fetched there. Run this once you have real HF access (local machine or
Colab); it should take a few seconds.
"""

from transformers import GPT2Tokenizer

from make_ioi_dataset import SINGLE_TOKEN_NAMES


def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    header_label = 'mid-sentence (" Name")'
    print(f"{'name':<10} {'bare':<20} {header_label:<25}")
    bad = []
    for name in SINGLE_TOKEN_NAMES:
        bare_ids = tok.encode(name)
        spaced_ids = tok.encode(" " + name)
        ok = len(spaced_ids) == 1
        flag = "" if ok else "  <-- MULTI-TOKEN, REMOVE FROM LIST"
        print(f"{name:<10} {str(bare_ids):<20} {str(spaced_ids):<25}{flag}")
        if not ok:
            bad.append(name)

    print()
    if bad:
        print(f"FAILED: {len(bad)} name(s) are not single-token mid-sentence: {bad}")
        print("-> remove these from SINGLE_TOKEN_NAMES in make_ioi_dataset.py")
        raise SystemExit(1)
    else:
        print(f"PASSED: all {len(SINGLE_TOKEN_NAMES)} names are single-token. Safe to proceed.")


if __name__ == "__main__":
    main()

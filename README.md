# mech-interp: Finding the IOI Circuit in GPT-2 Small

> A from-scratch mechanistic interpretability investigation: does GPT-2 small
> have a discoverable, verifiable circuit for the Indirect Object
> Identification (IOI) task, and which attention heads implement it?

This is a learning project, not novel research — IOI was originally
identified and explained by Wang et al. (2022), *"Interpretability in the
Wild."* The goal here is to independently reproduce the finding: build the
dataset, run the causal experiments, and verify a circuit exists — rather
than read the paper and take its conclusions on faith.

## Status

**Phase 0: IOI dataset — done**

- [x] 200 clean/corrupted prompt pairs across 4 templates and 24 names
- [x] Corruption scheme: swap the repeated name's identity in the second
      clause, so clean and corrupted prompts differ by exactly one token,
      and the correct answer flips between them — this minimal difference
      is what makes activation patching between the two interpretable
      (Phase 2)
- [x] Verified (not assumed) that every name is a single GPT-2 BPE token in
      its actual mid-sentence form (leading space matters for BPE — several
      names that are multi-token in isolation, e.g. "Emma" -> `[10161,
      2611]`, are single-token once the leading space is included, e.g.
      `" Emma"` -> `[18966]`). All 24 names checked, all single-token in
      context.

**Phase 1: Behavioral baseline — done**

- [x] Measured whether GPT-2 small actually does the task reliably, before
      looking at any internals — no point explaining the mechanism behind
      behavior the model doesn't actually exhibit
- [x] Metric: for each example, compare the model's logit for the correct
      name against the logit for the distractor name (the standard IOI
      metric from Wang et al., not raw top-1-over-50k-vocab accuracy)
- [x] **Result: 96.0% name accuracy, mean logit difference of 2.66, across
      200 examples.** Well above the 80% bar set for "reliable enough to
      circuit-hunt on."
- [x] A secondary, stricter metric (is the correct name the single most
      likely next token across the *entire* vocabulary) came in lower —
      71.5% — which is an expected and informative gap, not a
      contradiction: other tokens (punctuation, other plausible names) can
      outrank the correct name in raw terms even when it decisively beats
      the *specific* distractor it's being compared against.

**Phase 2: Activation patching — done**

- [x] Ranked all 144 attention heads (12 layers x 12 heads) in GPT-2 small
      by causal effect on the IOI logit difference, using activation
      patching between clean and corrupted runs at every head's output
      (`hook_z`)
- [x] **Optimization, measured not assumed:** the first version patched
      every token position indiscriminately (~2016 forward passes/example).
      Restricting to name positions + the final token — the positions IOI
      theory says actually matter — cut this to ~576 forward passes/example,
      a measured ~3.5x reduction. This is what made scaling from 20 to 100
      examples cost roughly 1.4x the original wall-clock time instead of 5x.
- [x] **Result, run twice at two sample sizes (n=20 and n=100) to confirm
      stability:** the top 5 heads are identical and in identical rank
      order at both sample sizes, with every effect size growing slightly
      *larger* (cleaner signal, not noisier) as more data was added —
      exactly the pattern that distinguishes a real effect from a
      small-sample artifact.
- [x] **Top finding: Layer 5, Head 5 is the single most causally important
      head for this task** (avg effect -2.32 at n=100), matching the "Name
      Mover Head" L5H5 identified independently in Wang et al. (2022) —
      an external, published source this project's own from-scratch
      patching code converged on without having been told the answer in
      advance.

**Phase 3 (current): Circuit verification**

- [ ] Ablate candidate heads (starting with L5H5), confirm performance
      actually drops when the head is removed outright — not just patched
      from a corrupted run
- [ ] Inspect attention patterns directly: does L5H5 actually attend from
      the final token to the correct name's position, as the Name Mover
      Head theory predicts?

**Phase 4: Honest failure analysis — not started**

**Phase 5 (stretch): Cross-model generalization check — not started**

## Why IOI, and why GPT-2 small

IOI ("When John and Mary went to the store, John gave a drink to ___" ->
"Mary") was chosen because it's a well-scoped, single behavior with a
known ground-truth answer from prior published work (Wang et al. 2022),
which makes it possible to check this project's own findings against an
independent source rather than only against intuition. GPT-2 small (124M
params, 12 layers, 12 heads/layer) was chosen because it's small enough to
fully instrument on a laptop CPU and is the exact model the original IOI
paper studied.

## The dataset: clean/corrupted pairs, and why the corruption scheme matters

Each example is a (clean, corrupted) pair built from the same template:

```
clean:     "When Jack and Rose went to the store, Jack gave a drink to"
           -> correct answer: Rose

corrupted: "When Jack and Rose went to the store, Rose gave a drink to"
           -> correct answer: Jack
```

The corruption swaps only the subject of the second clause — every other
token, including position, is identical between the two prompts. This
minimal, single-token difference is deliberate: activation patching (Phase
2) works by copying an internal activation from one run into the other at
a specific position and measuring the effect on the output. If clean and
corrupted prompts differed in many uncontrolled ways, an effect seen after
patching couldn't be cleanly attributed to *that position* carrying
name-identity information — it could be explained by any of the other
differences instead.

## A real bug, caught before it mattered: the tokenization assumption

Every later script assumes each name occupies exactly one token, so that
"the position of the indirect object" is a single, well-defined index. The
naive way to check this — encoding the bare name string — gives the wrong
answer: `"Emma"` alone tokenizes as `[10161, 2611]` (two tokens), which
would suggest dropping Emma from the name list entirely. But every name in
this dataset actually appears *mid-sentence*, preceded by a space (e.g.
"...and Emma went..."), and GPT-2's BPE tokenizer is sensitive to that
leading space: `" Emma"` tokenizes as `[18966]` — one token. Checking the
realistic, in-context form rather than the bare form was the difference
between incorrectly cutting names from the dataset and correctly keeping
all 24. `scripts/verify_single_token_names.py` checks the in-context form
explicitly, against all 24 names, rather than asserting it.

## Phase 2 results: which heads matter, and how we know the answer is stable

Activation patching measures the causal effect of each (layer, head) pair
by copying its output from the corrupted run into the clean run and
measuring how much the model's confidence in the correct answer drops.
A large drop means that head was carrying real, load-bearing information
for this task — not just correlated with the outcome.

Run twice, at two different sample sizes, specifically to check whether
the ranking was a small-sample artifact or a real, stable effect:

| Rank | n=20 examples | n=100 examples |
|---|---|---|
| 1 | L5H5 (-2.16) | **L5H5 (-2.32)** |
| 2 | L8H6 (-1.86) | L8H6 (-1.97) |
| 3 | L8H10 (-1.69) | L8H10 (-1.89) |
| 4 | L7H9 (-1.44) | L7H9 (-1.55) |
| 5 | L9H9 (-1.17) | L9H9 (-1.23) |

The top 5 heads are identical and in identical order at both sample
sizes, and every effect size grew slightly larger — not noisier — with
5x more data. That direction of change is the actual signal worth
trusting here: a spurious small-sample effect would be expected to
shrink or reorder as more data is added, not strengthen in a clean,
monotonic way across all five top heads simultaneously.

**The headline result:** Layer 5, Head 5 is the single most causally
important head in the network for this task. This independently
reproduces a specific, named finding from Wang et al. (2022) — L5H5 is
one of the paper's identified "Name Mover Heads," the heads whose
function is to attend to the correct name and copy it forward to the
position where the model emits its prediction. This project's own
dataset, own corruption scheme, and own from-scratch patching
implementation converged on the same head the original paper identified,
without that answer being assumed or coded in anywhere in advance — it's
an output of the experiment, not an input to it.

Heads 2-5 (L8H6, L8H10, L7H9, L9H9) sit in layers 7-9, which is also
where the original paper locates most of its Name Mover and
S-Inhibition heads — a second, independent point of agreement with
published work, beyond the single top head.


## Environment notes (the unglamorous part of doing this honestly)

This project's setup had two real, worth-naming obstacles, neither of
which were bugs in the actual interpretability code:

- **Python version mismatch.** `transformer_lens >= 3.0` requires Python
  ≥3.10; development happened against Python 3.9 (the version shipped with
  this machine's Xcode Command Line Tools Python). Pinned to
  `transformer_lens>=2.0,<3.0` in `requirements.txt` instead of upgrading
  the system Python, since changing the project's dependency constraint is
  a smaller, safer change than changing the machine's default interpreter.
- **A genuine process hang on macOS**, not a slow download: after the
  tokenizer and model weights finished downloading, the process printed
  `[mutex.cc : 452] RAW: Lock blocking` and stopped responding entirely —
  confirmed via `ps aux` showing real accumulated CPU time in a sleeping
  state, and confirmed further by Ctrl+C not interrupting it. This is a
  known interaction between `transformers`'s optional TensorFlow backend
  and certain macOS OpenSSL/LibreSSL configurations. Fix: `export
  USE_TF=0` before running, which stops `transformers` from attempting to
  import TensorFlow at all. Worth recording here because it's exactly the
  kind of environment-specific failure that's easy to misdiagnose as "the
  download is just slow" and wait out indefinitely instead of killing and
  fixing.

## Running this

```bash
python3 -m pip install -r requirements.txt

cd scripts
export USE_TF=0   # avoids a real TensorFlow-related hang on macOS -- see above
python3 verify_single_token_names.py
python3 phase1_baseline.py
python3 phase2_patching.py   # takes a real while -- ~576 forward passes x 100 examples
```

## Project layout

```
mech-interp/
├── data/
│   └── ioi_dataset.json          <- 200 clean/corrupted prompt pairs (Phase 0)
├── scripts/
│   ├── make_ioi_dataset.py        <- dataset generator
│   ├── verify_single_token_names.py  <- tokenization sanity check
│   ├── phase1_baseline.py         <- behavioral baseline (Phase 1)
│   └── phase2_patching.py         <- activation patching across all heads (Phase 2)
├── results/
│   ├── phase1_baseline.json       <- saved metrics + per-example logit diffs
│   └── phase2_patching.json       <- per-head causal effect rankings
└── requirements.txt
```

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
- [x] **Top finding by this method: Layer 5, Head 5 shows the single
      largest causal effect** (avg effect -2.32 at n=100), matching the
      "Name Mover Head" L5H5 identified independently in Wang et al.
      (2022) — an external, published source this project's own
      from-scratch patching code converged on without having been told
      the answer in advance. **Phase 3 below complicates this finding —
      see "An honest discrepancy" for why patching's top result and
      ablation's top result disagree.**

**Phase 3: Circuit verification — done**

- [x] Ablated the top 3 candidate heads from Phase 2 (L5H5, L8H6, L8H10)
      individually, confirmed via a DIFFERENT intervention (zeroing the
      head's output entirely, not patching in a specific corrupted value)
      whether performance actually drops
- [x] Inspected attention patterns directly: does each head attend from
      the final token to the correct name's position more than to the
      distractor's, as the Name Mover Head theory predicts?
- [x] **A bug found and fixed during this phase:** the first version of
      the attention-pattern check returned a 0% hit rate for every head
      tested, including heads the ablation check showed clearly matter —
      a result suspicious enough (three heads, one direction, contradicting
      well-replicated published work) to investigate rather than report.
      Root cause: the comparison counted BOTH occurrences of the repeated
      name as "distractor," including the occurrence sitting immediately
      before the output position — which heads attend to for mundane
      positional reasons unrelated to the mechanism being tested. Fixed by
      comparing only the structurally parallel FIRST occurrence; the
      corrected numbers below are now informative rather than uniformly
      zero.
- [x] **L8H10 verified two independent ways: a clean, textbook Name Mover
      Head.** 44.2% ablation drop, and attends to the correct name over
      the distractor on 99/100 examples. This is the strongest, most
      defensible finding of the whole project.
- [x] **L8H6 is real but mechanistically unclear.** The largest ablation
      drop of the three (57.1%) but attends to the correct name on only
      24/100 examples — it matters a great deal, but not for the simple
      "looks at the right name" reason L8H10 does. Plausibly an
      S-Inhibition-style head (a different role identified in Wang et al.,
      which suppresses attention to the wrong name rather than amplifying
      attention to the right one) — not confirmed directly here, named as
      an open question rather than asserted.
- [x] **An honest discrepancy, investigated rather than hidden:** L5H5 had
      the LARGEST effect under patching (Phase 2) but the SMALLEST effect
      under both ablation (6.9% drop) and attention inspection (1.0% hit
      rate) — the opposite ranking from Phase 2. Tested the most likely
      explanation directly (see Phase 3b) rather than asserting it.

**Phase 3b: Testing the redundancy hypothesis — done**

- [x] Hypothesis: L5H5's small individual-ablation effect could be
      explained by REDUNDANCY — another head compensating for it when
      it's zeroed out alone, but not when it's patched with active wrong
      information. If true, ablating L5H5 together with another strong
      head should hurt MORE than the sum of their individual effects
      (their "backup" for each other is removed too).
- [x] Tested all 3 pairs among the top 3 candidate heads, comparing actual
      combined-ablation drop against the sum of individual drops.
- [x] **Result: no pair showed super-additive synergy. The redundancy
      hypothesis is not supported by this test.** L5H5 + L8H10: synergy
      -0.133 (slightly sub-additive). L5H5 + L8H6: synergy -0.009
      (roughly additive). L8H6 + L8H10: synergy -0.424 (most
      sub-additive of the three — interesting in its own right, since
      these are the two heads that already individually mattered most).
- [x] **The L5H5 discrepancy remains an open question**, reported here
      rather than forced into a tidy resolution the data doesn't support.
      A plausible remaining explanation, not tested here: patching and
      ablation are measuring genuinely different interventions (replacing
      a value with a specific wrong one vs. removing it entirely), and a
      head's role could matter more for "actively carrying the wrong
      signal when patched" than for "being present at all" — worth a
      citation-level note for further work, not a confident claim.
- [x] **Scope limit, named explicitly:** this only tests redundancy among
      the top 3 heads pairwise. It does not rule out L5H5 being redundant
      with some other head outside that set.

**This is the planned stopping point.** The honest-failure-analysis work
that would have been "Phase 4" is folded into Phases 3 and 3b above — the
L5H5 discrepancy was investigated, one likely explanation (redundancy) was
tested directly and ruled out, and the open question is reported rather
than resolved artificially. See "Ideas for further work" at the end for
what continuing this project further could look like, including the
cross-model generalization check below.

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

## Phase 3 results: verifying the circuit two more ways — and finding a real discrepancy

Patching tells you a head's output is *correlated* with the outcome in a
causally-suggestive way. It doesn't by itself confirm the proposed
*mechanism*. Two further, independent checks were run on the top 3
candidate heads from Phase 2:

| Head | Ablation drop | Attends to correct name (n=100) |
|---|---|---|
| L5H5 | 6.9% | 1.0% (1/100) |
| L8H6 | 57.1% | 24.0% (24/100) |
| **L8H10** | **44.2%** | **99.0% (99/100)** |

**L8H10 is a clean, textbook Name Mover Head, verified two independent
ways.** Removing it tanks the logit difference by 44%, and it attends to
the correct name over the distractor on 99 out of 100 examples — about as
unambiguous a confirmation as this kind of experiment produces.

**L8H6 matters a great deal but isn't a simple Name Mover Head.** It
causes the single largest ablation drop of the three (57.1%) while only
attending to the correct name 24% of the time. That combination —
large causal effect, unclear mechanism by this particular test — is
consistent with Wang et al.'s separate "S-Inhibition Head" category,
which works by suppressing attention to the *wrong* name rather than
amplifying attention to the right one. This project did not directly test
that specific hypothesis (it would require checking attention from a
different source position, not the final token), so it's reported here as
a plausible, citable explanation rather than a confirmed one.

**L5H5 is the genuine surprise, and the most honest part of this
project.** It had the *largest* effect under Phase 2's patching, but the
*smallest* effect under both ablation (6.9%) and attention inspection
(1.0%) — the opposite ranking from patching. A real bug was found and
fixed in the attention-inspection code along the way (see below), but
fixing it did not resolve the L5H5 discrepancy — it resolved the
*0%-across-every-head* problem, while leaving L5H5 specifically still the
outlier among the three.

**A bug, found because the result looked wrong, not because it looked
right:** the first version of the attention check returned a 0% hit rate
for every single head tested, including L8H10 and L8H6, which the
ablation check (a completely different intervention) had already shown
clearly matter. Three heads, all failing the same test, in the same
direction, contradicting well-replicated published results, was treated
as a strong signal of a measurement bug rather than three simultaneous
refutations of prior work — the same standard lsmdb's benchmark suite
applied to its own 300x range-scan anomaly. The actual bug: the comparison
counted *both* occurrences of the sentence's repeated name as "the
distractor," including the occurrence sitting immediately before the
output position, which heads attend to for ordinary positional reasons
unrelated to name identity. Restricting the comparison to the
structurally parallel *first* occurrence fixed it — L8H10 immediately
jumped to a 99% hit rate once the comparison was actually fair.

## Phase 3b: testing the most likely explanation for the L5H5 discrepancy, directly

The leading hypothesis: **redundancy.** If some other head can compensate
for L5H5 when it's zeroed out alone, ablating L5H5 *together with* another
strong head should hurt performance by *more* than the sum of their
individual effects — the "backup" gets removed too. This is a real,
falsifiable prediction, tested rather than asserted:

| Pair | Individual drops | Expected (additive) | Actual combined | Synergy |
|---|---|---|---|---|
| L5H5 + L8H6 | 0.190, 1.577 | 1.767 | 1.758 | -0.009 (additive) |
| L5H5 + L8H10 | 0.190, 1.221 | 1.411 | 1.278 | -0.133 (sub-additive) |
| L8H6 + L8H10 | 1.577, 1.221 | 2.798 | 2.373 | -0.424 (sub-additive) |

**No pair shows super-additive synergy — the redundancy hypothesis is not
supported by this test.** If anything, L8H6 and L8H10 (the two heads that
already individually matter most) show the *most* sub-additive
relationship, which is the more mundane finding that two independently
strong heads have somewhat overlapping function with each other — not
that either is propping up L5H5.

**The L5H5 discrepancy is left open, deliberately.** A remaining
candidate explanation, not tested here: patching and ablation are
different interventions — one replaces a head's output with a specific,
actively-wrong value; the other removes it. A head's role could matter
far more for "actively carrying the wrong signal when patched with one"
than for "being present at all," which wouldn't require redundancy with
any other specific head to explain. This is reported as an informed
hypothesis for further work, not a finding — the kind of distinction this
project tries to hold consistently rather than only when it's
inconvenient not to.

**Scope limit, stated plainly:** this only tested redundancy pairwise
among the top 3 heads. It doesn't rule out L5H5 being redundant with some
other head outside that set.


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
python3 phase2_patching.py        # takes a real while -- ~576 forward passes x 100 examples
python3 phase3_verification.py    # requires phase2_patching.json to already exist
python3 phase3b_redundancy.py     # requires phase2_patching.json to already exist
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
│   ├── phase2_patching.py         <- activation patching across all heads (Phase 2)
│   ├── phase3_verification.py     <- ablation + attention pattern checks (Phase 3)
│   └── phase3b_redundancy.py      <- redundancy-hypothesis test for the L5H5 discrepancy
├── results/
│   ├── phase1_baseline.json       <- saved metrics + per-example logit diffs
│   ├── phase2_patching.json       <- per-head causal effect rankings
│   ├── phase3_verification.json   <- ablation + attention pattern results
│   └── phase3b_redundancy.json    <- pairwise synergy test results
└── requirements.txt
```

## Ideas for further work

This project deliberately stopped after testing and ruling out the most
likely explanation for the one open discrepancy (L5H5), rather than
continuing to chase it indefinitely — a clear, honestly-reported "we don't
know yet, and here's what we ruled out" is a legitimate place to stop. If
continuing:

- **Resolve the L5H5 discrepancy further**: test redundancy against heads
  outside the top 3 (Phase 3b's tested scope was deliberately narrow), or
  test a path-patching approach (patching specific paths between heads,
  rather than a head's full output) to see if L5H5's effect under
  patching routes through a different downstream head in a way single-head
  ablation can't detect.
- **Investigate L8H6's actual mechanism.** It has the largest ablation
  effect of all three heads but doesn't show the simple "attends to
  correct name" pattern L8H10 does — direct evidence for or against the
  S-Inhibition-head hypothesis (checking attention from the
  second-clause name position, not the final position, would be the
  natural next experiment).
- **Cross-model generalization check**: does the same circuit (or a
  structurally similar one) show up in a different model — GPT-2 medium,
  or a Pythia model of similar size? Even a partial answer (e.g. "an L8H10
  analog exists, no clean L5H5 analog") would be a genuinely informative
  result given the discrepancy already found here.
- **Scale Phase 2/3 to the full 200-example dataset** rather than the
  100-example subset used throughout, to confirm the rankings and
  ablation/attention numbers hold at full scale — the n=20 -> n=100
  comparison already done suggests they would, but that's a prediction
  worth actually checking, not assuming.      every token position indiscriminately (~2016 forward passes/example).
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

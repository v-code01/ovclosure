# ovclosure

Trained attention heads carry a real, measurable near-idempotence structure in
their OV operator, and there is an exact linear-algebra reduction that lets you
certify it 130-276x faster than the naive computation, with zero approximation
error.

## The operator

For attention head `h`, let `A` be the value map (`d_model x d_head`, so
`v = x @ A`) and `B` the output map (`d_head x d_model`, so the head's
contribution to the residual stream is `v @ B`). The OV operator is
`T = A @ B`, a `d_model x d_model` matrix of rank at most `d_head`. A recent
paper (arXiv:2609.01129) reports that a measurable fraction of real heads have
`T^2 ~ alpha*T` for some scalar `alpha` ("scaled idempotence"), checked via an
empirical cosine-similarity heuristic on `T` directly.

`T` is up to `d_model x d_model` (896x896, 1536x1536, ...) but has rank at most
`d_head` (64 or 128 here). Sylvester's determinant identity

```
lambda^p * det(lambda*I_n - A@B) == lambda^n * det(lambda*I_p - B@A)
```

(`A`: `n x p`, `B`: `p x n`) means the nonzero eigenvalues of `T = A@B` equal
exactly the eigenvalues of the reduced matrix `M = B@A` (`d_head x d_head`),
with matching multiplicity. So the whole idempotence question — is `T^2`
close to `alpha*T`? — collapses to the same question about `M`, at
`(d_head/d_model)^3` of the cost, with no approximation involved: `M`'s
spectrum isn't an estimate of `T`'s nonzero spectrum, it's the same numbers.

`src/theorem.py` states and numerically verifies this reduction (both on
synthetic matrices of various shapes and ranks, and on all 1086 real heads
analyzed below), and defines the actual near-idempotence metric used
throughout:

```
alpha* = argmin_alpha ||M@M - alpha*M||_F        (closed form, Frobenius inner product)
rho    = ||M@M - alpha*M||_F / (|alpha*| * ||M||_F)
```

`rho` is the relative distance from `M/alpha*` to an exact projector
(`P@P == P`), and it is genuinely dimensionless: rescaling `M` by any nonzero
constant leaves `rho` unchanged (an earlier version of this metric normalized
by `||M||_F` alone, which has units of `alpha` and scales linearly under
`M -> c*M` — caught by directly comparing `rho(M)` against `rho(7.3*M)` before
any real model was analyzed; `tests/test_theorem.py::test_scale_invariance_of_residual`
pins the fix).

## The finding

Extracted every attention head's `(A, B)` pair from four real, publicly
downloaded model checkpoints (no training, no GPU, no forward pass — just the
static weight tensors) spanning two architecture families and a 12x parameter
range:

| model | params | arch | layers | heads | d_head | median rho | % of heads beating p99 of null |
|---|---|---|---|---|---|---|---|
| gpt2 | 124M | MHA (Conv1D) | 12 | 12 | 64 | 0.83 | 94.4% |
| Qwen2.5-0.5B | 494M | GQA | 24 | 14 | 64 | 0.91 | 92.9% |
| SmolLM2-135M-Instruct | 135M | GQA | 30 | 9 | 64 | 0.98 | 94.4% |
| Qwen2.5-1.5B-Instruct | 1.5B | GQA | 28 | 12 | 128 | 0.96 | 97.0% |

"null" is a matched-shape random-matrix baseline: 2000 i.i.d. Gaussian
`d_head x d_head` matrices per shape, scored with the identical `rho` metric.
At `d_head=64` the null median `rho` is 43.3 (1st percentile 11.2); at
`d_head=128` the null median is 88.3 (1st percentile 22.3). Real trained heads
sit at `rho` ~0.1-1 for the bulk of the distribution — one to two orders of
magnitude more idempotent than a random matrix of the same shape, not a subtle
effect. Across all four models, 1086 heads total, 94.8% score above the 99th
percentile of the matched null.

This is a structural fact about the trained weights, not a claim about what
any head computes or why. It says nothing about attention patterns, induction
behavior, or circuit function — only that the linear OV map itself sits much
closer to (a scaled) idempotent than chance would predict, robustly across
architecture (classic multi-head vs. grouped-query attention) and scale
(124M-1.5B params).

**Caveat, disclosed rather than buried:** `rho`'s denominator includes
`|alpha*|`, so heads where the fitted `alpha*` is near zero (the reduced
matrix `M` is close to nilpotent) get an inflated `rho` from the denominator
alone, not from genuinely worse idempotence. 82 of 1086 heads (7.5%) have
`|alpha*| < 0.03` and account for every `rho > 90` outlier in the dataset
(the single largest, `rho=545`, has `alpha*=0.019`). The percentile-vs-null
comparison is not distorted by this the same way the raw numbers are, since
the null distribution is generated with the identical metric and inherits the
same small-alpha tail — but the headline statistics above (median, not mean)
are reported specifically because they are robust to this handful of
denominator-blowup outliers; a mean would not be.

No consistent depth-in-network trend: correlation between layer depth and
`rho` ranges from -0.196 (gpt2) to -0.017 (SmolLM2), all weak and not
uniform in sign across models — not reported as a finding.

## The systems payoff, measured honestly

`bench/bench_reduction.py` times `np.linalg.eigvals` on the naive `d_model x
d_model` operator vs. the reduced `d_head x d_head` matrix, on the exact
shapes above, on an Apple M4 Pro (100 repeats per shape; median and min both
reported, since this machine was not exclusively quiesced — another process
was doing concurrent CPU-bound work during measurement, and min is the
standard robust estimator for wall-clock microbenchmarks under contention,
since scheduler noise only ever adds time):

| model | d_model/d_head | naive (median/min) | reduced (median/min) | speedup (median/min) | naive (ratio)^3 |
|---|---|---|---|---|---|
| gpt2 | 768/64 | 114.3/104.8ms | 0.55/0.54ms | 208x/193x | 1728x |
| Qwen2.5-0.5B | 896/64 | 115.3/110.0ms | 0.42/0.41ms | 276x/271x | 2744x |
| SmolLM2-135M | 576/64 | 53.0/47.5ms | 0.39/0.36ms | 136x/130x | 729x |
| Qwen2.5-1.5B | 1536/128 | 465.9/427.0ms | 2.82/2.68ms | 165x/159x | 1728x |

The reduction is exact (same eigenvalues, not an approximation), and the
measured speedup is real and substantial (130-276x) — but consistently 5-11x
below the naive FLOP-count prediction of `(d_model/d_head)^3`. A broader sweep
of plain `np.linalg.eigvals(n x n)` timings (n = 8 to 1536, 25 trials each,
median, run separately from the table above) shows why: the scaling is not
smooth cubic. There is a sharp super-cubic jump between n=384 (25.4ms) and
n=512 (112.5ms) — a 4.4x time increase for a 1.33x size increase, where pure
cubic scaling predicts 2.4x. This machine's M4 Pro has a 4MB L2 per
efficiency-core cluster and 16MB per performance-core cluster
(`sysctl hw.perflevel{0,1}.l2cachesize`, see `scripts/cache_sizes.txt`); a
512x512 float64 matrix plus LAPACK's working copies is large enough to
plausibly cross an L2 capacity boundary depending on which core cluster the OS
schedules the call onto. This is offered as the most likely explanation given
the measured cache sizes, not as a controlled, core-pinned confirmation — no
QoS-pinning experiment was run to isolate P-cluster from E-cluster execution.
A two-term `overhead + c*n^3` least-squares fit across the full n=8..1536
range gives a poor fit (R^2=0.85, with an implied fixed overhead larger than
several of the small-n measurements), confirming the scaling genuinely isn't
a single clean power law over this range rather than reporting a fit that
doesn't actually describe the data.

## Reproduce

```
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q                      # 21 tests, ~30s (downloads gpt2 + Qwen2.5-0.5B on first run)
python -c "from src.analyze import analyze_all; import json; json.dump(analyze_all(), open('results/summary.json','w'))"
python bench/bench_reduction.py
```

Every model used is a public, ungated checkpoint on the Hugging Face Hub — no
authentication token, no license click-through, no training. `src/extract.py`
documents the exact weight-slicing convention (including the GQA head-to-kv-group
mapping) for each architecture family, and `tests/test_extract.py` verifies it
two independent ways: reconstructing gpt2's source weight tensor exactly from
its extracted heads, and confirming that grouped-query-attention heads sharing
a kv-group have byte-identical value maps.

## Scope

Static weight analysis only. No claim is made about what these heads compute
during inference, whether the near-idempotent heads are the "important" ones,
or why training produces this structure. The reduction theorem itself
(nonzero spectrum of `A@B` equals that of `B@A`) is unconditional and doesn't
depend on the weights being trained, real, or well-behaved in any way — it's
run against every real head purely as a numerical sanity check on the
eigenvalue computation, not as a way to detect a mismatched or mis-extracted
head (a semantically wrong `(A, B)` pairing would still satisfy the theorem,
since it holds for any compatible matrices regardless of provenance).

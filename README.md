# discrete-gradient-synthesis (`seqsynth`)

A CPU-only Go command-line engine that compresses a small set of labeled real
sequences into a small number of **new, discrete synthetic sequences** whose
tokens come from the supplied vocabulary. It does **not** select or resample
real sequences.

The fixed classifier represents each sequence by the **mean of its token
embeddings** and applies a linear softmax classifier (cross-entropy). The
synthesis objective is to make the **average classifier-parameter gradient of
the synthetic samples match that of the real samples**.

## What the core actually does

This is scaled-form ADMM over every synthetic slot position `(j, l)`:

- `z_{j,l}`: continuous embedding, optimized by genuine inner optimization
  (gradient steps with Armijo backtracking) on
  `f(z) + (penalty/2) || z - x + u ||^2`, where `f` is the gradient-matching
  loss.
- `x_{j,l}`: a vocabulary embedding chosen by projecting the
  **multiplier-corrected** point `z + u` onto the nearest vocabulary embedding.
- `u_{j,l}`: scaled consistency multiplier, updated as `u <- u + z - x`.

Per outer round the solver records:

- primal residual `rms(z - x)` — continuous/discrete agreement;
- dual residual `penalty * rms(x - x_prev)` — projected structure change.

Convergence is reported **only when both residuals are at or below `tol`**.
If the outer budget is exhausted first, the real result is still returned and
is explicitly flagged `converged=false`, `budget_exhausted=true`.

The final discrete output is produced by **recomputing the classifier gradient
from scratch over the emitted token sequences**. Two matching errors are
reported separately:

- `match_loss_continuous`: pre-projection error of the final continuous `z`;
- `match_loss_discrete`: post-projection error of the actual token output.

Continuous numbers are never substituted for discrete quality.

Projection distance ties resolve to the **smallest token id**, and all
randomness comes from a seeded source, so duplicate embeddings and repeated
runs are bit-for-bit reproducible.

## Build and run

```sh
go build ./...

# built-in, self-contained example (exits on its own)
go run . -example

# from a configuration file
go run . -config docs/config.example.json
```

`docs/config.budget-exhausted.json` is a deliberately tight-budget example
that returns a valid, non-converged, budget-exhausted result.

## Output

The result is JSON on stdout (errors go to stderr with a non-zero exit code and
produce no partial output):

```json
{
  "converged": true,
  "rounds_used": 17,
  "budget_exhausted": false,
  "synthetic_tokens": [[3, 3, 2], [0, 2, 3], [0, 4, 2], [1, 1, 1]],
  "synthetic_labels": [0, 0, 1, 1],
  "match_loss_discrete": 0.0192,
  "match_loss_continuous": 0.0193,
  "grad_diff_weights_discrete": 0.0,
  "grad_diff_bias_discrete": 0.0,
  "grad_diff_weights_continuous": 0.0,
  "grad_diff_bias_continuous": 0.0,
  "iterations": [
    {"round": 1, "primal_residual": 0.202, "dual_residual": 0.00624, "tol": 0.001}
  ]
}
```

## Configuration

See [docs/CONFIG.md](docs/CONFIG.md) for the full field reference, validation
rules, and tuning guidance.

## Tests

```sh
go test ./...
```

Tests cover the full synthesis flow, vocabulary feasibility, residual and
termination rules (both residuals required), projection ties and duplicate
embeddings, reproducibility, every class of illegal input (empty vocabulary,
out-of-range labels, dimension mismatch, non-positive penalty, non-finite
values), discrete-error recomputation, and independent verification of true
gradients (finite differences and a hand-built tiny example).

## Constraints

- Pure Go standard library; no third-party modules (`go list -m all` shows only
  the module itself).
- CPU only; no external services, language models, or GPU calls.

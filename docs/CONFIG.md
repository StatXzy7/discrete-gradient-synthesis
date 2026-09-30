# Configuration reference

`seqsynth -config <file>` reads a single JSON document. Unknown fields are
rejected, and validation happens **before** any computation; invalid input
returns an error and produces no partial output.

## Classifier and vocabulary

| Field | Type | Meaning |
| --- | --- | --- |
| `vocab_size` | int > 0 | Number of tokens `V`. Empty vocabulary is rejected. |
| `dim` | int > 0 | Embedding/feature dimension `D`. |
| `classes` | int > 0 | Number of classes `K`. |
| `embeddings` | `V` x `D` floats | Token embedding table; all entries must be finite. |
| `weights` | `K` x `D` floats | Fixed linear classifier matrix; all finite. |
| `bias` | `K` floats | Fixed classifier bias; all finite. |

Sequence score: `logits_k = mean_l(embedding[token_l]) . weights[k] + bias[k]`,
then softmax cross-entropy.

## Real data

| Field | Type | Rule |
| --- | --- | --- |
| `real_tokens` | array of int arrays | At least one non-empty sequence. |
| `real_labels` | int array | Same length as `real_tokens`, each in `[0, K)`. |

Every token must be in `[0, vocab_size)`.

## Synthesis request

| Field | Type | Rule |
| --- | --- | --- |
| `synth_count` | int > 0 | Number of synthetic sequences. |
| `seq_length` | int > 0 | Length `L` of every synthetic sequence. |
| `synth_labels` | int array | Exactly `synth_count` labels, each in `[0, K)`. |

Synthetic labels are fixed request parameters.

## Optimization controls

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `penalty` | finite float > 0 | required | ADMM consistency penalty `rho`. |
| `inner_steps` | int > 0 | required | Continuous `z` update steps per outer round. |
| `step_size` | finite float > 0 | `0.5` | Initial inner gradient step; Armijo shrinks it. |
| `max_outer` | int > 0 | required | Finite outer-round budget. |
| `tol` | finite float > 0 | required | Residual tolerance (RMS scale). |
| `seed` | int64 | `0` | RNG seed for initialization; controls all randomness. |
| `init_spread` | finite float > 0 | `0.1` | Gaussian jitter scale for initial continuous points. |

## Validation failures (non-exhaustive)

- empty vocabulary (`vocab_size <= 0`);
- non-positive `dim` / `classes` / `synth_count` / `seq_length`;
- embedding, weight, or bias row lengths inconsistent with `dim`/`classes`;
- real or synthetic label outside `[0, classes)`;
- real token outside `[0, vocab_size)`, or empty real sequence;
- `real_tokens`/`real_labels` length mismatch;
- wrong number of synthetic labels;
- non-positive or non-finite `penalty`;
- non-positive `inner_steps` / `max_outer`;
- non-positive or non-finite `tol`;
- any `NaN`/`+Inf`/`-Inf` in embeddings, weights, or bias.

## Output fields

- `converged`: true only when the **last** round has both
  `primal_residual <= tol` and `dual_residual <= tol`.
- `budget_exhausted`: true when `max_outer` is reached without convergence; the
  current result is still returned.
- `rounds_used`: number of outer rounds actually run.
- `synthetic_tokens`: `synth_count` x `seq_length` integer token ids.
- `synthetic_labels`: echoed request labels.
- `match_loss_continuous`: `0.5||gW_s-gW_r||^2 + 0.5||gb_s-gb_r||^2` evaluated
  at the final continuous embeddings (pre-projection).
- `match_loss_discrete`: the same objective recomputed from the emitted token
  sequences (post-projection).
- `grad_diff_weights_*`, `grad_diff_bias_*`: Euclidean distances of synthetic
  vs. real average gradients, separately for continuous and discrete outputs.
- `iterations`: per-round `primal_residual`, `dual_residual`, `tol`.

## Tuning notes

- Smaller `penalty` allows tokens to change more freely and often reaches lower
  matching error, but needs more outer rounds; with too tight a budget it stays
  non-converged (by design).
- Larger `penalty` converges in fewer rounds but can lock the discrete
  structure early, giving less gradient matching.
- Increase `inner_steps` (and optionally `step_size`) to solve each continuous
  subproblem more accurately; Armijo backtracking prevents divergence.
- Different `seed` values change only the initial continuous points.

## Minimal example

See [`config.example.json`](config.example.json) (converges) and
[`config.budget-exhausted.json`](config.budget-exhausted.json) (returns a valid
budget-exhausted result).

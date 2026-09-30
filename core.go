// Package main implements a discrete sequence synthesis engine.
//
// It compresses a small set of labeled real sequences into a small number of
// labeled synthetic sequences whose tokens come from the supplied vocabulary.
// The classifier is fixed: it averages the token embeddings of a sequence and
// applies a linear softmax (cross-entropy) classifier. Synthesis matches the
// average classifier gradient of the real sequences. The core solver performs
// genuine alternating optimization of continuous embeddings, projection of the
// multiplier-corrected embeddings onto the vocabulary, and consistency
// multiplier updates (ADMM); it never selects real samples.
package main

import (
	"errors"
	"fmt"
	"math"
)

// Model is the fixed classifier: embedding table E (V x D), weights W (K x D)
// and per-class bias b (K). A sequence is scored as logits_k = mean(E[token]) .
// W[k] + b[k].
type Model struct {
	Embeddings [][]float64 // [vocab][dim]
	Weights    [][]float64 // [classes][dim]
	Bias       []float64   // [classes]
	Vocab      int
	Dim        int
	Classes    int
}

// Config is the synthesis configuration.
type Config struct {
	VocabSize  int         `json:"vocab_size"`
	Dim        int         `json:"dim"`
	Classes    int         `json:"classes"`
	Embeddings [][]float64 `json:"embeddings"`
	Weights    [][]float64 `json:"weights"`
	Bias       []float64   `json:"bias"`

	RealTokens [][]int `json:"real_tokens"`
	RealLabels []int   `json:"real_labels"`

	SynthCount  int   `json:"synth_count"`
	SeqLength   int   `json:"seq_length"`
	SynthLabels []int `json:"synth_labels"`

	Penalty    float64 `json:"penalty"`
	InnerSteps int     `json:"inner_steps"`
	StepSize   float64 `json:"step_size"`
	MaxOuter   int     `json:"max_outer"`
	Tol        float64 `json:"tol"`
	Seed       int64   `json:"seed"`
	InitSpread float64 `json:"init_spread"`
}

// Iteration records the primal and dual residuals at one outer round.
type Iteration struct {
	Round       int     `json:"round"`
	PrimalResid float64 `json:"primal_residual"`
	DualResid   float64 `json:"dual_residual"`
	Tol         float64 `json:"tol"`
}

// Result is the synthesis output.
type Result struct {
	Converged           bool        `json:"converged"`
	Rounds              int         `json:"rounds_used"`
	BudgetExhausted     bool        `json:"budget_exhausted"`
	Tokens              [][]int     `json:"synthetic_tokens"`
	Labels              []int       `json:"synthetic_labels"`
	MatchLossDiscrete   float64     `json:"match_loss_discrete"`
	MatchLossContinuous float64     `json:"match_loss_continuous"`
	GradDiffWDiscrete   float64     `json:"grad_diff_weights_discrete"`
	GradDiffBDiscrete   float64     `json:"grad_diff_bias_discrete"`
	GradDiffWContinuous float64     `json:"grad_diff_weights_continuous"`
	GradDiffBContinuous float64     `json:"grad_diff_bias_continuous"`
	Iterations          []Iteration `json:"iterations"`

	// contZ holds the final continuous solution; used by tests and not
	// serialized into the user-facing report.
	contZ []float64 `json:"-"`
}

func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func finiteVec(v []float64) bool {
	for _, x := range v {
		if !isFinite(x) {
			return false
		}
	}
	return true
}

func finiteMat(m [][]float64) bool {
	for _, row := range m {
		if !finiteVec(row) {
			return false
		}
	}
	return true
}

func norm2(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

// Validate checks every structural and numerical precondition. On failure the
// caller must not produce any partial synthesis output.
func (c *Config) Validate() error {
	if c.VocabSize <= 0 {
		return errors.New("vocab_size must be positive (empty vocabulary rejected)")
	}
	if c.Dim <= 0 {
		return errors.New("dim must be positive")
	}
	if c.Classes <= 0 {
		return errors.New("classes must be positive")
	}
	if len(c.Embeddings) != c.VocabSize {
		return fmt.Errorf("embeddings: got %d rows, want %d", len(c.Embeddings), c.VocabSize)
	}
	for i, e := range c.Embeddings {
		if len(e) != c.Dim {
			return fmt.Errorf("embedding %d: got dim %d, want %d", i, len(e), c.Dim)
		}
		if !finiteVec(e) {
			return fmt.Errorf("embedding %d contains non-finite value", i)
		}
	}
	if len(c.Weights) != c.Classes {
		return fmt.Errorf("weights: got %d rows, want %d", len(c.Weights), c.Classes)
	}
	for k, w := range c.Weights {
		if len(w) != c.Dim {
			return fmt.Errorf("weight class %d: got dim %d, want %d", k, len(w), c.Dim)
		}
		if !finiteVec(w) {
			return fmt.Errorf("weight class %d contains non-finite value", k)
		}
	}
	if len(c.Bias) != c.Classes {
		return fmt.Errorf("bias: got %d entries, want %d", len(c.Bias), c.Classes)
	}
	if !finiteVec(c.Bias) {
		return errors.New("bias contains non-finite value")
	}
	if len(c.RealTokens) == 0 {
		return errors.New("at least one real sequence is required")
	}
	if len(c.RealTokens) != len(c.RealLabels) {
		return fmt.Errorf("real_tokens (%d) and real_labels (%d) length mismatch",
			len(c.RealTokens), len(c.RealLabels))
	}
	for i, seq := range c.RealTokens {
		if len(seq) == 0 {
			return fmt.Errorf("real sequence %d is empty", i)
		}
		for _, t := range seq {
			if t < 0 || t >= c.VocabSize {
				return fmt.Errorf("real sequence %d: token %d out of range [0,%d)", i, t, c.VocabSize)
			}
		}
		if c.RealLabels[i] < 0 || c.RealLabels[i] >= c.Classes {
			return fmt.Errorf("real label %d out of range [0,%d)", c.RealLabels[i], c.Classes)
		}
	}
	if c.SynthCount <= 0 {
		return errors.New("synth_count must be positive")
	}
	if c.SeqLength <= 0 {
		return errors.New("seq_length must be positive")
	}
	if len(c.SynthLabels) != c.SynthCount {
		return fmt.Errorf("synth_labels: got %d entries, want %d", len(c.SynthLabels), c.SynthCount)
	}
	for _, y := range c.SynthLabels {
		if y < 0 || y >= c.Classes {
			return fmt.Errorf("synthetic label %d out of range [0,%d)", y, c.Classes)
		}
	}
	if !(c.Penalty > 0) || !isFinite(c.Penalty) {
		return errors.New("penalty must be a positive finite number")
	}
	if c.InnerSteps <= 0 {
		return errors.New("inner_steps must be positive")
	}
	if c.MaxOuter <= 0 {
		return errors.New("max_outer must be positive")
	}
	if !(c.Tol > 0) || !isFinite(c.Tol) {
		return errors.New("tol must be a positive finite number")
	}
	return nil
}

// stepSize returns the configured inner gradient step or a default.
func (c *Config) stepSize() float64 {
	if c.StepSize > 0 && isFinite(c.StepSize) {
		return c.StepSize
	}
	return 0.5
}

// initSpread returns the configured initial continuous spread or a default.
func (c *Config) initSpread() float64 {
	if c.InitSpread > 0 && isFinite(c.InitSpread) {
		return c.InitSpread
	}
	return 0.1
}

func newModel(c *Config) *Model {
	return &Model{
		Embeddings: c.Embeddings,
		Weights:    c.Weights,
		Bias:       c.Bias,
		Vocab:      c.VocabSize,
		Dim:        c.Dim,
		Classes:    c.Classes,
	}
}

// softmax writes the softmax of logits into out.
func softmax(logits, out []float64) {
	m := math.Inf(-1)
	for _, x := range logits {
		if x > m {
			m = x
		}
	}
	var sum float64
	for k, x := range logits {
		out[k] = math.Exp(x - m)
		sum += out[k]
	}
	for k := range out {
		out[k] /= sum
	}
}

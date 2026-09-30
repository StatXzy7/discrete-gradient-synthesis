package main

import (
	"math"
	"testing"
)

func validBase() *Config {
	c := ExampleConfig()
	c.MaxOuter = 5
	return c
}

func TestValidBaseSolves(t *testing.T) {
	if _, err := Solve(validBase()); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
}

func TestIllegalInputsRejected(t *testing.T) {
	cases := map[string]func(c *Config){
		"empty vocabulary":      func(c *Config) { c.VocabSize = 0 },
		"zero dimension":        func(c *Config) { c.Dim = 0 },
		"zero classes":          func(c *Config) { c.Classes = 0 },
		"embedding count":       func(c *Config) { c.Embeddings = c.Embeddings[:3] },
		"embedding dim":         func(c *Config) { c.Embeddings[0] = []float64{1, 2} },
		"weight class count":    func(c *Config) { c.Weights = c.Weights[:1] },
		"weight dim":            func(c *Config) { c.Weights[0] = []float64{1, 2} },
		"bias count":            func(c *Config) { c.Bias = []float64{0} },
		"no real sequences":     func(c *Config) { c.RealTokens = nil },
		"label length mismatch": func(c *Config) { c.RealLabels = []int{0} },
		"empty real sequence":   func(c *Config) { c.RealTokens[0] = nil },
		"real token too low":    func(c *Config) { c.RealTokens[0][0] = -1 },
		"real token too high":   func(c *Config) { c.RealTokens[0][0] = c.VocabSize },
		"real label high":       func(c *Config) { c.RealLabels[0] = c.Classes },
		"real label negative":   func(c *Config) { c.RealLabels[0] = -1 },
		"synth count zero":      func(c *Config) { c.SynthCount = 0 },
		"seq length zero":       func(c *Config) { c.SeqLength = 0 },
		"synth labels count":    func(c *Config) { c.SynthLabels = []int{0} },
		"synth label range":     func(c *Config) { c.SynthLabels[0] = 9 },
		"zero penalty":          func(c *Config) { c.Penalty = 0 },
		"negative penalty":      func(c *Config) { c.Penalty = -1 },
		"nan penalty":           func(c *Config) { c.Penalty = math.NaN() },
		"inf penalty":           func(c *Config) { c.Penalty = math.Inf(1) },
		"zero inner steps":      func(c *Config) { c.InnerSteps = 0 },
		"zero max outer":        func(c *Config) { c.MaxOuter = 0 },
		"zero tol":              func(c *Config) { c.Tol = 0 },
		"nan tol":               func(c *Config) { c.Tol = math.NaN() },
		"nan embedding":         func(c *Config) { c.Embeddings[1][0] = math.NaN() },
		"inf weight":            func(c *Config) { c.Weights[0][0] = math.Inf(-1) },
		"nan bias":              func(c *Config) { c.Bias[0] = math.NaN() },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			c := validBase()
			mutate(c)
			res, err := Solve(c)
			if err == nil {
				t.Fatalf("expected rejection, got result=%+v", res)
			}
			if res != nil {
				t.Fatalf("illegal input produced partial result: %+v", res)
			}
		})
	}
}

func TestValidateBeforeSolveDoesNotMutate(t *testing.T) {
	c := validBase()
	c.Penalty = 0
	if err := c.Validate(); err == nil {
		t.Fatal("non-positive penalty must fail validation")
	}
}

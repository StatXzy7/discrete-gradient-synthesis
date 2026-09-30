package main

import "testing"

// TestProjectionTieStable constructs an exact equidistant tie. With target at
// the origin and unit-axis embeddings, every token is the same distance. The
// smallest token id must win, deterministically, regardless of the order the
// caller permutes nothing (the projection itself scans id in increasing
// order). A slightly closer later token still wins on genuine distance.
func TestProjectionTieStable(t *testing.T) {
	m := &Model{
		Vocab:   4,
		Dim:     2,
		Classes: 2,
		Embeddings: [][]float64{
			{1, 0},
			{0, 1},
			{-1, 0},
			{0, -1},
		},
	}
	if id := projectOne(m, []float64{0, 0}); id != 0 {
		t.Fatalf("tie broken to token %d, want smallest id 0", id)
	}

	// Duplicate embeddings at larger ids must never beat the smaller id.
	m2 := &Model{
		Vocab:   3,
		Dim:     2,
		Classes: 2,
		Embeddings: [][]float64{
			{0.5, 0.5},
			{0.5, 0.5}, // exact duplicate of 0
			{0.5, 0.5}, // exact duplicate of 0
		},
	}
	if id := projectOne(m2, []float64{0.5, 0.5}); id != 0 {
		t.Fatalf("duplicate embedding resolved to %d, want id 0", id)
	}
	if id := projectOne(m2, []float64{0.6, 0.4}); id != 0 {
		t.Fatalf("duplicate-vocab nearest resolved to %d, want id 0", id)
	}

	// A strictly closer larger id wins (tie must not always favor 0).
	m3 := &Model{
		Vocab: 3, Dim: 1, Classes: 2,
		Embeddings: [][]float64{{10}, {5}, {0}},
	}
	if id := projectOne(m3, []float64{-0.1}); id != 2 {
		t.Fatalf("nearest token = %d, want 2", id)
	}
}

// TestProjectionAllWritesEmbeddingsAndTokens ensures x is exactly the selected
// vocabulary embedding and tokens agree with x.
func TestProjectionAllWritesEmbeddingsAndTokens(t *testing.T) {
	c := ExampleConfig()
	m := newModel(c)
	count := c.SynthCount * c.SeqLength
	z := make([]float64, count*c.Dim)
	u := make([]float64, len(z))
	x := make([]float64, len(z))
	tok := make([]int, count)
	projectAll(m, z, u, x, tok)
	for i, id := range tok {
		e := m.Embeddings[id]
		off := i * c.Dim
		for d := 0; d < c.Dim; d++ {
			if x[off+d] != e[d] {
				t.Fatalf("x[%d] mismatch embedding of token %d", i, id)
			}
		}
	}
}

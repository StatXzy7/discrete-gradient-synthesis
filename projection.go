package main

import "math"

// projectOne finds the vocabulary embedding closest to target and returns its
// token id. Exact-distance ties resolve to the smallest token id. Only a strict
// improvement replaces the current best, so iteration order (vocabulary id
// order) yields stable, reproducible tie breaking; duplicate embeddings at
// larger ids never beat the equal-distance smaller id.
func projectOne(m *Model, target []float64) int {
	best := 0
	bestDist := math.Inf(1)
	for id := 0; id < m.Vocab; id++ {
		var dist float64
		e := m.Embeddings[id]
		for d := 0; d < m.Dim; d++ {
			diff := target[d] - e[d]
			dist += diff * diff
		}
		if dist < bestDist {
			bestDist = dist
			best = id
		}
	}
	return best
}

// projectAll projects every multiplier-corrected point v = z + u onto the
// vocabulary, writing the nearest embedding into x and the token id into tok.
func projectAll(m *Model, z, u, x []float64, tok []int) {
	target := make([]float64, m.Dim)
	count := len(tok)
	for i := 0; i < count; i++ {
		off := i * m.Dim
		for d := 0; d < m.Dim; d++ {
			target[d] = z[off+d] + u[off+d]
		}
		id := projectOne(m, target)
		tok[i] = id
		e := m.Embeddings[id]
		for d := 0; d < m.Dim; d++ {
			x[off+d] = e[d]
		}
	}
}

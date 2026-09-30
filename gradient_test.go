package main

import (
	"math"
	"testing"
)

// tinyModel is an independent hand-built 2-class, 2-dim, 3-token model used to
// verify gradients against finite differences and hand calculations.
func tinyModel() *Model {
	return &Model{
		Vocab:   3,
		Dim:     2,
		Classes: 2,
		Embeddings: [][]float64{
			{1, 0},
			{0, 1},
			{1, 1},
		},
		Weights: [][]float64{
			{0.5, -0.25},
			{-0.4, 0.8},
		},
		Bias: []float64{0.1, -0.2},
	}
}

// independentlyComputeA re-derives a single slot mean without using slotMeans.
func independentlyComputeA(z []float64, length, dim int) []float64 {
	a := make([]float64, dim)
	for l := 0; l < length; l++ {
		for d := 0; d < dim; d++ {
			a[d] += z[l*dim+d]
		}
	}
	for d := range a {
		a[d] /= float64(length)
	}
	return a
}

// independentSoftmaxGrad computes g_k = softmax(logits)_k - 1{k=y} directly.
func independentSoftmaxGrad(m *Model, a []float64, y int) []float64 {
	logits := make([]float64, m.Classes)
	for k := 0; k < m.Classes; k++ {
		s := m.Bias[k]
		for d := 0; d < m.Dim; d++ {
			s += m.Weights[k][d] * a[d]
		}
		logits[k] = s
	}
	mx := math.Inf(-1)
	for _, v := range logits {
		if v > mx {
			mx = v
		}
	}
	p := make([]float64, m.Classes)
	var sum float64
	for k, v := range logits {
		p[k] = math.Exp(v - mx)
		sum += p[k]
	}
	for k := range p {
		p[k] /= sum
		if k == y {
			p[k] -= 1
		}
	}
	return p
}

func TestRealGradientHandCalculation(t *testing.T) {
	m := tinyModel()
	toks := [][]int{{0, 1}, {2}}
	labels := []int{0, 1}
	gw, gb := realAverageGradient(m, toks, labels)

	// Independent reference for sequence 0: mean embedding of {0,1} = (0.5,0.5).
	a0 := []float64{0.5, 0.5}
	g0 := independentSoftmaxGrad(m, a0, 0)
	// Sequence 1: embedding of token 2 = (1,1).
	a1 := []float64{1, 1}
	g1 := independentSoftmaxGrad(m, a1, 1)

	wantB := make([]float64, m.Classes)
	wantW := make([][]float64, m.Classes)
	for k := range wantW {
		wantW[k] = make([]float64, m.Dim)
	}
	add := func(a []float64, g []float64) {
		for k := 0; k < m.Classes; k++ {
			wantB[k] += g[k]
			for d := 0; d < m.Dim; d++ {
				wantW[k][d] += g[k] * a[d]
			}
		}
	}
	add(a0, g0)
	add(a1, g1)
	for k := range wantB {
		wantB[k] /= 2
		for d := range wantW[k] {
			wantW[k][d] /= 2
		}
	}

	for k := 0; k < m.Classes; k++ {
		if math.Abs(gb[k]-wantB[k]) > 1e-12 {
			t.Fatalf("gb[%d]=%.10f want %.10f", k, gb[k], wantB[k])
		}
		for d := 0; d < m.Dim; d++ {
			if math.Abs(gw[k][d]-wantW[k][d]) > 1e-12 {
				t.Fatalf("gw[%d][%d]=%.10f want %.10f", k, d, gw[k][d], wantW[k][d])
			}
		}
	}
}

func TestMatchGradientFiniteDifference(t *testing.T) {
	m := tinyModel()
	n, L := 2, 3
	labels := []int{0, 1}
	total := n * L * m.Dim

	// Deterministic but non-trivial continuous z.
	z := make([]float64, total)
	for i := range z {
		z[i] = 0.3*math.Sin(float64(i+1)) + 0.5
	}
	gwR, gbR := realAverageGradient(m, [][]int{{0, 2}, {1, 1, 2}}, []int{0, 1})
	info := newGradInfo(n, m.Classes, m.Dim)
	grad := make([]float64, total)

	_, gwS, gbS := rawMatchLoss(m, z, L, labels, gwR, gbR, info)
	matchGradient(m, gwS, gbS, gwR, gbR, labels, L, info, grad)

	eps := 1e-6
	for i := 0; i < total; i++ {
		zp := append([]float64(nil), z...)
		zm := append([]float64(nil), z...)
		zp[i] += eps
		zm[i] -= eps
		fp, _, _ := rawMatchLoss(m, zp, L, labels, gwR, gbR, newGradInfo(n, m.Classes, m.Dim))
		fm, _, _ := rawMatchLoss(m, zm, L, labels, gwR, gbR, newGradInfo(n, m.Classes, m.Dim))
		num := (fp - fm) / (2 * eps)
		den := math.Max(1, math.Abs(grad[i]))
		if math.Abs(num-grad[i])/den > 1e-6 {
			t.Fatalf("grad[%d]=%.8f finite-diff=%.8f", i, grad[i], num)
		}
	}
}

func TestAugmentedGradientFiniteDifference(t *testing.T) {
	m := tinyModel()
	n, L := 2, 2
	labels := []int{1, 0}
	total := n * L * m.Dim
	z := []float64{0.2, 0.9, 0.7, 0.4, 1.0, 0.1, 0.3, 0.6}
	x := []float64{0, 1, 1, 0, 1, 1, 0, 1}
	u := make([]float64, total)
	for i := range u {
		u[i] = 0.05 * float64(i-3)
	}
	rho := 0.37
	gwR, gbR := realAverageGradient(m, [][]int{{2, 0}, {1}}, []int{1, 0})
	info := newGradInfo(n, m.Classes, m.Dim)
	grad := make([]float64, total)
	_, gwS, gbS := augmentedObjective(m, z, x, u, rho, L, labels, gwR, gbR, info)
	matchGradient(m, gwS, gbS, gwR, gbR, labels, L, info, grad)
	for i := range z {
		grad[i] += rho * (z[i] - x[i] + u[i])
	}

	eps := 1e-6
	for i := 0; i < total; i++ {
		zp := append([]float64(nil), z...)
		zm := append([]float64(nil), z...)
		zp[i] += eps
		zm[i] -= eps
		fp, _, _ := augmentedObjective(m, zp, x, u, rho, L, labels, gwR, gbR, newGradInfo(n, m.Classes, m.Dim))
		fm, _, _ := augmentedObjective(m, zm, x, u, rho, L, labels, gwR, gbR, newGradInfo(n, m.Classes, m.Dim))
		num := (fp - fm) / (2 * eps)
		den := math.Max(1, math.Abs(grad[i]))
		if math.Abs(num-grad[i])/den > 1e-6 {
			t.Fatalf("aug grad[%d]=%.8f fd=%.8f", i, grad[i], num)
		}
	}
}

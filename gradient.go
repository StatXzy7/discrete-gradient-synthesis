package main

// meanEmbedding returns the mean embedding of a discrete sequence.
func meanEmbedding(m *Model, tokens []int, out []float64) {
	for d := 0; d < m.Dim; d++ {
		out[d] = 0
	}
	for _, t := range tokens {
		e := m.Embeddings[t]
		for d := 0; d < m.Dim; d++ {
			out[d] += e[d]
		}
	}
	inv := 1.0 / float64(len(tokens))
	for d := 0; d < m.Dim; d++ {
		out[d] *= inv
	}
}

// realAverageGradient computes the average classifier parameter gradient of
// the real sequences: grad W[k,d] = g_k * a_d and grad b_k = g_k, where
// g_k = softmax(logits)_k - 1{k=y}.
func realAverageGradient(m *Model, toks [][]int, labels []int) ([][]float64, []float64) {
	gw := make([][]float64, m.Classes)
	for k := range gw {
		gw[k] = make([]float64, m.Dim)
	}
	gb := make([]float64, m.Classes)
	a := make([]float64, m.Dim)
	logits := make([]float64, m.Classes)
	probs := make([]float64, m.Classes)
	for i, seq := range toks {
		meanEmbedding(m, seq, a)
		for k := 0; k < m.Classes; k++ {
			s := m.Bias[k]
			w := m.Weights[k]
			for d := 0; d < m.Dim; d++ {
				s += w[d] * a[d]
			}
			logits[k] = s
		}
		softmax(logits, probs)
		y := labels[i]
		for k := 0; k < m.Classes; k++ {
			gk := probs[k]
			if k == y {
				gk -= 1
			}
			gb[k] += gk
			row := gw[k]
			for d := 0; d < m.Dim; d++ {
				row[d] += gk * a[d]
			}
		}
	}
	n := float64(len(toks))
	for k := 0; k < m.Classes; k++ {
		gb[k] /= n
		for d := 0; d < m.Dim; d++ {
			gw[k][d] /= n
		}
	}
	return gw, gb
}

// slotMeans computes a_j = mean_l z[j,l] from the flat continuous buffer.
func slotMeans(z []float64, n, length, dim int, a [][]float64) {
	for j := 0; j < n; j++ {
		aj := a[j]
		base := j * length * dim
		for d := 0; d < dim; d++ {
			aj[d] = 0
		}
		for l := 0; l < length; l++ {
			off := base + l*dim
			for d := 0; d < dim; d++ {
				aj[d] += z[off+d]
			}
		}
		inv := 1.0 / float64(length)
		for d := 0; d < dim; d++ {
			aj[d] *= inv
		}
	}
}

// gradInfo caches slot means, softmax-input gradients and probabilities.
type gradInfo struct {
	a     [][]float64
	gL    [][]float64
	probs [][]float64
}

func newGradInfo(n, classes, dim int) *gradInfo {
	a := make([][]float64, n)
	gL := make([][]float64, n)
	probs := make([][]float64, n)
	for j := range a {
		a[j] = make([]float64, dim)
		gL[j] = make([]float64, classes)
		probs[j] = make([]float64, classes)
	}
	return &gradInfo{a: a, gL: gL, probs: probs}
}

// synthForward fills slot means a, softmax gradients gL and probabilities, and
// returns the synthetic mean parameter gradients.
func synthForward(m *Model, a [][]float64, labels []int, gL, probs [][]float64) ([][]float64, []float64) {
	n := len(labels)
	logits := make([]float64, m.Classes)
	p := make([]float64, m.Classes)
	gw := make([][]float64, m.Classes)
	for k := range gw {
		gw[k] = make([]float64, m.Dim)
	}
	gb := make([]float64, m.Classes)
	for j := 0; j < n; j++ {
		aj := a[j]
		for k := 0; k < m.Classes; k++ {
			s := m.Bias[k]
			w := m.Weights[k]
			for d := 0; d < m.Dim; d++ {
				s += w[d] * aj[d]
			}
			logits[k] = s
		}
		softmax(logits, p)
		copy(probs[j], p)
		y := labels[j]
		for k := 0; k < m.Classes; k++ {
			gk := p[k]
			if k == y {
				gk -= 1
			}
			gL[j][k] = gk
			gb[k] += gk
			row := gw[k]
			for d := 0; d < m.Dim; d++ {
				row[d] += gk * aj[d]
			}
		}
	}
	inv := 1.0 / float64(n)
	for k := 0; k < m.Classes; k++ {
		gb[k] *= inv
		for d := 0; d < m.Dim; d++ {
			gw[k][d] *= inv
		}
	}
	return gw, gb
}

// rawMatchLoss evaluates 0.5||gW_s-gW_r||^2 + 0.5||gb_s-gb_r||^2 and caches
// slot means, softmax gradients and probabilities.
func rawMatchLoss(m *Model, z []float64, length int, labels []int,
	gwR [][]float64, gbR []float64, info *gradInfo,
) (float64, [][]float64, []float64) {
	n := len(labels)
	slotMeans(z, n, length, m.Dim, info.a)
	gw, gb := synthForward(m, info.a, labels, info.gL, info.probs)
	var loss float64
	for k := 0; k < m.Classes; k++ {
		for d := 0; d < m.Dim; d++ {
			diff := gw[k][d] - gwR[k][d]
			loss += diff * diff
		}
		diff := gb[k] - gbR[k]
		loss += diff * diff
	}
	return 0.5 * loss, gw, gb
}

// matchGradient writes the analytic gradient of the raw matching loss with
// respect to every continuous embedding entry into grad.
//
//	dW[k,d] = gW_s[k,d]-gW_r[k,d],  db[k] = gb_s[k]-gb_r[k]
//	dL/dgL_jk = (1/n)(sum_d dW[k,d] a_jd + db[k])
//
// The softmax Jacobian requires removing the projection of dL/dgL onto the
// probability simplex: h_k = e_k - p_k(sum e). Then
//
//	dL/da_jd = sum_k h_k W[k,d] + (1/n) sum_k gL_jk dW[k,d]
//	dL/dz_jld = (1/L) dL/da_jd
func matchGradient(m *Model, gwS [][]float64, gbS []float64,
	gwR [][]float64, gbR []float64, labels []int, length int,
	info *gradInfo, grad []float64,
) {
	n := len(labels)
	dW := make([][]float64, m.Classes)
	for k := range dW {
		dW[k] = make([]float64, m.Dim)
		for d := 0; d < m.Dim; d++ {
			dW[k][d] = gwS[k][d] - gwR[k][d]
		}
	}
	db := make([]float64, m.Classes)
	for k := range db {
		db[k] = gbS[k] - gbR[k]
	}
	invN := 1.0 / float64(n)
	invL := 1.0 / float64(length)
	e := make([]float64, m.Classes)
	h := make([]float64, m.Classes)
	da := make([]float64, m.Dim)
	for j := 0; j < n; j++ {
		aj := info.a[j]
		p := info.probs[j]
		gLj := info.gL[j]
		var pe float64
		for k := 0; k < m.Classes; k++ {
			var dot float64
			row := dW[k]
			for d := 0; d < m.Dim; d++ {
				dot += row[d] * aj[d]
			}
			e[k] = invN * (dot + db[k])
			pe += p[k] * e[k]
		}
		// Softmax Jacobian acting on e:
		// (diag(p) - p p^T) e = p (*) e - p * (p^T e).
		for k := 0; k < m.Classes; k++ {
			h[k] = p[k]*e[k] - p[k]*pe
		}
		for d := 0; d < m.Dim; d++ {
			var direct float64
			for k := 0; k < m.Classes; k++ {
				direct += h[k] * m.Weights[k][d]
			}
			var outer float64
			for k := 0; k < m.Classes; k++ {
				outer += gLj[k] * dW[k][d]
			}
			da[d] = direct + invN*outer
		}
		base := j * length * m.Dim
		for l := 0; l < length; l++ {
			off := base + l*m.Dim
			for d := 0; d < m.Dim; d++ {
				grad[off+d] = invL * da[d]
			}
		}
	}
}

// allFinite reports whether every entry is a finite number.
func allFinite(v []float64) bool {
	for _, x := range v {
		if !isFinite(x) {
			return false
		}
	}
	return true
}

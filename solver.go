package main

import (
	"errors"
	"math"
	"math/rand"
)

// Solve runs the alternating optimization.
//
// For each synthetic slot position (j,l) it maintains:
//
//	z : continuous embedding (optimized by inner gradient steps)
//	x : projected vocabulary embedding (a discrete token)
//	u : scaled consistency multiplier
//
// The augmented-Lagrangian objective for the continuous update is
//
//	f(z) + (rho/2) || z - x + u ||^2 ,
//
// where f is the classifier-gradient matching loss. After the inner steps the
// multiplier-corrected points z+u are projected to the nearest vocabulary
// embedding, then u accumulates the primal error u <- u + z - x. This is
// standard scaled-form ADMM; discrete tokens are synthesized, never copied
// from the real data.
func Solve(c *Config) (*Result, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	m := newModel(c)

	gwR, gbR := realAverageGradient(m, c.RealTokens, c.RealLabels)

	n, L, D := c.SynthCount, c.SeqLength, c.Dim
	total := n * L * D

	rng := newSeededRand(c.Seed)
	z := make([]float64, total)
	u := make([]float64, total)
	x := make([]float64, total)
	tok := make([]int, n*L)

	// Initial continuous points: a vocabulary token chosen by the seeded RNG,
	// jittered by a small zero-mean Gaussian. Identical embeddings still give
	// identical projections because the RNG draws are position-deterministic.
	gauss := gaussian{rng: rng}
	spread := c.initSpread()
	for i := range tok {
		id := rng.Intn(m.Vocab)
		off := i * D
		e := m.Embeddings[id]
		for d := 0; d < D; d++ {
			z[off+d] = e[d] + spread*gauss.next()
		}
	}
	// Initialize x by direct projection of z (u == 0).
	projectAll(m, z, u, x, tok)

	info := newGradInfo(n, m.Classes, D)
	grad := make([]float64, total)
	rho := c.Penalty

	res := &Result{
		Tokens: make([][]int, n),
		Labels: append([]int(nil), c.SynthLabels...),
	}
	for j := 0; j < n; j++ {
		res.Tokens[j] = make([]int, L)
	}

	converged := false
	rounds := 0
	for r := 1; r <= c.MaxOuter; r++ {
		rounds = r

		// (1) Continuous z update: innerSteps Armijo gradient steps on
		// f(z) + (rho/2)||z-x+u||^2.
		if err := zUpdate(m, z, x, u, grad, c, gwR, gbR, info); err != nil {
			return nil, err
		}
		if !allFinite(z) {
			return nil, errors.New("non-finite continuous embedding during update")
		}

		// (2) Discrete projection of multiplier-corrected points.
		xPrev := make([]float64, total)
		copy(xPrev, x)
		projectAll(m, z, u, x, tok)

		// (3) Multiplier (dual) update.
		for i := 0; i < total; i++ {
			u[i] += z[i] - x[i]
		}

		// Residuals as RMS values so tol is scale-independent.
		var primalSq, dualSq float64
		for i := 0; i < total; i++ {
			p := z[i] - x[i]
			d := -rho * (x[i] - xPrev[i])
			primalSq += p * p
			dualSq += d * d
		}
		primal := math.Sqrt(primalSq / float64(total))
		dual := math.Sqrt(dualSq / float64(total))
		res.Iterations = append(res.Iterations, Iteration{
			Round:       r,
			PrimalResid: primal,
			DualResid:   dual,
			Tol:         c.Tol,
		})

		// Convergence requires BOTH primal and dual residual below tol.
		if residualsWithin(primal, dual, c.Tol) {
			converged = true
			break
		}
	}

	res.Rounds = rounds
	res.Converged = converged
	res.BudgetExhausted = !converged

	// Continuous (pre-projection) matching error, evaluated at final z.
	contLoss, gwC, gbC := rawMatchLoss(m, z, L, c.SynthLabels, gwR, gbR, info)
	res.MatchLossContinuous = contLoss
	res.GradDiffWContinuous = weightDiff(gwC, gwR)
	res.GradDiffBContinuous = biasDiff(gbC, gbR)
	res.contZ = append([]float64(nil), z...)

	// Discrete output: reassemble token sequences and recompute the classifier
	// gradient from scratch. Continuous results are never reported as discrete.
	for j := 0; j < n; j++ {
		for l := 0; l < L; l++ {
			res.Tokens[j][l] = tok[j*L+l]
		}
	}
	gwD, gbD := realAverageGradient(m, res.Tokens, c.SynthLabels)
	res.MatchLossDiscrete = discreteMatchLoss(gwD, gbD, gwR, gbR)
	res.GradDiffWDiscrete = weightDiff(gwD, gwR)
	res.GradDiffBDiscrete = biasDiff(gbD, gbR)

	return res, nil
}

// gaussian draws standard normal samples from a seeded source via Box-Muller.
type gaussian struct {
	rng   *rand.Rand
	have  bool
	cache float64
}

func (g *gaussian) next() float64 {
	if g.have {
		g.have = false
		return g.cache
	}
	u1 := g.rng.Float64()
	if u1 == 0 {
		u1 = 1e-300
	}
	u2 := g.rng.Float64()
	mag := math.Sqrt(-2 * math.Log(u1))
	g.cache = mag * math.Sin(2*math.Pi*u2)
	g.have = true
	return mag * math.Cos(2*math.Pi*u2)
}

func weightDiff(a, b [][]float64) float64 {
	var s float64
	for k := range a {
		for d := range a[k] {
			diff := a[k][d] - b[k][d]
			s += diff * diff
		}
	}
	return math.Sqrt(s)
}

func biasDiff(a, b []float64) float64 {
	var s float64
	for k := range a {
		diff := a[k] - b[k]
		s += diff * diff
	}
	return math.Sqrt(s)
}

// discreteMatchLoss is the same 0.5 squared-gradient objective used for the
// continuous evaluation, applied to genuinely recomputed discrete gradients.
func discreteMatchLoss(gwD [][]float64, gbD []float64, gwR [][]float64, gbR []float64) float64 {
	var s float64
	for k := range gwD {
		for d := range gwD[k] {
			diff := gwD[k][d] - gwR[k][d]
			s += diff * diff
		}
	}
	for k := range gbD {
		diff := gbD[k] - gbR[k]
		s += diff * diff
	}
	return 0.5 * s
}

// residualsWithin reports convergence: BOTH primal and dual residual must be
// at or below tol. Either one above tol means not converged.
func residualsWithin(primal, dual, tol float64) bool {
	return primal <= tol && dual <= tol
}

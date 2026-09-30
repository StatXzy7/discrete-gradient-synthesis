package main

import "errors"

// augmentedObjective evaluates f(z) + (rho/2)||z - x + u||^2 and returns the
// raw matching loss plus the quadratic consistency penalty.
func augmentedObjective(m *Model, z, x, u []float64, rho float64, length int,
	labels []int, gwR [][]float64, gbR []float64, info *gradInfo,
) (float64, [][]float64, []float64) {
	f, gwS, gbS := rawMatchLoss(m, z, length, labels, gwR, gbR, info)
	var pen float64
	for i := range z {
		d := z[i] - x[i] + u[i]
		pen += d * d
	}
	return f + 0.5*rho*pen, gwS, gbS
}

// zUpdate performs innerSteps Armijo-backtracking gradient steps on the
// augmented objective. Each accepted step yields a sufficient-decreasing
// objective (Armijo condition); the step is shortened until the condition
// holds or it becomes negligibly small.
func zUpdate(m *Model, z, x, u, grad []float64, cfg *Config,
	gwR [][]float64, gbR []float64, info *gradInfo,
) error {
	rho := cfg.Penalty
	cand := make([]float64, len(z))
	for s := 0; s < cfg.InnerSteps; s++ {
		obj, gwS, gbS := augmentedObjective(m, z, x, u, rho, cfg.SeqLength,
			cfg.SynthLabels, gwR, gbR, info)
		matchGradient(m, gwS, gbS, gwR, gbR, cfg.SynthLabels, cfg.SeqLength, info, grad)
		for i := range z {
			grad[i] += rho * (z[i] - x[i] + u[i])
		}
		if !allFinite(grad) {
			return errors.New("non-finite gradient during continuous update")
		}
		var gradSq float64
		for i := range z {
			gradSq += grad[i] * grad[i]
		}
		step := cfg.stepSize()
		const c1 = 1e-4
		const maxTrials = 40
		accepted := false
		for t := 0; t < maxTrials; t++ {
			for i := range z {
				cand[i] = z[i] - step*grad[i]
			}
			if !allFinite(cand) {
				step *= 0.5
				continue
			}
			newObj, _, _ := augmentedObjective(m, cand, x, u, rho, cfg.SeqLength,
				cfg.SynthLabels, gwR, gbR, info)
			if isFinite(newObj) && newObj <= obj-c1*step*gradSq {
				accepted = true
				break
			}
			step *= 0.5
		}
		if !accepted {
			// z is numerically stationary for this subproblem.
			break
		}
		copy(z, cand)
	}
	return nil
}

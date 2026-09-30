package main

// ExampleConfig returns a self-contained built-in demonstration configuration.
// It uses a small 5-token vocabulary with 3-dimensional embeddings, a binary
// linear classifier, four balanced labeled real sequences and four synthetic
// slots of length three. The penalty is small enough that projected tokens
// genuinely change across outer rounds before both residuals reach tolerance.
func ExampleConfig() *Config {
	return &Config{
		VocabSize: 5,
		Dim:       3,
		Classes:   2,
		Embeddings: [][]float64{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
			{0.0, 0.0, 1.0},
			{0.6, 0.6, 0.0},
			{0.0, 0.6, 0.6},
		},
		Weights: [][]float64{
			{1.2, -0.8, 0.3},
			{-0.6, 1.0, -0.4},
		},
		Bias: []float64{0.1, -0.1},

		RealTokens: [][]int{
			{0, 3, 1},
			{0, 0, 3},
			{2, 4, 1},
			{4, 4, 2},
		},
		RealLabels: []int{0, 0, 1, 1},

		SynthCount:  4,
		SeqLength:   3,
		SynthLabels: []int{0, 0, 1, 1},

		Penalty:    0.03,
		InnerSteps: 80,
		StepSize:   1.0,
		MaxOuter:   300,
		Tol:        1e-3,
		Seed:       7,
		InitSpread: 0.15,
	}
}

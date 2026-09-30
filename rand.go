package main

import "math/rand"

// newSeededRand returns the deterministic RNG source used everywhere a seed
// controls randomness, so results are reproducible.
func newSeededRand(seed int64) *rand.Rand { return rand.New(rand.NewSource(seed)) }

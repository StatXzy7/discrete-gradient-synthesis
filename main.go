// Command seqsynth compresses labeled real sequences into a small number of
// discrete synthetic sequences that match the real average classifier
// gradient. It is CPU-only, calls no external services, and has no
// third-party dependencies.
//
// Usage:
//
//	seqsynth -example                 run the built-in self-contained demo
//	seqsynth -config path/to/config.json
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "seqsynth:", err)
		os.Exit(1)
	}
}

func run() error {
	example := flag.Bool("example", false, "run the built-in self-contained example")
	configPath := flag.String("config", "", "path to the JSON configuration file")
	flag.Parse()

	var cfg *Config
	switch {
	case *example:
		cfg = ExampleConfig()
	case *configPath != "":
		var err error
		cfg, err = LoadConfig(*configPath)
		if err != nil {
			return err
		}
	default:
		return fmt.Errorf("either -example or -config <file> is required")
	}

	res, err := Solve(cfg)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(res)
}

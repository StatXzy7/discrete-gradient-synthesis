package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestExampleCLISelfExits builds the binary and runs the built-in example,
// asserting it terminates on its own with valid converged JSON.
func TestExampleCLISelfExits(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping external binary test in short mode")
	}
	bin := filepath.Join(t.TempDir(), "seqsynth")
	build := exec.Command("go", "build", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	cmd := exec.Command(bin, "-example")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("example exited non-zero: %v\n%s", err, out)
	}
	var res Result
	if err := json.Unmarshal(out, &res); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, out)
	}
	if !res.Converged || res.BudgetExhausted {
		t.Fatalf("built-in example should converge: %+v", res)
	}
	if len(res.Iterations) == 0 {
		t.Fatal("no per-round residuals reported")
	}
}

// TestCLIRejectsBadConfig writes an invalid config and expects a non-zero exit
// with an error on stderr and NO JSON on stdout.
func TestCLIRejectsBadConfig(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping external binary test in short mode")
	}
	bin := filepath.Join(t.TempDir(), "seqsynth")
	build := exec.Command("go", "build", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(cfgPath, []byte(`{"vocab_size": 0}`), 0o600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(bin, "-config", cfgPath)
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err == nil {
		t.Fatal("expected non-zero exit for empty vocabulary")
	}
	if strings.TrimSpace(stdout.String()) != "" {
		t.Fatalf("invalid input produced partial stdout: %q", stdout.String())
	}
	if !strings.Contains(stderr.String(), "vocab") {
		t.Fatalf("stderr should explain the rejection, got %q", stderr.String())
	}
}

// TestLoadConfigRoundTrip serializes the example config and loads it back,
// checking strict decoding rejects unknown fields.
func TestLoadConfigRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	b, err := json.Marshal(ExampleConfig())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := LoadConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if c.VocabSize != ExampleConfig().VocabSize {
		t.Fatal("round-trip mismatch")
	}
	bad := filepath.Join(dir, "unknown.json")
	if err := os.WriteFile(bad, append(b[:len(b)-1], []byte(`, "bogus": 1}`)...), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(bad); err == nil {
		t.Fatal("unknown JSON field must be rejected")
	}
}

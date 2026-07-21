package core

import "testing"

func TestResultCite(t *testing.T) {
	r := Result{Tier: TierSemantic, SourceID: "doc1#3"}
	if got, want := r.Cite(), "[semantic:doc1#3]"; got != want {
		t.Fatalf("Cite() = %q, want %q", got, want)
	}
}

func TestChunkEmbedded(t *testing.T) {
	withCtx := Chunk{Text: "body", Context: "situating sentence"}
	if got, want := withCtx.Embedded(), "situating sentence\n\nbody"; got != want {
		t.Fatalf("Embedded() = %q, want %q", got, want)
	}
	bare := Chunk{Text: "body"}
	if got := bare.Embedded(); got != "body" {
		t.Fatalf("bare Embedded() = %q, want %q", got, "body")
	}
}

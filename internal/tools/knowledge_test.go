package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/knowledge"
)

func loadTestStore(t *testing.T) *knowledge.Store {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "calc.md"),
		[]byte("The calculator tool evaluates arithmetic expressions exactly."), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	store, err := knowledge.Load(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return store
}

// TestKnowledgeSearchCall は検索結果が出典付きで整形されることを確認します。
func TestKnowledgeSearchCall(t *testing.T) {
	tool := KnowledgeSearch{Store: loadTestStore(t)}
	out, err := tool.Call(context.Background(), json.RawMessage(`{"query":"calculator"}`))
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if !strings.Contains(out, "calc.md") {
		t.Errorf("expected source citation in output, got: %s", out)
	}
	if !strings.Contains(out, "calculator") {
		t.Errorf("expected matched content in output, got: %s", out)
	}
}

// TestKnowledgeSearchNoStore はナレッジ未設定でもエラーにならないことを確認します。
func TestKnowledgeSearchNoStore(t *testing.T) {
	tool := KnowledgeSearch{}
	out, err := tool.Call(context.Background(), json.RawMessage(`{"query":"anything"}`))
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if out == "" {
		t.Error("expected a message, got empty output")
	}
}

// TestKnowledgeSearchEmptyQuery は空クエリがエラーになることを確認します。
func TestKnowledgeSearchEmptyQuery(t *testing.T) {
	tool := KnowledgeSearch{Store: loadTestStore(t)}
	if _, err := tool.Call(context.Background(), json.RawMessage(`{"query":"  "}`)); err == nil {
		t.Error("expected error for empty query")
	}
}

// TestKnowledgeSearchNoMatch は一致無しのときに分かりやすいメッセージを返すことを確認します。
func TestKnowledgeSearchNoMatch(t *testing.T) {
	tool := KnowledgeSearch{Store: loadTestStore(t)}
	out, err := tool.Call(context.Background(), json.RawMessage(`{"query":"zzzzz nonexistent"}`))
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if !strings.Contains(strings.ToLower(out), "no relevant") {
		t.Errorf("expected a no-match message, got: %s", out)
	}
}

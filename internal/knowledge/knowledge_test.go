package knowledge

import (
	"os"
	"path/filepath"
	"testing"
)

// writeKnowledge は一時ディレクトリにナレッジファイルを作成するヘルパーです。
func writeKnowledge(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, content := range files {
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	return dir
}

// TestLoadMissingDir はディレクトリが無くてもエラーにならず空 Store を返すことを確認します。
func TestLoadMissingDir(t *testing.T) {
	store, err := Load(filepath.Join(t.TempDir(), "does-not-exist"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if store.Len() != 0 {
		t.Errorf("expected 0 chunks, got %d", store.Len())
	}
}

// TestLoadAndSources は対応拡張子のファイルだけが読み込まれることを確認します。
func TestLoadAndSources(t *testing.T) {
	dir := writeKnowledge(t, map[string]string{
		"a.md":       "# Title A\n\nThe calculator tool evaluates arithmetic expressions.",
		"sub/b.txt":  "The knowledge base stores documents for retrieval.",
		"ignore.log": "should not be loaded",
	})

	store, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if store.Len() == 0 {
		t.Fatal("expected chunks to be loaded")
	}

	sources := store.Sources()
	want := []string{"a.md", filepath.Join("sub", "b.txt")}
	if len(sources) != len(want) {
		t.Fatalf("sources = %v, want %v", sources, want)
	}
	for i := range want {
		if sources[i] != want[i] {
			t.Errorf("sources[%d] = %q, want %q", i, sources[i], want[i])
		}
	}
}

// TestSearchRanksRelevant は関連するチャンクが上位に来ることを確認します。
func TestSearchRanksRelevant(t *testing.T) {
	dir := writeKnowledge(t, map[string]string{
		"calc.md": "The calculator tool evaluates arithmetic expressions exactly.",
		"time.md": "The current_time tool reports the present date and time.",
	})
	store, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	got := store.Search("how does the calculator work", 2)
	if len(got) == 0 {
		t.Fatal("expected at least one result")
	}
	if src, _ := got[0].Metadata["source"].(string); src != "calc.md" {
		t.Errorf("top result source = %q, want calc.md", src)
	}
	if got[0].Score <= 0 {
		t.Errorf("expected positive score, got %v", got[0].Score)
	}
}

// TestSearchNoMatch は一致が無いときに空を返すことを確認します。
func TestSearchNoMatch(t *testing.T) {
	dir := writeKnowledge(t, map[string]string{
		"a.md": "The calculator tool evaluates arithmetic.",
	})
	store, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := store.Search("zzzzz nonexistent term", 3); len(got) != 0 {
		t.Errorf("expected no results, got %d", len(got))
	}
}

// TestSearchJapanese は日本語クエリでもチャンクを検索できることを確認します。
func TestSearchJapanese(t *testing.T) {
	dir := writeKnowledge(t, map[string]string{
		"ja.md": "電卓ツールは算術式を正確に評価します。",
		"en.md": "The clock reports the time.",
	})
	store, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got := store.Search("電卓", 2)
	if len(got) == 0 {
		t.Fatal("expected a result for Japanese query")
	}
	if src, _ := got[0].Metadata["source"].(string); src != "ja.md" {
		t.Errorf("top result source = %q, want ja.md", src)
	}
}

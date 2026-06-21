// Package tools はエージェントがモデルへ公開するツール抽象化を定義し、
// 安全で依存関係のない組み込みツールを一式提供します。
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/llm"
)

// Tool はエージェントがモデルへ公開できる単一の機能を表します。
// 実装は並行呼び出しに対して安全でなければなりません。
//
// このインターフェースは LangChain の Tool インターフェースに相当します。
// 各ツールは名前・説明・パラメータスキーマ・実行ロジックを持ちます。
type Tool interface {
	// Name はモデルへ公開される一意識別子です。
	// モデルはこの名前でツールを参照し、呼び出しをリクエストします。
	Name() string
	// Description はモデルがいつ・どのようにツールを使うかを説明します。
	// 具体的な説明を書くことでモデルの判断精度が向上します。
	Description() string
	// Parameters はツールの引数を記述する JSON Schema を返します。
	// モデルはこのスキーマに従って引数 JSON を生成します。
	Parameters() json.RawMessage
	// Call はモデルが提供した生 JSON 引数でツールを実行し、
	// 会話に返す文字列結果を返します。
	// エラーはエージェントによってテキストとしてモデルへ返されます。
	Call(ctx context.Context, args json.RawMessage) (string, error)
}

// Registry はエージェントが利用できるツールのセットを保持します。
// LangChain の ToolRegistry / BaseTool のコレクション管理に相当します。
type Registry struct {
	// tools はツール名からツール実装へのマップです。
	tools map[string]Tool
}

// NewRegistry は空の Registry を作成します。
func NewRegistry() *Registry {
	return &Registry{tools: make(map[string]Tool)}
}

// Register はツールを追加します。名前の重複がある場合はエラーを返すため、
// 起動時のワイヤリングミスが大きな音を立てて失敗します。
func (r *Registry) Register(t Tool) error {
	if t == nil {
		return fmt.Errorf("tools: cannot register nil tool")
	}
	name := t.Name()
	if name == "" {
		return fmt.Errorf("tools: tool has empty name")
	}
	if _, exists := r.tools[name]; exists {
		return fmt.Errorf("tools: duplicate tool name %q", name)
	}
	r.tools[name] = t
	return nil
}

// MustRegister は Register と同様ですが、エラー時にパニックします。
// 信頼できる組み込みツールの静的ワイヤリングに便利です。
// 動的にロードされるツールには Register を使用してください。
func (r *Registry) MustRegister(t Tool) {
	if err := r.Register(t); err != nil {
		panic(err)
	}
}

// Get は名前でツールを検索し、見つかったかどうかを返します。
// エージェントのツール実行時に使用されます。
func (r *Registry) Get(name string) (Tool, bool) {
	t, ok := r.tools[name]
	return t, ok
}

// Len は登録されているツールの数を返します。
func (r *Registry) Len() int { return len(r.tools) }

// Names は登録されているツール名をソート順で返します。
// CLI の起動メッセージ表示などに使用されます。
func (r *Registry) Names() []string {
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Definitions はすべての登録済みツールを、モデルが理解できる llm.Tool スキーマに
// 変換して返します。名前で決定論的にソートされます。
//
// この結果は ChatRequest.Tools に設定され、LLM API の tools パラメータとして
// モデルへ送信されます。モデルはこれらの定義を見てツール呼び出しを判断します。
func (r *Registry) Definitions() []llm.Tool {
	defs := make([]llm.Tool, 0, len(r.tools))
	for _, name := range r.Names() {
		t := r.tools[name]
		defs = append(defs, llm.Tool{
			Type: "function",
			Function: llm.FunctionDefinition{
				Name:        t.Name(),
				Description: t.Description(),
				Parameters:  t.Parameters(), // JSON Schema のまま渡す
			},
		})
	}
	return defs
}

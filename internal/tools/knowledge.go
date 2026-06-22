package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/knowledge"
)

// KnowledgeSearch はナレッジベース（リポジトリに配置した Markdown 等）から
// 質問に関連する情報を検索するツールです。モデルはユーザーの質問に答える前に
// このツールを呼び出し、得られた抜粋を根拠として回答を組み立てられます
// （Retrieval-Augmented Generation / 検索拡張生成）。
type KnowledgeSearch struct {
	// Store は検索対象のナレッジを保持します。nil の場合、検索は常に空を返します。
	Store *knowledge.Store
	// TopK は1回の検索で返すチャンクの既定数です。0 の場合は 4 を使います。
	TopK int
}

// Name implements Tool.
func (KnowledgeSearch) Name() string { return "knowledge_search" }

// Description implements Tool.
func (KnowledgeSearch) Description() string {
	return "Search the local knowledge base (Markdown documents shipped with " +
		"this repository) for passages relevant to a question, and return the " +
		"matching excerpts with their source file. Use this whenever the user " +
		"asks about this project, its configuration, or any topic that may be " +
		"documented in the knowledge base, and ground your answer in the results."
}

// Parameters implements Tool.
func (KnowledgeSearch) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"query": {
				"type": "string",
				"description": "The search query or question to look up in the knowledge base."
			},
			"top_k": {
				"type": "integer",
				"description": "Optional maximum number of passages to return (default 4)."
			}
		},
		"required": ["query"]
	}`)
}

type knowledgeSearchArgs struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k"`
}

// Call implements Tool.
func (k KnowledgeSearch) Call(_ context.Context, args json.RawMessage) (string, error) {
	var a knowledgeSearchArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return "", fmt.Errorf("knowledge_search: invalid arguments: %w", err)
	}
	query := strings.TrimSpace(a.Query)
	if query == "" {
		return "", fmt.Errorf("knowledge_search: empty query")
	}
	if k.Store == nil || k.Store.Len() == 0 {
		return "No knowledge base is configured.", nil
	}

	topK := a.TopK
	if topK <= 0 {
		topK = k.TopK
	}

	docs := k.Store.Search(query, topK)
	if len(docs) == 0 {
		return "No relevant passages were found in the knowledge base.", nil
	}

	// 各抜粋を出典付きで整形して返します。モデルが出典を引用できるようにします。
	var b strings.Builder
	fmt.Fprintf(&b, "Found %d relevant passage(s):\n", len(docs))
	for i, d := range docs {
		source, _ := d.Metadata["source"].(string)
		if source == "" {
			source = "unknown"
		}
		fmt.Fprintf(&b, "\n[%d] source: %s\n%s\n", i+1, source, strings.TrimSpace(d.PageContent))
	}
	return strings.TrimRight(b.String(), "\n"), nil
}

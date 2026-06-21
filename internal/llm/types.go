// Package llm は、チャット補完形式の大規模言語モデルAPIに対する
// プロバイダー非依存クライアントを提供します。実際のトランスポートと
// プロトコル処理は langchaingo ライブラリ (github.com/tmc/langchaingo) に
// 委譲されており、同一クライアントで OpenAI・OpenRouter・その他
// OpenAI互換ゲートウェイを対象にできます。このパッケージは langchaingo の
// 汎用LLMインターフェースを、エージェントループが扱う小さな
// リクエスト/レスポンス形状に適合させます。
package llm

import "encoding/json"

// Role は会話内のメッセージを誰が生成したかを識別します。
// OpenAI Chat Completions API の仕様に従い4種類のロールを定義しています。
const (
	// RoleSystem はシステムプロンプト（モデルへの指示）に使用します。
	// 会話の先頭に1件だけ置くのが一般的です。
	RoleSystem = "system"
	// RoleUser はユーザーからの入力メッセージに使用します。
	RoleUser = "user"
	// RoleAssistant はモデルが生成した応答メッセージに使用します。
	// ツール呼び出しのみの場合は Content が空になることもあります。
	RoleAssistant = "assistant"
	// RoleTool はツール実行結果をモデルへ返すメッセージに使用します。
	// ToolCallID フィールドで対応するツール呼び出しと紐付けます。
	RoleTool = "tool"
)

// Message は会話履歴の1エントリを表します。
// OpenAI Chat Completions API の messages 配列の各要素に対応しています。
type Message struct {
	// Role は誰がこのメッセージを生成したかを示します (system/user/assistant/tool)。
	Role string `json:"role"`
	// Content はテキスト本文です。ツール呼び出しのみをリクエストする
	// アシスタントメッセージでは空になる場合があります。
	Content string `json:"content"`
	// Name はツール結果メッセージでツール名をエコーするための任意フィールドです。
	Name string `json:"name,omitempty"`
	// ToolCalls はモデルがツール呼び出しを要求するアシスタントメッセージで設定されます。
	// 複数のツールを同時に呼び出すことも可能です。
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID はツール結果メッセージを対応する呼び出しに紐付けます。
	// モデルが複数のツール呼び出しを同時発行した場合に結果を区別するために使用します。
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// ToolCall は、モデルが特定のツールをJSON引数で呼び出すよう要求する
// リクエストを表します。モデルはアシスタントメッセージの tool_calls 配列に
// この構造体を含めて返します。
type ToolCall struct {
	// ID はこのツール呼び出しの一意識別子です。ツール結果メッセージの
	// ToolCallID と対応させることで、複数の並行ツール呼び出しを管理できます。
	ID string `json:"id"`
	// Type は常に "function" です。将来的な拡張のために明示的に保持しています。
	Type string `json:"type"`
	// Function は呼び出すツールの名前とJSON引数を保持します。
	Function FunctionCall `json:"function"`
}

// FunctionCall はツール名とJSON形式の生引数を保持します。
// Arguments は JSON 文字列（オブジェクト形式）としてモデルから送られてきます。
type FunctionCall struct {
	// Name は呼び出すツールの識別子です。Registry に登録されているツール名と一致する必要があります。
	Name string `json:"name"`
	// Arguments はツールへ渡すパラメータのJSON文字列です。
	// 各ツールの Parameters() が定義するJSON Schemaに従ったオブジェクトになります。
	Arguments string `json:"arguments"`
}

// Tool はモデルが呼び出せる関数の定義を表します。
// Chat Completions API の tools パラメータの各要素に対応しています。
type Tool struct {
	// Type は常に "function" です。
	Type string `json:"type"`
	// Function にツールの名前・説明・パラメータスキーマが含まれます。
	Function FunctionDefinition `json:"function"`
}

// FunctionDefinition はモデルへ公開される1つのツールのスキーマです。
// モデルはこの情報をもとに、どのツールをいつ呼び出すかを判断します。
type FunctionDefinition struct {
	// Name はツールの識別子で、ToolCall.Function.Name と対応します。
	Name string `json:"name"`
	// Description はモデルへのツール使用ガイダンスです。
	// 「いつ・なぜこのツールを使うか」を説明することで、モデルの判断精度が向上します。
	Description string `json:"description"`
	// Parameters はツール引数のJSON Schemaです。
	// モデルはこのスキーマに従ってArguments JSONを生成します。
	Parameters json.RawMessage `json:"parameters"`
}

// ChatRequest は単一のチャット補完リクエストを表します。
// LLM Client の Chat メソッドに渡します。
type ChatRequest struct {
	// Model は使用するモデル名です。空の場合はクライアントのデフォルトモデルが使用されます。
	Model string `json:"model"`
	// Messages はシステムプロンプトを含む会話履歴の全体です。
	Messages []Message `json:"messages"`
	// Tools はモデルが呼び出せるツールの定義一覧です。空の場合はツール呼び出し機能が無効になります。
	Tools []Tool `json:"tools,omitempty"`
	// ToolChoice はツール使用ポリシーを制御します ("auto", "none", "required")。
	// "auto" はモデルが自律的にツール使用を判断します。
	ToolChoice string `json:"tool_choice,omitempty"`
	// Temperature はサンプリング温度です (0.0〜2.0)。
	// 低いほど決定論的、高いほど多様な応答になります。nil の場合はモデルデフォルト。
	Temperature *float64 `json:"temperature,omitempty"`
	// MaxTokens は生成する最大トークン数です。0 はモデルデフォルトを使用します。
	MaxTokens int `json:"max_tokens,omitempty"`
}

// ChatResponse はチャット補完リクエストのデコード済みレスポンスです。
type ChatResponse struct {
	// Choices は生成された補完候補の一覧です。通常は1件のみです。
	Choices []Choice `json:"choices"`
	// Usage はプロバイダーが報告する場合のトークン使用量です。
	Usage Usage `json:"usage"`
}

// Choice は単一の補完候補を表します。
// 通常 Choices[0] を使用します。
type Choice struct {
	// Index は複数の choices がある場合のインデックスです。
	Index int `json:"index"`
	// Message にモデルが生成したアシスタントメッセージが含まれます。
	// ToolCalls フィールドが設定されている場合はツール呼び出しが要求されています。
	Message Message `json:"message"`
	// FinishReason は生成が停止した理由です ("stop", "tool_calls", "length" など)。
	FinishReason string `json:"finish_reason"`
}

// Usage はリクエストのトークン使用量を報告します。プロバイダーが提供する場合のみ有効です。
// コスト管理やレート制限の監視に使用できます。
type Usage struct {
	// PromptTokens はリクエスト（会話履歴＋ツール定義）に使用したトークン数です。
	PromptTokens int `json:"prompt_tokens"`
	// CompletionTokens はモデルが生成したレスポンスのトークン数です。
	CompletionTokens int `json:"completion_tokens"`
	// TotalTokens は PromptTokens + CompletionTokens の合計です。
	TotalTokens int `json:"total_tokens"`
}

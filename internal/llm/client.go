package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

// Provider は既知のLLMゲートウェイを識別します。
// 各プロバイダーにはデフォルトのエンドポイントURLが紐付いています。
type Provider string

const (
	// ProviderOpenAI は公式 OpenAI API をターゲットにします。
	ProviderOpenAI Provider = "openai"
	// ProviderOpenRouter は OpenRouter をターゲットにします。
	// OpenRouter は多数のモデルベンダーを1つの OpenAI 互換 API で束ねるゲートウェイです。
	ProviderOpenRouter Provider = "openrouter"
)

// defaultBaseURLs は既知プロバイダーとそのチャット補完ベースURLのマッピングです。
// AGENT_BASE_URL 環境変数でオーバーライドできます。
var defaultBaseURLs = map[Provider]string{
	ProviderOpenAI:     "https://api.openai.com/v1",
	ProviderOpenRouter: "https://openrouter.ai/api/v1",
}

// DefaultBaseURL は既知プロバイダーの標準ベースURLを返します。
// 未知のプロバイダーの場合は空文字列を返します。
func DefaultBaseURL(p Provider) string {
	return defaultBaseURLs[Provider(strings.ToLower(string(p)))]
}

// Config は Client の設定を保持します。
// APIKey と Provider または BaseURL のいずれかは必須です。それ以外はデフォルト値が使用されます。
type Config struct {
	// Provider は組み込みエンドポイントを選択します。BaseURL が設定されている場合は無視されます。
	Provider Provider
	// BaseURL はエンドポイントを完全にオーバーライドします。
	// 自己ホスト型モデルや OpenAI 互換プロキシゲートウェイに便利です。
	BaseURL string
	// APIKey は Authorization: Bearer ヘッダーを通じてリクエストを認証します。
	APIKey string
	// Model はリクエストにモデル名が指定されていない場合に使用されるデフォルトモデル名です。
	Model string
	// Timeout は単一の HTTP リクエストの上限時間です。デフォルトは120秒です。
	Timeout time.Duration
	// HTTPClient を使うとカスタムトランスポートを注入できます（テストなどで有用です）。
	HTTPClient *http.Client
	// Headers はすべてのリクエストに付加される追加ヘッダーです。
	// OpenRouter では HTTP-Referer と X-Title の設定が推奨されています。
	Headers map[string]string
}

// Client は langchaingo の LLM をラップし、エージェントが使用する
// リクエスト/レスポンス形状に適合させます。
// ライブラリが OpenAI ワイヤープロトコル・認証・トランスポートを処理します。
type Client struct {
	// llm は langchaingo が提供する OpenAI LLM インスタンスです。
	// 実際の HTTP 通信と OpenAI プロトコルの処理はこのオブジェクトが担当します。
	llm *openai.LLM
	// model はこのクライアントのデフォルトモデル名です。
	model string
}

// NewClient は cfg から Client を構築します。
// プロバイダーのデフォルト値を適用し、必須フィールドの存在を検証します。
func NewClient(cfg Config) (*Client, error) {
	// BaseURL の末尾スラッシュを除去して正規化します。
	baseURL := strings.TrimRight(cfg.BaseURL, "/")
	if baseURL == "" {
		// BaseURL が未設定の場合はプロバイダーのデフォルトURLを使用します。
		baseURL = DefaultBaseURL(cfg.Provider)
	}
	if baseURL == "" {
		return nil, fmt.Errorf("llm: no base URL: set BaseURL or a known Provider (got %q)", cfg.Provider)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("llm: APIKey is required")
	}

	// タイムアウトが未設定の場合は120秒をデフォルトとして使用します。
	// LLMのレスポンスは長文生成時に時間がかかるため、余裕のある値を設定しています。
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: timeout}
	}
	// 追加ヘッダーがある場合、トランスポートをラップして
	// langchaingo が発行するすべてのリクエストにヘッダーを付加します。
	// （例: OpenRouter のランキングヘッダー HTTP-Referer, X-Title）
	if len(cfg.Headers) > 0 {
		base := httpClient.Transport
		if base == nil {
			base = http.DefaultTransport
		}
		clone := *httpClient
		clone.Transport = &headerTransport{base: base, headers: cfg.Headers}
		httpClient = &clone
	}

	// langchaingo の OpenAI クライアントを設定します。
	// WithToken: APIキー認証ヘッダーの設定
	// WithBaseURL: エンドポイントURL（OpenRouter等の互換APIにも対応）
	// WithHTTPClient: カスタムHTTPクライアント（追加ヘッダー・タイムアウト設定済み）
	opts := []openai.Option{
		openai.WithToken(cfg.APIKey),
		openai.WithBaseURL(baseURL),
		openai.WithHTTPClient(httpClient),
	}
	if cfg.Model != "" {
		opts = append(opts, openai.WithModel(cfg.Model))
	}

	model, err := openai.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("llm: init langchaingo client: %w", err)
	}

	return &Client{llm: model, model: cfg.Model}, nil
}

// Model はクライアントのデフォルトモデル名を返します。
func (c *Client) Model() string { return c.model }

// Chat は単一のチャット補完リクエストを実行し、デコードされたレスポンスを返します。
// req.Model が空の場合はクライアントのデフォルトモデルが使用されます。
//
// 処理の流れ:
//  1. メッセージを langchaingo 形式に変換
//  2. ツール定義を langchaingo 形式に変換
//  3. langchaingo 経由で LLM API を呼び出す
//  4. レスポンスをエージェント形式に変換して返す
func (c *Client) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	model := req.Model
	if model == "" {
		model = c.model
	}
	if model == "" {
		return nil, fmt.Errorf("llm: no model specified")
	}

	// エージェントのメッセージ履歴を langchaingo の MessageContent 形式に変換します。
	// この変換によりシステム/ユーザー/アシスタント/ツール の各ロールが適切に処理されます。
	messages, err := toLangChainMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	// langchaingo の呼び出しオプションを構築します。
	callOpts := []llms.CallOption{llms.WithModel(model)}
	if req.Temperature != nil {
		callOpts = append(callOpts, llms.WithTemperature(*req.Temperature))
	}
	if req.MaxTokens > 0 {
		callOpts = append(callOpts, llms.WithMaxTokens(req.MaxTokens))
	}
	// ツール定義がある場合は Function Calling 機能を有効にします。
	// モデルはツール定義を見て、どのツールをいつ呼び出すかを自律的に判断します。
	if len(req.Tools) > 0 {
		tools, err := toLangChainTools(req.Tools)
		if err != nil {
			return nil, err
		}
		callOpts = append(callOpts, llms.WithTools(tools))
		if req.ToolChoice != "" {
			// "auto": モデルが自律判断、"none": ツール使用禁止、"required": 必ずツールを使用
			callOpts = append(callOpts, llms.WithToolChoice(req.ToolChoice))
		}
	}

	// langchaingo 経由でLLM APIを呼び出します。
	// 内部的には OpenAI Chat Completions API へのHTTP POSTリクエストが発行されます。
	resp, err := c.llm.GenerateContent(ctx, messages, callOpts...)
	if err != nil {
		return nil, fmt.Errorf("llm: request failed: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("llm: response contained no choices")
	}

	// langchaingo のレスポンス形式をエージェントが使用する ChatResponse 形式に変換します。
	return fromLangChainResponse(resp), nil
}

// toLangChainMessages はエージェントの会話履歴を langchaingo のメッセージ表現に変換します。
// ツール呼び出しとツール結果を適切に変換することで、モデルが次のリクエストで
// 正しい形式の複数ターン履歴を参照できます。
//
// ロール別の変換処理:
//   - system    → ChatMessageTypeSystem
//   - user      → ChatMessageTypeHuman
//   - assistant → ChatMessageTypeAI (ToolCall が含まれる場合も対応)
//   - tool      → ChatMessageTypeTool (ToolCallResponse として格納)
func toLangChainMessages(in []Message) ([]llms.MessageContent, error) {
	out := make([]llms.MessageContent, 0, len(in))
	for _, m := range in {
		switch m.Role {
		case RoleSystem:
			// システムプロンプト: モデルへの全体的な指示と行動方針
			out = append(out, llms.TextParts(llms.ChatMessageTypeSystem, m.Content))
		case RoleUser:
			// ユーザーメッセージ: エンドユーザーからの入力
			out = append(out, llms.TextParts(llms.ChatMessageTypeHuman, m.Content))
		case RoleAssistant:
			// アシスタントメッセージ: モデルの応答
			// テキスト内容とツール呼び出しの両方を含む可能性があります
			parts := []llms.ContentPart{}
			if m.Content != "" {
				parts = append(parts, llms.TextContent{Text: m.Content})
			}
			// ツール呼び出しリクエストをlangchaingo形式に変換します
			for _, tc := range m.ToolCalls {
				parts = append(parts, llms.ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					FunctionCall: &llms.FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				})
			}
			out = append(out, llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: parts})
		case RoleTool:
			// ツール結果メッセージ: ツール実行後の結果をモデルへ返す
			// ToolCallID でどのツール呼び出しに対する結果かを示します
			out = append(out, llms.MessageContent{
				Role: llms.ChatMessageTypeTool,
				Parts: []llms.ContentPart{llms.ToolCallResponse{
					ToolCallID: m.ToolCallID,
					Name:       m.Name,
					Content:    m.Content,
				}},
			})
		default:
			return nil, fmt.Errorf("llm: unknown message role %q", m.Role)
		}
	}
	return out, nil
}

// toLangChainTools はエージェントのツール定義を langchaingo のツールスキーマに変換します。
// JSON Schema パラメータは langchaingo が any 型を期待するため、生のバイト列からデコードします。
func toLangChainTools(in []Tool) ([]llms.Tool, error) {
	out := make([]llms.Tool, 0, len(in))
	for _, t := range in {
		// JSON Schema を生のバイト列から any 型にデコードします。
		// langchaingo はこれをAPIリクエスト時に再度JSONエンコードしてモデルへ送信します。
		var params any
		if len(t.Function.Parameters) > 0 {
			if err := json.Unmarshal(t.Function.Parameters, &params); err != nil {
				return nil, fmt.Errorf("llm: tool %q has invalid parameters schema: %w", t.Function.Name, err)
			}
		}
		out = append(out, llms.Tool{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  params,
			},
		})
	}
	return out, nil
}

// fromLangChainResponse は langchaingo の最初のコンテンツチョイスを
// エージェントの ChatResponse 形状にマッピングします。
// リクエストされたツール呼び出しも適切に再構築します。
func fromLangChainResponse(resp *llms.ContentResponse) *ChatResponse {
	// 通常は最初のチョイスのみを使用します（n=1がデフォルト）
	choice := resp.Choices[0]

	// アシスタントメッセージを構築します
	msg := Message{Role: RoleAssistant, Content: choice.Content}
	// ツール呼び出しリクエストを langchaingo 形式からエージェント形式に変換します
	for _, tc := range choice.ToolCalls {
		call := ToolCall{ID: tc.ID, Type: tc.Type}
		// Type が空の場合は "function" をデフォルトとして設定します
		if call.Type == "" {
			call.Type = "function"
		}
		if tc.FunctionCall != nil {
			call.Function = FunctionCall{
				Name:      tc.FunctionCall.Name,
				Arguments: tc.FunctionCall.Arguments,
			}
		}
		msg.ToolCalls = append(msg.ToolCalls, call)
	}

	out := &ChatResponse{
		Choices: []Choice{{Message: msg, FinishReason: choice.StopReason}},
	}
	// トークン使用量をプロバイダーの GenerationInfo から取得します
	out.Usage = usageFromGenerationInfo(choice.GenerationInfo)
	return out
}

// usageFromGenerationInfo はプロバイダーがチョイスの GenerationInfo で報告する場合に
// トークン使用量を抽出します。フィールドがない場合はゼロを返します。
func usageFromGenerationInfo(info map[string]any) Usage {
	var u Usage
	if info == nil {
		return u
	}
	// GenerationInfo の値は int, int64, float64 のいずれかで返ってくる可能性があるため
	// 型スイッチで安全に int に変換します
	asInt := func(v any) int {
		switch n := v.(type) {
		case int:
			return n
		case int64:
			return int(n)
		case float64:
			return int(n)
		}
		return 0
	}
	u.PromptTokens = asInt(info["PromptTokens"])
	u.CompletionTokens = asInt(info["CompletionTokens"])
	u.TotalTokens = asInt(info["TotalTokens"])
	return u
}

// headerTransport はすべての送信リクエストに固定のヘッダーセットを注入します。
// http.RoundTripper インターフェースを実装することで、langchaingo が使用する
// HTTPクライアントのトランスポート層に透過的に組み込まれます。
type headerTransport struct {
	base    http.RoundTripper
	headers map[string]string
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// 呼び出し元が再利用する可能性があるリクエストを変更しないようにクローンします。
	// http.Request はミュータブルであるため、直接変更するとレース条件が発生します。
	clone := req.Clone(req.Context())
	for k, v := range t.headers {
		clone.Header.Set(k, v)
	}
	return t.base.RoundTrip(clone)
}

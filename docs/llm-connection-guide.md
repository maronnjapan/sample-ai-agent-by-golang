# LLM 接続処理 詳細解説ガイド

このドキュメントは、本リポジトリの **LLM（大規模言語モデル）との接続処理** が
どのように実装されているかを、ソースコードを引用しながら日本語で細かく解説する
ものです。エージェントが「ユーザー入力を受け取り → LLM に問い合わせ → ツールを
実行 → 結果を LLM に返す → 最終回答を得る」という一連の流れを、各レイヤーごとに
追っていきます。

---

## 0. 最初に: 「LangChain」について

このリポジトリのブランチ名やタスクには **LangChain** という単語が出てきますが、
**本実装は LangChain（および Go 版の [langchaingo](https://github.com/tmc/langchaingo)）
を一切使用していません**。`go.mod` を見ると依存はゼロで、Go の標準ライブラリだけ
で構成されています。

```go
// go.mod
module github.com/maronnjapan/sample-ai-agent-by-golang

go 1.24
```

第三者パッケージの記載が一行もないことが、依存ゼロであることの証拠です。

ではなぜ LangChain が引き合いに出されるのか? それは、本実装が LangChain が提供する
のと **同じ概念**（チャットモデルの抽象化、ツール／ファンクションコーリング、
エージェントの reason–act ループ）を、外部ライブラリに頼らず **自前で実装** している
からです。LangChain を「使う」のではなく、「LangChain がやっていることを Go の
標準ライブラリだけで再現する」プロジェクトだと理解するのが正確です。

そのため本ガイドでは、各実装が LangChain でいうところのどの概念に対応するのかを
随時補足しながら解説します。

### 全体アーキテクチャ

```
cmd/agent           CLI: REPL + ワンショット実行、ツール実行トレースの描画
internal/config     環境変数 / .env からの設定読み込みとプロバイダ既定値
internal/llm        OpenAI 互換 chat-completions クライアントと型定義  ← 接続処理の核
internal/agent      reason–act ループ（モデル ⇄ ツール）とイベント通知
internal/tools      Tool インターフェース、レジストリ、組み込みツール群
```

LLM への「接続処理」そのものは **`internal/llm`** パッケージに集約されています。
それを呼び出して会話を駆動するのが **`internal/agent`** です。以下、データの流れ
（設定 → クライアント生成 → リクエスト送信 → ループ）に沿って読み解きます。

---

## 1. 設定の読み込み (`internal/config`)

LLM に接続するには、まず「どのプロバイダの」「どのエンドポイントに」「どの API
キーで」「どのモデルを」使うのかを決める必要があります。これを担うのが
`config.Load()` です。

```go
// internal/config/config.go
func Load() (*Config, error) {
	_ = loadDotEnv(".env") // ベストエフォート。ファイルが無くてもエラーにしない

	provider := llm.Provider(strings.ToLower(getenv("AGENT_PROVIDER", string(llm.ProviderOpenAI))))

	cfg := &Config{
		Provider:     provider,
		BaseURL:      os.Getenv("AGENT_BASE_URL"),
		Model:        os.Getenv("AGENT_MODEL"),
		SystemPrompt: os.Getenv("AGENT_SYSTEM_PROMPT"),
		Temperature:  0.7,
		MaxSteps:     10,
		ExtraHeaders: map[string]string{},
	}

	cfg.APIKey = resolveAPIKey(provider)
	if cfg.Model == "" {
		cfg.Model = defaultModel(provider)
	}
	// ... AGENT_TEMPERATURE / AGENT_MAX_STEPS のパース ...
}
```

### ポイント解説

- **`.env` の自動読み込み**: `loadDotEnv(".env")` が最初に呼ばれ、`.env` ファイルの
  `KEY=VALUE` を環境変数として読み込みます。ただし **既に実体の環境変数に設定済みの
  値は上書きしない**（後述）ため、本番環境の設定が `.env` で潰されることはありません。

- **プロバイダの選択**: `AGENT_PROVIDER` が `openai`（既定）か `openrouter` かを決めます。
  これは接続先エンドポイントの既定値を切り替えるためのものです。

- **API キーのフォールバック**: `resolveAPIKey` は汎用の `AGENT_API_KEY` を優先し、
  無ければプロバイダ慣習の環境変数（`OPENAI_API_KEY` / `OPENROUTER_API_KEY`）に
  フォールバックします。

  ```go
  func resolveAPIKey(p llm.Provider) string {
  	if v := os.Getenv("AGENT_API_KEY"); v != "" {
  		return v
  	}
  	return os.Getenv(apiKeyEnvName(p))
  }
  ```

- **モデルの既定値**: モデル名が未指定なら、プロバイダごとの既定モデルを当てます
  （OpenAI なら `gpt-4o-mini`、OpenRouter なら `openai/gpt-4o-mini`）。

- **OpenRouter 向け追加ヘッダ**: OpenRouter はアプリ識別用に `HTTP-Referer` と
  `X-Title` ヘッダを推奨しているため、該当環境変数があれば `ExtraHeaders` に積みます。
  これらは後でリクエストごとに付与されます。

### `.env` パーサの安全設計

```go
// internal/config/config.go
func loadDotEnv(path string) error {
	// ... 1 行ずつ読み、コメント行・空行をスキップ ...
	key = strings.TrimSpace(key)
	value = strings.Trim(strings.TrimSpace(value), `"'`)
	// 既に実環境に設定済みの変数は上書きしない
	if _, exists := os.LookupEnv(key); !exists {
		_ = os.Setenv(key, value)
	}
}
```

`os.LookupEnv` で存在チェックしてから `Setenv` することで、**実環境の設定を優先**
するという行儀の良い挙動になっています。LangChain の `dotenv` 相当を最小実装した
ものと考えてよいでしょう。

---

## 2. LLM クライアント (`internal/llm`) — 接続処理の核心

ここが本題です。`internal/llm` パッケージは **OpenAI の "chat completions" ワイヤ
フォーマット** を喋る、依存ゼロの最小クライアントです。OpenAI のフォーマットは
OpenRouter をはじめ多くのゲートウェイ（Azure OpenAI, Groq, Together, ローカルの
`llama.cpp` サーバ等）が実装しているため、**1 つのクライアントで複数プロバイダに
対応** できます。これは LangChain の `ChatOpenAI` が担う役割に相当します。

### 2-1. プロバイダとエンドポイントの定義

```go
// internal/llm/client.go
type Provider string

const (
	ProviderOpenAI     Provider = "openai"
	ProviderOpenRouter Provider = "openrouter"
)

// 既知プロバイダ → chat-completions のベース URL
var defaultBaseURLs = map[Provider]string{
	ProviderOpenAI:     "https://api.openai.com/v1",
	ProviderOpenRouter: "https://openrouter.ai/api/v1",
}

func DefaultBaseURL(p Provider) string {
	return defaultBaseURLs[Provider(strings.ToLower(string(p)))]
}
```

プロバイダ名と既定ベース URL のマッピングを持つだけのシンプルな仕組みです。
`BaseURL` を明示すれば任意の OpenAI 互換エンドポイントに向けられます。

### 2-2. クライアントの構築 — `NewClient`

```go
// internal/llm/client.go
func NewClient(cfg Config) (*Client, error) {
	baseURL := strings.TrimRight(cfg.BaseURL, "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL(cfg.Provider)
	}
	if baseURL == "" {
		return nil, fmt.Errorf("llm: no base URL: set BaseURL or a known Provider (got %q)", cfg.Provider)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("llm: APIKey is required")
	}

	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: timeout}
	}

	return &Client{
		baseURL: baseURL,
		apiKey:  cfg.APIKey,
		model:   cfg.Model,
		headers: cfg.Headers,
		http:    httpClient,
	}, nil
}
```

### ポイント解説

- **`BaseURL` 優先、無ければプロバイダ既定**: `BaseURL` が空のときだけ
  `DefaultBaseURL(cfg.Provider)` を使います。どちらも空なら設定ミスとして早期に
  エラーを返します（**fail fast**）。

- **必須項目の検証**: `APIKey` が無ければここで弾きます。実行時に初めて 401 で
  気づく、といった事態を防ぎます。

- **タイムアウトの既定**: 未指定なら 120 秒。1 リクエストの上限時間です。

- **HTTP クライアントの注入**: `cfg.HTTPClient` を差し替え可能にしている点が重要で、
  **テスト時にモックトランスポートを注入** できます（実際にクライアントのテストで
  使われています）。これにより、ネットワークに出ずに接続処理を検証できます。

### 2-3. チャット補完リクエストの送信 — `Chat`

クライアントの心臓部です。会話履歴を受け取り、HTTP POST で
`/chat/completions` を叩き、レスポンスをデコードして返します。

```go
// internal/llm/client.go
func (c *Client) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	// (1) モデル名の補完
	if req.Model == "" {
		req.Model = c.model
	}
	if req.Model == "" {
		return nil, fmt.Errorf("llm: no model specified")
	}

	// (2) リクエストボディを JSON 化
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal request: %w", err)
	}

	// (3) HTTP リクエストの構築（context 付き）
	url := c.baseURL + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("llm: build request: %w", err)
	}

	// (4) 認証・追加ヘッダの付与
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	for k, v := range c.headers {
		httpReq.Header.Set(k, v)
	}

	// (5) 送信
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llm: request failed: %w", err)
	}
	defer resp.Body.Close()

	// (6) レスポンスボディの読み取り
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("llm: read response: %w", err)
	}

	// (7) JSON デコードと多段のエラーハンドリング
	var out ChatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("llm: decode response (status %d): %w: %s", resp.StatusCode, err, truncate(string(raw), 500))
	}
	if out.Error != nil {
		return nil, fmt.Errorf("llm: provider error (status %d): %w", resp.StatusCode, out.Error)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("llm: unexpected status %d: %s", resp.StatusCode, truncate(string(raw), 500))
	}
	if len(out.Choices) == 0 {
		return nil, fmt.Errorf("llm: response contained no choices")
	}
	return &out, nil
}
```

### 番号ごとの詳細解説

1. **モデル名の補完**: リクエストにモデルが無ければクライアント既定を使い、
   それでも空なら明示的にエラー。「どのモデルに投げているか不明」を防ぎます。

2. **リクエストの JSON 化**: `ChatRequest` 構造体を `encoding/json` で直列化します。
   構造体タグ（後述）が OpenAI のフィールド名（`messages`, `tools` 等）に対応します。

3. **`context.Context` 付きリクエスト**: `http.NewRequestWithContext` を使うことで、
   呼び出し側が `ctx` をキャンセルすれば **送信中の HTTP も即座に中断** されます。
   CLI 側では Ctrl-C（SIGINT）でこの ctx をキャンセルしており、ハングを防ぎます。

4. **ヘッダの付与**:
   - `Content-Type: application/json`
   - `Authorization: Bearer <API キー>` ← OpenAI 互換 API の認証方式
   - `c.headers` の追加ヘッダ（OpenRouter の `HTTP-Referer` / `X-Title` 等）を後付け。

5. **送信**: 注入された `*http.Client` で実行。テストではここがモックに差し替わります。

6. **ボディの全読み**: `io.ReadAll` で一旦全て読み込みます。これにより、JSON で
   なかった場合（後述）に **生のボディをエラーメッセージに含められる** 利点があります。

7. **堅牢なエラーハンドリング（多段防御）**: ここが実装の丁寧なところです。
   - まず JSON デコードを試みる。**失敗したら生ボディを 500 文字まで切り詰めて
     エラーに含める**（`truncate`）。ゲートウェイが HTML のエラーページを返すような
     ケースでも、原因が握り潰されません。
   - `out.Error`（プロバイダのエラーエンベロープ）があれば、それを優先して返す。
   - HTTP ステータスが 2xx 以外なら異常として扱う。
   - `Choices` が空（=モデルの回答が無い）なら明示的にエラー。

```go
// internal/llm/client.go
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
```

`truncate` は巨大なエラーボディでログを溢れさせないためのユーティリティです。

---

## 3. ワイヤフォーマットの型定義 (`internal/llm/types.go`)

`Chat` がやり取りする JSON の構造を定義しているのが `types.go` です。OpenAI の
chat-completions スキーマを Go の構造体で表現しています。LangChain でいう
`Message` / `ChatResult` / `Tool` スキーマに相当します。

### 3-1. ロール定数とメッセージ

```go
// internal/llm/types.go
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}
```

会話は `Message` の配列で表現します。4 つのロールがあります。

- `system`: エージェントの振る舞いを指示するシステムプロンプト。
- `user`: ユーザーの入力。
- `assistant`: モデルの応答。**本文だけでなく、ツール呼び出し要求（`ToolCalls`）も
  ここに乗る**ことがあります。
- `tool`: ツール実行結果。`ToolCallID` でどの呼び出しに対する結果かを紐付けます。

`omitempty` タグにより、空のフィールドは JSON に出力されません。例えば通常の
ユーザーメッセージには `tool_calls` は付きません。

### 3-2. ツール（ファンクションコーリング）関連の型

```go
// internal/llm/types.go
// モデルが「このツールをこの引数で呼んで」と要求するときの型
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON 文字列（モデルが生成）
}

// 「こういうツールが使えるよ」とモデルに広告するときの型
type Tool struct {
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`
}

type FunctionDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"` // JSON Schema
}
```

ここで **2 方向** の型がある点を押さえてください。

- `Tool` / `FunctionDefinition`: アプリ → モデルへの **「使えるツールの宣言」**。
  `Parameters` は引数の JSON Schema を `json.RawMessage`（生 JSON）で持ちます。
- `ToolCall` / `FunctionCall`: モデル → アプリへの **「このツールを呼びたい」要求**。
  `Arguments` は **モデルが生成した JSON 文字列** です（構造体ではなく文字列なのは、
  モデルが任意のスキーマで引数を組み立てるため）。

### 3-3. リクエストとレスポンス

```go
// internal/llm/types.go
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Tools       []Tool    `json:"tools,omitempty"`
	ToolChoice  string    `json:"tool_choice,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

type ChatResponse struct {
	ID      string    `json:"id"`
	Model   string    `json:"model"`
	Choices []Choice  `json:"choices"`
	Usage   Usage     `json:"usage"`
	Error   *APIError `json:"error,omitempty"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}
```

### 設計上の注目点

- **`Temperature *float64`（ポインタ）**: ポインタにしているのは、`omitempty` と
  組み合わせて **「未設定（nil）」と「0 を明示」を区別** するためです。値型 `float64`
  だと 0 が「未設定」と区別できず、温度 0 を送りたいケースで困ります。

- **`Error *APIError`**: OpenAI 互換プロバイダはエラーを `{"error": {...}}` の形で
  返します。これをレスポンス内に取り込むことで、`Chat` がエラーを判定できます。

```go
// internal/llm/types.go
type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    any    `json:"code"`
}

func (e *APIError) Error() string {
	if e == nil {
		return ""
	}
	if e.Type != "" {
		return e.Type + ": " + e.Message
	}
	return e.Message
}
```

`APIError` は Go の `error` インターフェースを実装しているため、`fmt.Errorf("...%w", out.Error)`
でそのままラップできます。

---

## 4. エージェントの reason–act ループ (`internal/agent`)

`internal/llm` が「1 回の HTTP 往復」を担うのに対し、`internal/agent` は
**「モデルに問い合わせ → ツール実行 → 結果を返す → 最終回答まで繰り返す」** という
ループ全体を担います。これは LangChain の `AgentExecutor` に相当する部分です。

### 4-1. エージェントの構造と生成

```go
// internal/agent/agent.go
type Agent struct {
	client   *llm.Client
	registry *tools.Registry

	systemPrompt string
	maxSteps     int
	temperature  *float64
}

func New(client *llm.Client, registry *tools.Registry, opts ...Option) *Agent {
	a := &Agent{
		client:       client,
		registry:     registry,
		systemPrompt: DefaultSystemPrompt,
		maxSteps:     10,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}
```

`Agent` は **`*llm.Client`（接続）** と **`*tools.Registry`（ツール集合）** という
2 つの抽象だけに依存します。これにより各部品を独立してテスト可能にしています。
`Option`（関数オプションパターン）でシステムプロンプト・最大ステップ数・温度を
カスタマイズできます。

### 4-2. 会話の保持

```go
// internal/agent/agent.go
type Conversation struct {
	Messages []llm.Message
}

func (a *Agent) NewConversation() *Conversation {
	return &Conversation{
		Messages: []llm.Message{{Role: llm.RoleSystem, Content: a.systemPrompt}},
	}
}
```

会話はシステムプロンプトを先頭に置いた `Message` 配列として保持されます。REPL では
この `Conversation` を使い回すことで **マルチターンの文脈** を維持します。

### 4-3. ループ本体 — `Run`

接続処理を理解する上で最重要のメソッドです。

```go
// internal/agent/agent.go
func (a *Agent) Run(ctx context.Context, conv *Conversation, userInput string, obs Observer) (string, error) {
	if conv == nil {
		return "", fmt.Errorf("agent: conversation is nil")
	}
	// (A) ユーザー入力を会話履歴に追加
	conv.Messages = append(conv.Messages, llm.Message{Role: llm.RoleUser, Content: userInput})

	// (B) 登録済みツールを LLM 向けスキーマに変換
	toolDefs := a.registry.Definitions()

	// (C) reason–act ループ。maxSteps 回まで往復する
	for step := 0; step < a.maxSteps; step++ {
		req := llm.ChatRequest{
			Messages:    conv.Messages,
			Temperature: a.temperature,
		}
		if len(toolDefs) > 0 {
			req.Tools = toolDefs
			req.ToolChoice = "auto" // ツールを使うかはモデルに委ねる
		}

		// (D) LLM へ問い合わせ（ここで internal/llm の Chat が呼ばれる）
		resp, err := a.client.Chat(ctx, req)
		if err != nil {
			return "", fmt.Errorf("agent: chat request: %w", err)
		}

		// (E) アシスタントの応答を履歴に「そのまま」追加
		msg := resp.Choices[0].Message
		conv.Messages = append(conv.Messages, msg)

		// (F) ツール呼び出しが無ければ、それが最終回答
		if len(msg.ToolCalls) == 0 {
			emit(obs, Event{Type: EventFinalAnswer, Text: msg.Content})
			return msg.Content, nil
		}

		// (G) 要求された全ツールを実行し、結果を tool メッセージとして追加
		for _, tc := range msg.ToolCalls {
			result := a.invokeTool(ctx, tc, obs)
			conv.Messages = append(conv.Messages, llm.Message{
				Role:       llm.RoleTool,
				Name:       tc.Function.Name,
				ToolCallID: tc.ID,
				Content:    result,
			})
		}
	}

	// (H) 上限ステップに達しても終わらなかった場合
	return "", fmt.Errorf("agent: exceeded max steps (%d) without a final answer", a.maxSteps)
}
```

### ステップごとの詳細解説

- **(A) ユーザー入力の追加**: 新しいユーザー発話を履歴末尾に積みます。

- **(B) ツール定義の生成**: レジストリが各ツールを `llm.Tool` スキーマ（名前・説明・
  引数の JSON Schema）に変換します。これが LLM に「使えるツール一覧」として渡されます。

- **(C) ループと上限**: `maxSteps`（既定 10）で往復回数を制限。**ツールの無限ループ
  暴走を防ぐガード** です。

- **(D) LLM 問い合わせ**: ここで `internal/llm` の `Chat` が呼ばれ、第 2 章で見た
  HTTP 送信が実行されます。`ToolChoice: "auto"` は「ツールを使うか自分で判断してよい」
  という指示です。

- **(E) 応答を「そのまま」履歴へ**: ここが地味に重要です。モデルの応答（本文や
  `tool_calls`）を **改変せず verbatim で履歴に追加** します。次の往復でプロバイダに
  整合の取れた履歴を見せるため、ツール呼び出しメッセージとその結果メッセージは
  ペアで正しく並んでいる必要があります。

- **(F) 終了条件**: `ToolCalls` が空 = モデルがもうツールを必要としていない =
  これが最終回答。`EventFinalAnswer` を通知して回答文字列を返します。

- **(G) ツール実行**: モデルが要求した各ツールを `invokeTool` で実行し、結果を
  `role: tool` メッセージとして履歴に追加。`ToolCallID` で呼び出しと結果を紐付けます。
  次のループ (D) でこの結果を含めて再度モデルに問い合わせます。

- **(H) ステップ超過**: 上限内に最終回答が出なければエラー。

この (D)→(F)/(G)→(D)... の繰り返しが、いわゆる **reason–act（推論と行動）ループ**
であり、ツール対応エージェントの本質です。

### 4-4. ツール実行の詳細 — `invokeTool`

```go
// internal/agent/agent.go
func (a *Agent) invokeTool(ctx context.Context, tc llm.ToolCall, obs Observer) string {
	name := tc.Function.Name
	rawArgs := json.RawMessage(tc.Function.Arguments)
	emit(obs, Event{Type: EventToolCall, ToolName: name, ToolArgs: tc.Function.Arguments})

	tool, ok := a.registry.Get(name)
	if !ok {
		err := fmt.Errorf("unknown tool %q", name)
		emit(obs, Event{Type: EventToolError, ToolName: name, Err: err})
		return "Error: " + err.Error()
	}

	result, err := tool.Call(ctx, rawArgs)
	if err != nil {
		emit(obs, Event{Type: EventToolError, ToolName: name, Err: err})
		return "Error: " + err.Error()
	}

	emit(obs, Event{Type: EventToolResult, ToolName: name, ToolResult: result})
	return result
}
```

### 重要な設計判断: ツールのエラーは「Go の error」ではなく「文字列」で返す

`invokeTool` は、未知ツールやツール実行失敗のときに **`return "Error: ..."` と文字列を
返す**（Go の `error` を上に伝播させない）点に注目してください。これは
コメントにも明記されている意図的な設計です。

> ツールエラーを（Go の error ではなく）テキストで返すことで、モデルが「何が
> 失敗したか」を見て自力でリカバリできる。Run 全体を中断させない。

つまり、ツールが失敗しても **その失敗内容をモデルに伝え、別の方法を試させる** という、
エージェントとして自然な回復挙動を実現しています。

### 4-5. 進捗の通知 — Observer パターン

```go
// internal/agent/agent.go
type Observer func(Event)

func emit(obs Observer, e Event) {
	if obs != nil {
		obs(e)
	}
}
```

`Run` の途中で起きること（ツール呼び出し・結果・エラー・最終回答）を `Event` として
コールバックに流します。これにより、**エージェント本体は特定のフロントエンド
（CLI / テスト / UI）を知らずに済み**、関心を分離できます。CLI 側はこの Observer を
使ってツール実行のトレースを描画します。

---

## 5. ツールのスキーマ化 (`internal/tools`)

モデルにツールを「広告」するには、各ツールを JSON Schema 付きの定義に変換する必要が
あります。それを担うのがレジストリです。

### 5-1. Tool インターフェース

```go
// internal/tools/registry.go
type Tool interface {
	Name() string                   // モデルに見せる一意な名前
	Description() string            // いつ・どう使うかの説明
	Parameters() json.RawMessage    // 引数の JSON Schema
	Call(ctx context.Context, args json.RawMessage) (string, error)
}
```

新しいツールはこの 4 メソッドを実装するだけで追加できます。LangChain の `Tool`
抽象に対応します。

### 5-2. 定義の生成 — `Definitions`

```go
// internal/tools/registry.go
func (r *Registry) Definitions() []llm.Tool {
	defs := make([]llm.Tool, 0, len(r.tools))
	for _, name := range r.Names() { // 名前順（決定的）に並べる
		t := r.tools[name]
		defs = append(defs, llm.Tool{
			Type: "function",
			Function: llm.FunctionDefinition{
				Name:        t.Name(),
				Description: t.Description(),
				Parameters:  t.Parameters(),
			},
		})
	}
	return defs
}
```

`Run` の (B) で呼ばれていたのがこれです。各ツールの `Name/Description/Parameters` を
`llm.Tool` に詰め替え、`ChatRequest.Tools` に乗せてモデルへ送ります。`Names()` が
ソート済みを返すため、**送るツール順序が決定的**（テストや再現性に有利）です。

### 5-3. 具体例: Calculator ツール

```go
// internal/tools/calculator.go
func (Calculator) Name() string { return "calculator" }

func (Calculator) Description() string {
	return "Evaluate a mathematical expression and return the numeric result. ..."
}

func (Calculator) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"expression": {
				"type": "string",
				"description": "The arithmetic expression to evaluate, e.g. \"(3 + 4) * 2 ^ 3\"."
			}
		},
		"required": ["expression"]
	}`)
}

func (Calculator) Call(_ context.Context, args json.RawMessage) (string, error) {
	var a calcArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return "", fmt.Errorf("calculator: invalid arguments: %w", err)
	}
	// ... 自前のシャンティングヤード法で式を評価 ...
}
```

`Call` は **モデルが生成した JSON 引数** を `json.Unmarshal` で構造体に展開してから
処理します。`Parameters()` の JSON Schema が `required: ["expression"]` を宣言して
いるため、モデルは `{"expression": "..."}` という形で引数を生成します。
スキーマと `Call` のパースが対になっている点を確認してください。

（Calculator はモデルの不正確な暗算に頼らず正確な計算をさせるためのツールで、
本体は自前の四則演算パーサですが、本ガイドの主題は LLM 接続なので評価器の詳細は
割愛します。）

---

## 6. 全部品をつなぐ配線 (`cmd/agent/main.go`)

最後に、設定 → クライアント → ツール → エージェントをどう組み立てて起動するかを
見ます。

```go
// cmd/agent/main.go
func run(prompt string, quiet, verbose bool) error {
	// (1) 設定読み込み
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	// (2) LLM クライアント生成
	client, err := llm.NewClient(llm.Config{
		Provider: cfg.Provider,
		BaseURL:  cfg.BaseURL,
		APIKey:   cfg.APIKey,
		Model:    cfg.Model,
		Headers:  cfg.ExtraHeaders,
	})
	if err != nil {
		return err
	}

	// (3) ツールレジストリ構築
	registry := buildToolRegistry()

	// (4) エージェント生成（オプション適用）
	opts := []agent.Option{
		agent.WithMaxSteps(cfg.MaxSteps),
		agent.WithTemperature(cfg.Temperature),
	}
	if cfg.SystemPrompt != "" {
		opts = append(opts, agent.WithSystemPrompt(cfg.SystemPrompt))
	}
	ag := agent.New(client, registry, opts...)

	// (5) Ctrl-C で実行中の処理をきれいにキャンセルする context
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// ... ワンショット or REPL の分岐 ...
}
```

```go
// cmd/agent/main.go
func buildToolRegistry() *tools.Registry {
	registry := tools.NewRegistry()
	registry.MustRegister(tools.Calculator{})
	registry.MustRegister(tools.Clock{})
	registry.MustRegister(tools.HTTPGet{})
	return registry
}
```

### 配線の流れ

1. `config.Load()` で環境変数 / `.env` から設定を確定。
2. その設定から `llm.NewClient` で **接続クライアント** を生成（第 2 章）。
3. 組み込みツール 3 つ（`calculator`, `current_time`, `http_get`）を登録。
4. クライアントとレジストリを `agent.New` に渡してエージェントを生成。
5. `signal.NotifyContext` で **Ctrl-C / SIGTERM 時にキャンセルされる ctx** を作成。
   この ctx が `ag.Run` → `client.Chat` → `http.NewRequestWithContext` まで伝播し、
   **中断シグナルで送信中の HTTP まで即座に止まる** ようになっています。

ワンショットモード（`-p` か引数）では 1 ターン実行して終了、そうでなければ REPL で
`Conversation` を使い回しながら対話を続けます。

---

## 7. データフロー全体まとめ

ユーザーが「12345 × 6789 はいくつ?」と尋ねたときの 1 ターンを例に、接続処理の
流れを通しで追うと次のようになります。

```
[ユーザー入力] "12345 * 6789 は?"
      │
      ▼  cmd/agent: ag.Run(ctx, conv, input, obs)
[agent] 会話履歴に user メッセージを追加
      │  registry.Definitions() でツール一覧を JSON Schema 化
      ▼
[agent → llm] client.Chat(ctx, ChatRequest{Messages, Tools, ToolChoice:"auto"})
      │  internal/llm: JSON 化 → POST {baseURL}/chat/completions
      │                Authorization: Bearer <key> + 追加ヘッダ
      ▼
[LLM 応答 #1] assistant: tool_calls=[calculator{"expression":"12345*6789"}]
      │  agent: ToolCalls があるのでループ継続
      ▼
[agent → tools] invokeTool → Calculator.Call → "83810205"
      │  結果を role:tool メッセージ（ToolCallID で紐付け）で履歴に追加
      ▼
[agent → llm] client.Chat(ctx, ...) 履歴に tool 結果を含めて再問い合わせ
      ▼
[LLM 応答 #2] assistant: "12345 × 6789 は 83,810,205 です。"（tool_calls 無し）
      │  agent: ToolCalls が空 → これが最終回答
      ▼
[最終回答] 呼び出し元へ文字列を返す / CLI が標準出力に表示
```

### 接続処理の要点（おさらい）

| 観点 | 実装 | 該当箇所 |
| --- | --- | --- |
| プロバイダ抽象化 | OpenAI 互換フォーマットで複数プロバイダ対応 | `internal/llm` |
| 認証 | `Authorization: Bearer <APIKey>` ヘッダ | `client.go: Chat` |
| エンドポイント切替 | `BaseURL` / プロバイダ既定値 | `client.go: NewClient` |
| キャンセル | `context.Context` を HTTP まで伝播 | `Chat` / `main.go` |
| エラー堅牢性 | JSON 失敗時に生ボディを切詰めて表面化 | `Chat` の多段防御 |
| ツール宣言 | JSON Schema をモデルへ広告 | `registry.Definitions` |
| reason–act ループ | ツール実行と再問い合わせの反復 | `agent.go: Run` |
| 暴走防止 | `maxSteps` による往復上限 | `agent.go: Run` |
| ツール失敗の回復 | エラーを文字列でモデルに返す | `agent.go: invokeTool` |
| テスト容易性 | `HTTPClient` 注入でモック可能 | `client.go: NewClient` |

---

## 付録: LangChain との対応関係

外部ライブラリは使っていませんが、概念は LangChain とよく対応します。

| LangChain の概念 | 本リポジトリの対応実装 |
| --- | --- |
| `ChatOpenAI` などのチャットモデル | `internal/llm.Client`（`Chat` メソッド） |
| `BaseMessage`（System/Human/AI/Tool） | `internal/llm.Message`（4 つの Role） |
| `Tool` / `StructuredTool` | `internal/tools.Tool` インターフェース |
| ツールの JSON Schema バインド | `Registry.Definitions` → `llm.Tool` |
| `AgentExecutor`（実行ループ） | `internal/agent.Agent.Run` |
| `max_iterations` | `Agent.maxSteps` |
| コールバック / トレース | `agent.Observer` / `Event` |
| `dotenv` での設定読込 | `internal/config.loadDotEnv` |

このように、本実装は LangChain を「使う」のではなく、その**中核アイデアを Go の標準
ライブラリだけで透明に再現**したものです。依存が無いぶん、各処理がどう動くかを
ソースから直接たどれるのが学習用として大きな利点です。

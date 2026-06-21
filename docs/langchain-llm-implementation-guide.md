# LangChainを使ったLLM接続処理の詳細解説

このドキュメントでは、`sample-ai-agent-by-golang` における LangChain（langchaingo）を使った LLM 接続処理の実装を、ソースコードと紐付けながら詳細に解説します。

---

## 目次

1. [プロジェクト全体構成](#1-プロジェクト全体構成)
2. [依存ライブラリ: langchaingo とは](#2-依存ライブラリ-langchaingo-とは)
3. [設定管理: `internal/config/config.go`](#3-設定管理-internalconfigconfiggo)
4. [型定義: `internal/llm/types.go`](#4-型定義-internalllmtypesgo)
5. [LLMクライアント: `internal/llm/client.go`](#5-llmクライアント-internalllmclientgo)
6. [ツールシステム: `internal/tools/`](#6-ツールシステム-internaltools)
7. [エージェントループ: `internal/agent/agent.go`](#7-エージェントループ-internalagentAgentgo)
8. [CLI エントリポイント: `cmd/agent/main.go`](#8-cli-エントリポイント-cmdagentmaingo)
9. [データフロー全体図](#9-データフロー全体図)
10. [LangChain との対応関係](#10-langchain-との対応関係)
11. [テスト設計](#11-テスト設計)

---

## 1. プロジェクト全体構成

```
sample-ai-agent-by-golang/
├── cmd/agent/main.go              # CLIエントリポイント（REPLとワンショットモード）
├── internal/
│   ├── config/
│   │   ├── config.go              # 環境変数・.envファイルによる設定管理
│   │   └── config_test.go
│   ├── llm/                       # LangChain LLMクライアント（中核）
│   │   ├── client.go              # OpenAI互換チャットクライアント
│   │   ├── types.go               # ワイヤーフォーマットの型定義
│   │   └── client_test.go
│   ├── agent/                     # Reason-Actループのオーケストレーション
│   │   ├── agent.go               # エージェントコア + Runループ
│   │   └── agent_test.go
│   └── tools/                     # ツールインターフェースと実装
│       ├── registry.go            # ツールレジストリとスキーマ生成
│       ├── calculator.go          # 算術評価ツール
│       ├── time.go                # 時刻・タイムゾーンツール
│       ├── http.go                # 読み取り専用HTTPフェッチツール
│       └── tools_test.go
├── go.mod
├── go.sum
├── .env.example
└── README.md
```

各パッケージの責務は明確に分離されており、依存方向は下図の通りです：

```
cmd/agent
    ↓
internal/agent  ←→  internal/tools
    ↓
internal/llm
    ↓
github.com/tmc/langchaingo  (外部ライブラリ)
    ↓
OpenAI / OpenRouter API  (外部サービス)
```

---

## 2. 依存ライブラリ: langchaingo とは

**ファイル:** `go.mod`

```go
require (
    github.com/tmc/langchaingo v0.1.14
)
```

`langchaingo` は Go 向けの LangChain 移植実装です。このプロジェクトでは以下の機能を使用しています：

| langchaingo の機能 | 使用箇所 | 役割 |
|---|---|---|
| `llms/openai.New()` | `internal/llm/client.go` | OpenAI互換クライアントの初期化 |
| `llms.GenerateContent()` | `internal/llm/client.go:Chat()` | チャット補完APIの呼び出し |
| `llms.MessageContent` | `internal/llm/client.go:toLangChainMessages()` | メッセージ形式の変換 |
| `llms.Tool` / `llms.FunctionDefinition` | `internal/llm/client.go:toLangChainTools()` | ツール定義の変換 |
| `llms.ToolCallResponse` | `internal/llm/client.go:toLangChainMessages()` | ツール結果の変換 |

> **重要:** `langchaingo` の使用は `internal/llm` パッケージに完全に閉じ込められています。
> `agent`・`tools`・`cmd` パッケージは langchaingo を直接参照せず、このプロジェクト独自の型だけを使います。
> これにより、ライブラリを別のものに差し替えても影響範囲が最小になります。

---

## 3. 設定管理: `internal/config/config.go`

### 3.1 Config 構造体

```go
type Config struct {
    Provider     llm.Provider      // "openai" または "openrouter"
    BaseURL      string            // エンドポイントURL上書き（省略可）
    APIKey       string            // APIキー（必須）
    Model        string            // モデル名（例: "gpt-4o-mini"）
    Temperature  float64           // サンプリング温度（デフォルト: 0.7）
    MaxSteps     int               // 最大ステップ数（デフォルト: 10）
    SystemPrompt string            // カスタムシステムプロンプト（省略可）
    ExtraHeaders map[string]string // 追加HTTPヘッダー（OpenRouter等で使用）
}
```

### 3.2 設定の読み込み順序

`Load()` 関数は以下の優先順位で設定を解決します：

```
1. 実際の環境変数（最優先）
        ↓
2. .env ファイルの値（環境変数が未設定の場合のみ）
        ↓
3. プログラムのデフォルト値（最低優先）
```

### 3.3 認識される環境変数

| 環境変数 | 説明 | デフォルト値 |
|---|---|---|
| `AGENT_PROVIDER` | プロバイダー選択 | `openai` |
| `AGENT_BASE_URL` | エンドポイントURL上書き | プロバイダーのデフォルト |
| `AGENT_API_KEY` | APIキー（優先） | — |
| `OPENAI_API_KEY` | OpenAI用フォールバック | — |
| `OPENROUTER_API_KEY` | OpenRouter用フォールバック | — |
| `AGENT_MODEL` | モデル名 | `gpt-4o-mini` |
| `AGENT_TEMPERATURE` | サンプリング温度 | `0.7` |
| `AGENT_MAX_STEPS` | 最大ステップ数 | `10` |
| `AGENT_SYSTEM_PROMPT` | カスタムシステムプロンプト | （組み込みデフォルト） |
| `OPENROUTER_REFERER` | OpenRouterランキング用 | — |
| `OPENROUTER_TITLE` | OpenRouterランキング用 | — |

### 3.4 .env ファイルの解析ロジック

```go
// loadDotEnv の処理:
// 1. ファイルを行単位でスキャン
// 2. 空行・コメント行（#）をスキップ
// 3. "export " プレフィックスを除去（bash スクリプトスタイルに対応）
// 4. KEY=VALUE 形式を解析
// 5. 引用符（" または '）を除去
// 6. 実際の環境変数を上書きしない
```

---

## 4. 型定義: `internal/llm/types.go`

このファイルは OpenAI Chat Completions API のワイヤーフォーマットを Go の型として定義しています。

### 4.1 メッセージロールの4種類

```
会話履歴の構造:

[system] "You are a helpful AI assistant..."
[user]   "東京の現在時刻は？"
[assistant] (tool_calls: [{id:"call_1", function:{name:"current_time", arguments:"{\"timezone\":\"Asia/Tokyo\"}"}}])
[tool]   "2024-01-15 14:30:00 JST (Mon)" (tool_call_id:"call_1")
[assistant] "東京の現在時刻は2024年1月15日 14:30（月曜日）です。"
```

### 4.2 型の階層構造

```
ChatRequest
├── Messages: []Message
│   ├── Role: string ("system"/"user"/"assistant"/"tool")
│   ├── Content: string
│   ├── ToolCalls: []ToolCall
│   │   ├── ID: string
│   │   ├── Type: string ("function")
│   │   └── Function: FunctionCall
│   │       ├── Name: string
│   │       └── Arguments: string (JSON)
│   └── ToolCallID: string
└── Tools: []Tool
    └── Function: FunctionDefinition
        ├── Name: string
        ├── Description: string
        └── Parameters: json.RawMessage (JSON Schema)

ChatResponse
├── Choices: []Choice
│   └── Message: Message (上記と同じ型)
└── Usage: Usage
    ├── PromptTokens: int
    ├── CompletionTokens: int
    └── TotalTokens: int
```

### 4.3 ToolCall のライフサイクル

```
1. モデルがツール呼び出しを要求
   → アシスタントメッセージに ToolCalls が設定される
   → ToolCall.ID = "call_abc123" (モデルが生成する一意ID)

2. エージェントがツールを実行
   → ToolCall.Function.Arguments の JSON を解析して実行

3. ツール結果をモデルへ返す
   → role="tool" のメッセージを作成
   → ToolCallID = "call_abc123" で対応する呼び出しと紐付け
```

---

## 5. LLMクライアント: `internal/llm/client.go`

### 5.1 クライアントの初期化フロー

```
NewClient(cfg Config)
    │
    ├── 1. BaseURL の解決
    │   └── cfg.BaseURL が設定されている場合 → そのまま使用
    │       未設定の場合 → defaultBaseURLs[cfg.Provider] を使用
    │
    ├── 2. HTTPクライアントの構築
    │   ├── cfg.HTTPClient が設定されている場合 → そのまま使用（テスト用）
    │   └── 未設定の場合 → &http.Client{Timeout: 120秒} を作成
    │
    ├── 3. 追加ヘッダーの設定
    │   └── cfg.Headers が設定されている場合
    │       → HTTPクライアントのトランスポートを headerTransport でラップ
    │
    └── 4. langchaingo クライアントの初期化
        └── openai.New(opts...) を呼び出し
```

### 5.2 Chat メソッドの処理フロー

```
Chat(ctx, req ChatRequest) → *ChatResponse

    1. モデル名の解決
       → req.Model が設定されている場合 → そのまま使用
       → 未設定の場合 → c.model（クライアントデフォルト）を使用

    2. メッセージ変換: toLangChainMessages(req.Messages)
       → エージェントの []Message → langchaingo の []llms.MessageContent

    3. 呼び出しオプションの構築
       → llms.WithModel(model)
       → llms.WithTemperature(...)  ← 設定されている場合
       → llms.WithMaxTokens(...)    ← MaxTokens > 0 の場合
       → llms.WithTools(tools)      ← ツールがある場合
       → llms.WithToolChoice(...)   ← ToolChoice が設定されている場合

    4. LLM API 呼び出し
       → c.llm.GenerateContent(ctx, messages, callOpts...)
       → 内部的に POST /v1/chat/completions へのHTTPリクエスト

    5. レスポンス変換: fromLangChainResponse(resp)
       → langchaingo の *llms.ContentResponse → エージェントの *ChatResponse
```

### 5.3 メッセージ変換の詳細

`toLangChainMessages` はロールに応じて変換方法を切り替えます：

```go
// system → TextParts(ChatMessageTypeSystem, content)
case RoleSystem:
    out = append(out, llms.TextParts(llms.ChatMessageTypeSystem, m.Content))

// user → TextParts(ChatMessageTypeHuman, content)
case RoleUser:
    out = append(out, llms.TextParts(llms.ChatMessageTypeHuman, m.Content))

// assistant → MessageContent{Role:AI, Parts:[TextContent, ...ToolCall]}
case RoleAssistant:
    parts := []llms.ContentPart{}
    if m.Content != "" {
        parts = append(parts, llms.TextContent{Text: m.Content})
    }
    for _, tc := range m.ToolCalls {
        parts = append(parts, llms.ToolCall{...})
    }
    out = append(out, llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: parts})

// tool → MessageContent{Role:Tool, Parts:[ToolCallResponse]}
case RoleTool:
    out = append(out, llms.MessageContent{
        Role: llms.ChatMessageTypeTool,
        Parts: []llms.ContentPart{llms.ToolCallResponse{
            ToolCallID: m.ToolCallID,
            Name:       m.Name,
            Content:    m.Content,
        }},
    })
```

### 5.4 headerTransport によるヘッダー注入

`langchaingo` は内部で作成した HTTP クライアントを使うため、追加ヘッダーを注入するには `http.RoundTripper` インターフェースを実装した独自トランスポートでラップする必要があります：

```go
type headerTransport struct {
    base    http.RoundTripper    // 元のトランスポート
    headers map[string]string    // 注入するヘッダー
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
    // リクエストをクローンしてから変更（元のリクエストを汚染しない）
    clone := req.Clone(req.Context())
    for k, v := range t.headers {
        clone.Header.Set(k, v)
    }
    return t.base.RoundTrip(clone)
}
```

これにより、OpenRouter の `HTTP-Referer` や `X-Title` ヘッダーが透過的にすべてのリクエストへ付加されます。

---

## 6. ツールシステム: `internal/tools/`

### 6.1 Tool インターフェース

```go
type Tool interface {
    Name()        string           // モデルへ公開される識別子
    Description() string           // いつ・どう使うかのガイダンス
    Parameters()  json.RawMessage  // 引数の JSON Schema
    Call(ctx context.Context, args json.RawMessage) (string, error)
}
```

各ツール実装は以下の3つのステップで動作します：

```
1. Parameters() → JSON Schema をモデルへ公開
   モデルがこれを見てどんな引数が必要かを理解する

2. モデルが Call() を要求（ToolCall）
   Arguments: JSON Schema に従った JSON オブジェクト

3. Call() の実行
   args を JSON デコード → ロジック実行 → 結果を文字列で返す
```

### 6.2 組み込みツールの詳細

#### Calculator（算術計算）

```go
// ツール名: "calculator"
// 引数: {"expression": "(3 + 4) * 2 ^ 3"}
// 返値: "56"

// 実装の特徴:
// - shunting-yard アルゴリズムによる式の解析
// - 対応演算子: + - * / % ^ ( ) 単項+/-
// - 定数: pi, e
// - 整数結果は ".0" なしで返す
```

**shunting-yard アルゴリズムの流れ:**
```
入力: "(3 + 4) * 2 ^ 3"

1. トークン化: [(, 3, +, 4, ), *, 2, ^, 3]

2. 中置→逆ポーランド記法変換 (RPN):
   出力キュー: [3, 4, +, 2, 3, ^, *]

3. RPN の評価:
   スタック: [3] → [3,4] → [7] → [7,2] → [7,2,3] → [7,8] → [56]

結果: 56
```

#### Clock（時刻取得）

```go
// ツール名: "current_time"
// 引数: {"timezone": "Asia/Tokyo"}（省略可、省略時はUTC）
// 返値: "2024-01-15 14:30:00 JST (Mon)"

// 実装の特徴:
// - IANA タイムゾーン名に対応（例: "Asia/Tokyo", "UTC", "America/New_York"）
// - Now フィールドが注入可能（テスト用）
// - Go の time.LoadLocation() でタイムゾーン解決
```

#### HTTPGet（Webフェッチ）

```go
// ツール名: "http_get"
// 引数: {"url": "https://example.com/api"}
// 返値: "HTTP 200 OK\n\n<レスポンスボディ（最大16KiB）>"

// 実装の特徴:
// - http:// と https:// のみ許可（セキュリティ）
// - レスポンスボディを 16KiB に制限（モデルのコンテキスト節約）
// - User-Agent ヘッダーを設定
// - context によるキャンセルに対応
// - タイムアウト: 30秒（デフォルト）
```

### 6.3 Registry の役割

```go
type Registry struct {
    tools map[string]Tool
}

// Register: 起動時にツールを登録（重複名はpanicで即座に発見）
registry.MustRegister(tools.Calculator{})
registry.MustRegister(tools.Clock{})
registry.MustRegister(tools.HTTPGet{})

// Definitions(): ツール定義をLLM API形式に変換して返す
// → ChatRequest.Tools に設定され、モデルへ送信される
```

---

## 7. エージェントループ: `internal/agent/agent.go`

### 7.1 Reason-Act ループの詳細フロー

```
ag.Run(ctx, conv, "東京の現在時刻は？", obs)

┌─────────────────────────────────────────────────────────────┐
│ ① ユーザー入力を会話履歴に追加                               │
│   conv.Messages = append(..., {Role:"user", Content:"東京の..."}│
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ ② LLM へリクエスト（ステップ 0）                             │
│   ChatRequest{                                              │
│     Messages: [system, user],                              │
│     Tools: [calculator_def, current_time_def, http_get_def],│
│     ToolChoice: "auto"                                      │
│   }                                                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ ③ モデルのレスポンス                                         │
│   Choice.Message = {                                        │
│     Role: "assistant",                                      │
│     Content: "",                                            │
│     ToolCalls: [{                                           │
│       ID: "call_xyz",                                       │
│       Type: "function",                                     │
│       Function: {                                           │
│         Name: "current_time",                              │
│         Arguments: "{\"timezone\":\"Asia/Tokyo\"}"          │
│       }                                                     │
│     }]                                                      │
│   }                                                         │
└─────────────────────┬───────────────────────────────────────┘
                      │ ToolCalls が空でない → ツール実行
┌─────────────────────▼───────────────────────────────────────┐
│ ④ ツール実行: invokeTool(ctx, tc, obs)                       │
│   obs に EventToolCall を通知                               │
│   registry.Get("current_time") → tools.Clock{}              │
│   Clock{}.Call(ctx, {"timezone":"Asia/Tokyo"})              │
│   → "2024-01-15 14:30:00 JST (Mon)"                        │
│   obs に EventToolResult を通知                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ ⑤ ツール結果を会話履歴に追加                                 │
│   {Role:"tool", Name:"current_time",                        │
│    ToolCallID:"call_xyz",                                   │
│    Content:"2024-01-15 14:30:00 JST (Mon)"}                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ ⑥ LLM へ再リクエスト（ステップ 1）                           │
│   Messages: [system, user, assistant(tool_call), tool(result)]│
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ ⑦ モデルの最終回答                                          │
│   Choice.Message = {                                        │
│     Role: "assistant",                                      │
│     Content: "東京の現在時刻は2024年1月15日 14:30（月曜日）です。",│
│     ToolCalls: []  ← 空                                     │
│   }                                                         │
└─────────────────────┬───────────────────────────────────────┘
                      │ ToolCalls が空 → 最終回答
                      ▼
             obs に EventFinalAnswer を通知
             "東京の現在時刻は2024年1月15日 14:30（月曜日）です。" を返す
```

### 7.2 ステップ上限（maxSteps）の役割

```go
for step := 0; step < a.maxSteps; step++ {
    // ...
}
return "", fmt.Errorf("agent: exceeded max steps (%d) without a final answer", a.maxSteps)
```

`maxSteps`（デフォルト: 10）は以下のシナリオを防ぎます：
- モデルが同じツールを何度も繰り返し呼び出す
- ツール結果を見てもさらに別のツールを呼び出し続ける
- 何らかの理由で最終回答を生成しない

### 7.3 エラーをテキストとして返す設計

```go
func (a *Agent) invokeTool(...) string {
    result, err := tool.Call(ctx, rawArgs)
    if err != nil {
        emit(obs, Event{Type: EventToolError, ...})
        return "Error: " + err.Error()  // ← Go error ではなく文字列として返す
    }
    return result
}
```

ツールエラーを Go の `error` として伝播させず、テキストとしてモデルへ返す理由：

- モデルがエラー内容を読んで**回復できる**（例：引数を修正して再試行）
- エラーが1つあっても Run 全体が中断されない
- モデルがユーザーへ適切なエラーメッセージを生成できる

### 7.4 Observer パターン

```go
type Observer func(Event)

// Agent のコアロジック（agent.go）はフロントエンドを知らない
// ↑ emit(obs, Event{...}) で通知するだけ

// CLI（main.go）が Observer を実装して表示する
obs := func(e agent.Event) {
    switch e.Type {
    case agent.EventToolCall:
        fmt.Printf("  ↳ %s(%s)\n", e.ToolName, e.ToolArgs)
    case agent.EventToolResult:
        // verbose モードの場合のみ表示
    case agent.EventToolError:
        fmt.Printf("  ! %s: %v\n", e.ToolName, e.Err)
    }
}
```

---

## 8. CLI エントリポイント: `cmd/agent/main.go`

### 8.1 コンポーネントの組み立て

```
main()
  └── run(prompt, quiet, verbose)
        ├── config.Load()           → Config
        ├── llm.NewClient(cfg)      → *llm.Client
        ├── buildToolRegistry()     → *tools.Registry
        ├── agent.New(client, reg)  → *Agent
        ├── signal.NotifyContext()  → ctx（Ctrl-C対応）
        └── newObserver(quiet, verbose) → Observer
              ├── ワンショットモード: ag.Run(ctx, conv, oneShot, obs)
              └── REPLモード: repl(ctx, ag, obs, cfg, registry)
```

### 8.2 REPL ループのコマンド

| コマンド | 動作 |
|---|---|
| （通常のテキスト） | エージェントへ送信 |
| `/reset` | 会話履歴をリセット（システムプロンプトのみの状態に戻す） |
| `/exit` または `/quit` | プログラムを終了 |
| Ctrl-D (EOF) | プログラムを終了 |
| Ctrl-C | 進行中のリクエストをキャンセル（REPL は継続） |

### 8.3 シグナル処理

```go
ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
defer stop()
```

`context.Context` を使ったキャンセルは以下のように伝播します：

```
Ctrl-C / SIGTERM
    ↓ signal.NotifyContext がキャンセル
    ↓ ctx.Done() チャネルが閉じる
    ↓ ag.Run() → a.client.Chat() → c.llm.GenerateContent()
    ↓ → http.NewRequestWithContext(ctx, ...) のHTTPリクエストがキャンセル
    ↓ エラーが伝播
    → REPL では ctx.Err() != nil を検出して正常終了（エラー表示なし）
```

---

## 9. データフロー全体図

```
ユーザー入力 "125 * 137 は？"
          │
          ▼
    cmd/agent/main.go
    ag.Run(ctx, conv, "125 * 137 は？", obs)
          │
          ▼
    internal/agent/agent.go
    conv.Messages に {role:"user", content:"125 * 137 は？"} を追加
    registry.Definitions() でツール定義一覧を取得
          │
          ▼
    client.Chat(ctx, ChatRequest{
        Messages: [system, user],
        Tools: [calculator_def, current_time_def, http_get_def],
        ToolChoice: "auto"
    })
          │
          ▼
    internal/llm/client.go
    toLangChainMessages() でlangchaingo形式に変換
    toLangChainTools() でツール定義をlangchaingo形式に変換
    c.llm.GenerateContent(ctx, messages, opts...)
          │
          ▼
    github.com/tmc/langchaingo
    HTTP POST https://api.openai.com/v1/chat/completions
    Authorization: Bearer sk-...
    Content-Type: application/json
    Body: {
        "model": "gpt-4o-mini",
        "messages": [...],
        "tools": [...],
        "tool_choice": "auto",
        "temperature": 0.7
    }
          │
          ▼
    OpenAI API レスポンス
    {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": "{\"expression\":\"125 * 137\"}"
                    }
                }]
            }
        }]
    }
          │
          ▼
    fromLangChainResponse() でエージェント形式に変換
    ChatResponse を返す
          │
          ▼
    internal/agent/agent.go
    ToolCalls が空でない → ツール実行
    invokeTool(ctx, {ID:"call_abc", Function:{Name:"calculator", Arguments:"..."}}, obs)
    obs に EventToolCall を通知 → CLI が "↳ calculator(...)" を表示
          │
          ▼
    internal/tools/registry.go
    registry.Get("calculator") → tools.Calculator{}
          │
          ▼
    internal/tools/calculator.go
    Calculator{}.Call(ctx, {"expression":"125 * 137"})
    → evalExpression("125 * 137") → 17125.0 → "17125"
          │
          ▼
    internal/agent/agent.go
    obs に EventToolResult を通知
    conv.Messages に {role:"tool", name:"calculator", tool_call_id:"call_abc", content:"17125"} を追加
    次のループへ → client.Chat() を再呼び出し
          │
          ▼
    OpenAI API (2回目)
    {messages: [system, user, assistant(tool_call), tool(result)]}
    → {"role":"assistant", "content":"125 × 137 = 17,125 です。"}
          │
          ▼
    ToolCalls が空 → 最終回答
    obs に EventFinalAnswer を通知
    "125 × 137 = 17,125 です。" を返す
          │
          ▼
    cmd/agent/main.go
    fmt.Printf("\n%s\n", answer)
    "> " プロンプトを再表示
```

---

## 10. LangChain との対応関係

このプロジェクトは LangChain の主要概念を Go で実装しています：

| LangChain の概念 | このプロジェクトでの実装 | ファイル |
|---|---|---|
| **ChatModel** | `llm.Client` + langchaingo `openai.LLM` | `internal/llm/client.go` |
| **HumanMessage / AIMessage / SystemMessage** | `llm.Message` (Role フィールドで区別) | `internal/llm/types.go` |
| **Tool / StructuredTool** | `tools.Tool` インターフェース | `internal/tools/registry.go` |
| **ToolRegistry** | `tools.Registry` | `internal/tools/registry.go` |
| **AgentExecutor** | `agent.Agent` + `Run()` メソッド | `internal/agent/agent.go` |
| **ConversationHistory** | `agent.Conversation` | `internal/agent/agent.go` |
| **Callbacks / CallbackManager** | `agent.Observer` | `internal/agent/agent.go` |
| **Function Calling** | `llm.ToolCall` / `llm.Tool` | `internal/llm/types.go` |

---

## 11. テスト設計

### 11.1 モックベーステスト

実際のAPIを呼び出さずにテストするため、`HTTPClient` を注入可能にしています：

```go
// テスト用のモックHTTPサーバー
server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
    // 固定のレスポンスを返す
    json.NewEncoder(w).Encode(mockResponse)
}))
defer server.Close()

client, _ := llm.NewClient(llm.Config{
    BaseURL:    server.URL,
    APIKey:     "test-key",
    HTTPClient: server.Client(), // ← テスト用クライアントを注入
})
```

### 11.2 テストカバレッジ

| ファイル | テスト内容 |
|---|---|
| `internal/config/config_test.go` | 設定読み込みとデフォルト値の検証 |
| `internal/llm/client_test.go` | リクエスト/レスポンスの変換処理 |
| `internal/agent/agent_test.go` | ツール実行を含む Reason-Act ループ |
| `internal/tools/tools_test.go` | 各ツールの実行ロジック |

### 11.3 Clock ツールの時刻注入

```go
// テストで固定時刻を注入できる
clock := tools.Clock{
    Now: func() time.Time {
        return time.Date(2024, 1, 15, 14, 30, 0, 0, time.UTC)
    },
}
result, _ := clock.Call(context.Background(), []byte(`{"timezone":"Asia/Tokyo"}`))
// → "2024-01-15 23:30:00 JST (Mon)"
```

---

## まとめ

このプロジェクトにおける LangChain（langchaingo）を使った LLM 接続処理の核心は以下の通りです：

1. **`internal/llm/client.go`** が langchaingo の唯一の接触点です。`openai.New()` でクライアントを初期化し、`GenerateContent()` で API を呼び出します。

2. **型変換の橋渡し** として、エージェントの型（`llm.Message`, `llm.Tool`）と langchaingo の型（`llms.MessageContent`, `llms.Tool`）を相互変換する関数群（`toLangChainMessages`, `toLangChainTools`, `fromLangChainResponse`）が存在します。

3. **Reason-Act ループ**（`internal/agent/agent.go`）がエージェントの中核で、LLM 呼び出し → ツール実行 → 結果返却を繰り返し、最終回答を得ます。

4. **ツールシステム**（`internal/tools/`）は共通インターフェースと Registry で管理され、JSON Schema を通じてモデルへ機能を公開します。

5. **langchaingo の使用はパッケージ境界内に封じ込め**られており、他の層は独自の型のみを使用します。これにより将来的なライブラリ変更が容易になっています。

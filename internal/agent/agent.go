// Package agent はコアとなる Reason-Act ループを実装します。
// 会話をモデルへ送信し、モデルが要求するツールを実行し、結果を返し、
// モデルが最終的な回答を生成するまでこれを繰り返します。
//
// このパターンは ReAct (Reasoning + Acting) とも呼ばれ、LangChain の
// AgentExecutor が実装するループと同等のものを Go で自前実装しています。
package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/llm"
	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/tools"
)

// DefaultSystemPrompt はエージェントがツールを使って正確な回答をするよう
// 誘導するデフォルトのシステムプロンプトです。
// モデルの推論能力とツール実行能力を組み合わせた回答を促します。
const DefaultSystemPrompt = "You are a helpful AI assistant with access to tools. " +
	"When a tool can produce a more accurate answer than your own reasoning " +
	"(for example arithmetic, the current time, or fetching a web page), prefer " +
	"calling the tool. If a knowledge_search tool is available, consult the " +
	"knowledge base for questions about this project or topics it may document, " +
	"and ground your answer in the retrieved passages, citing their source. " +
	"Think step by step and give concise, correct answers."

// Event は Run 中に発生した出来事を記述します。
// コールバックを通じて呼び出し元（CLI・テスト・UI）へ配信されることで、
// エージェント自身が特定のフロントエンドを知ることなく進捗を表示できます。
type Event struct {
	// Type はこのイベントの種類を示します。
	Type EventType
	// Text は EventAssistantMessage / EventFinalAnswer のアシスタントコンテンツを保持します。
	Text string
	// ToolName / ToolArgs / ToolResult はツールアクティビティを説明します。
	ToolName   string
	ToolArgs   string
	ToolResult string
	// Err は EventToolError の場合に設定されます。
	Err error
}

// EventType はエージェントが発行する Event の種類を列挙します。
type EventType int

const (
	// EventToolCall はツールが呼び出される直前に発行されます。
	// CLI はこれを使ってツール呼び出しのトレースを表示できます。
	EventToolCall EventType = iota
	// EventToolResult はツールが正常に返された後に発行されます。
	EventToolResult
	// EventToolError はツール呼び出しが失敗した際に発行されます。
	EventToolError
	// EventFinalAnswer はモデルの最終的なテキスト回答とともに一度だけ発行されます。
	EventFinalAnswer
)

// Observer は Run 中に Event を受け取るコールバック関数型です。nil でも構いません。
// この Observer パターンにより、エージェントのコアロジックをフロントエンドから分離できます。
type Observer func(Event)

// Agent は LLM クライアントとツールレジストリを結びつけます。
// エージェントは会話履歴を管理し、Reason-Act ループを駆動します。
type Agent struct {
	// client は LLM への通信を担当します（langchaingo ラッパー）。
	client *llm.Client
	// registry は利用可能なツールのセットを保持します。
	registry *tools.Registry

	// systemPrompt はすべての会話の先頭に挿入されるシステムメッセージです。
	systemPrompt string
	// maxSteps は1回の Run で許可するモデル/ツールの往復回数の上限です。
	// 無限ループを防ぐためのセーフガードです。
	maxSteps int
	// temperature はモデルに渡すサンプリング温度です（nil の場合はモデルデフォルト）。
	temperature *float64
}

// Option は Agent をカスタマイズする関数型です（Functional Options パターン）。
type Option func(*Agent)

// WithSystemPrompt はデフォルトのシステムプロンプトを上書きします。
// 空文字列の場合は無視されます。
func WithSystemPrompt(p string) Option {
	return func(a *Agent) {
		if p != "" {
			a.systemPrompt = p
		}
	}
}

// WithMaxSteps は1回の Run が行えるモデル/ツール往復回数の上限を設定します。
// ツールの連鎖呼び出しが暴走するのを防ぎます。1未満の値は無視されます。
func WithMaxSteps(n int) Option {
	return func(a *Agent) {
		if n > 0 {
			a.maxSteps = n
		}
	}
}

// WithTemperature はモデルへ渡すサンプリング温度を設定します。
// 低いほど決定論的、高いほど多様な応答になります。
func WithTemperature(t float64) Option {
	return func(a *Agent) { a.temperature = &t }
}

// New は Agent を構築します。
// デフォルト値: systemPrompt=DefaultSystemPrompt, maxSteps=10
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

// Conversation は複数ターンのセッションにわたる実行中のメッセージ履歴を保持します。
// 同一 Conversation を Run に繰り返し渡すことで、文脈を保持した多ターン対話が実現します。
type Conversation struct {
	// Messages はシステムプロンプトから始まる全会話履歴です。
	// 各 Run の実行後にメッセージが追記されていきます。
	Messages []llm.Message
}

// NewConversation はエージェントのシステムプロンプトを先頭に持つ会話を開始します。
// 新しい独立した会話セッションを始める際に使用します。
func (a *Agent) NewConversation() *Conversation {
	return &Conversation{
		Messages: []llm.Message{{Role: llm.RoleSystem, Content: a.systemPrompt}},
	}
}

// Run はユーザーの1ターンを処理して会話を進めます。
// ユーザー入力を追記し、モデルが最終回答を返す（またはステップ上限に達する）まで
// Reason-Act ループを駆動し、その回答を返します。
// 会話はインプレースで変更されるため、ターンをまたいで再利用できます。
//
// Reason-Act ループの流れ:
//  1. ユーザー入力をメッセージ履歴に追加
//  2. 利用可能なツール定義とともに LLM へリクエストを送信
//  3. モデルがツール呼び出しを要求した場合 → ツールを実行し結果を履歴に追加 → 2へ戻る
//  4. モデルがテキスト回答を返した場合 → その回答を返して終了
func (a *Agent) Run(ctx context.Context, conv *Conversation, userInput string, obs Observer) (string, error) {
	if conv == nil {
		return "", fmt.Errorf("agent: conversation is nil")
	}
	// ユーザーの入力を会話履歴に追加します
	conv.Messages = append(conv.Messages, llm.Message{Role: llm.RoleUser, Content: userInput})

	// 利用可能なツールの定義を JSON Schema 形式で取得します
	toolDefs := a.registry.Definitions()

	// Reason-Act ループ: モデルが最終回答を返すかステップ上限に達するまで繰り返します
	for step := 0; step < a.maxSteps; step++ {
		req := llm.ChatRequest{
			Messages:    conv.Messages,
			Temperature: a.temperature,
		}
		if len(toolDefs) > 0 {
			req.Tools = toolDefs
			// "auto" モード: モデルがツールを使うかどうかを自律的に判断します
			req.ToolChoice = "auto"
		}

		// LLM へリクエストを送信します（langchaingo 経由で OpenAI API を呼び出し）
		resp, err := a.client.Chat(ctx, req)
		if err != nil {
			return "", fmt.Errorf("agent: chat request: %w", err)
		}

		msg := resp.Choices[0].Message
		// アシスタントターン（テキスト内容またはツール呼び出し）を会話履歴に記録します。
		// 次の往復でプロバイダーが整形された履歴を受け取れるようにするために必要です。
		conv.Messages = append(conv.Messages, msg)

		// モデルがツール呼び出しを要求していない場合 → これが最終回答です
		if len(msg.ToolCalls) == 0 {
			emit(obs, Event{Type: EventFinalAnswer, Text: msg.Content})
			return msg.Content, nil
		}

		// モデルが要求したすべてのツール呼び出しを実行し、その結果を追記します。
		// 複数のツールを同時に要求された場合も順番に実行します。
		for _, tc := range msg.ToolCalls {
			result := a.invokeTool(ctx, tc, obs)
			// ツール結果をモデルへ返すメッセージとして会話履歴に追加します
			conv.Messages = append(conv.Messages, llm.Message{
				Role:       llm.RoleTool,
				Name:       tc.Function.Name,
				ToolCallID: tc.ID, // モデルが複数ツールを呼んだ場合の識別子
				Content:    result,
			})
		}
		// ツール結果を追加したら次のループでモデルへ再度リクエストします
	}

	return "", fmt.Errorf("agent: exceeded max steps (%d) without a final answer", a.maxSteps)
}

// invokeTool は単一のツール呼び出しを実行し、モデルへ返す文字列を返します。
// ツールエラーは Go エラーとして返すのではなく、テキストとして返します。
// これにより、エラー内容をモデルが確認して回復できるようになります。
// （例: 不正な引数の場合、モデルが引数を修正して再試行できる）
func (a *Agent) invokeTool(ctx context.Context, tc llm.ToolCall, obs Observer) string {
	name := tc.Function.Name
	rawArgs := json.RawMessage(tc.Function.Arguments)
	// ツール呼び出し前に Observer へイベントを通知します（CLI のトレース表示等）
	emit(obs, Event{Type: EventToolCall, ToolName: name, ToolArgs: tc.Function.Arguments})

	// レジストリからツールを検索します
	tool, ok := a.registry.Get(name)
	if !ok {
		err := fmt.Errorf("unknown tool %q", name)
		emit(obs, Event{Type: EventToolError, ToolName: name, Err: err})
		// 未知のツール名はエラーテキストとしてモデルへ返します
		return "Error: " + err.Error()
	}

	// ツールを実行します（calculator, current_time, http_get など）
	result, err := tool.Call(ctx, rawArgs)
	if err != nil {
		emit(obs, Event{Type: EventToolError, ToolName: name, Err: err})
		// ツールエラーもテキストとしてモデルへ返すことでモデルが状況を把握できます
		return "Error: " + err.Error()
	}

	emit(obs, Event{Type: EventToolResult, ToolName: name, ToolResult: result})
	return result
}

// emit は Observer が nil でない場合にイベントを送信します。
func emit(obs Observer, e Event) {
	if obs != nil {
		obs(e)
	}
}

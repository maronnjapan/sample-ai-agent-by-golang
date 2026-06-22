// Command agent は AI エージェントの CLI フロントエンドです。
// ワンショットモード（-p フラグまたは位置引数でプロンプトを渡す）と
// インタラクティブ REPL モードをサポートします。
//
// 設定は環境変数から読み込みます。認識される変数については
// internal/config パッケージまたは .env.example を参照してください。
// AGENT_PROVIDER または AGENT_BASE_URL を切り替えることで、
// OpenAI・OpenRouter・その他 OpenAI 互換ゲートウェイを対象にできます。
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/agent"
	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/config"
	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/knowledge"
	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/llm"
	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/tools"
)

func main() {
	// コマンドラインフラグの定義
	var (
		// -p: ワンショットモードでプロンプトを指定します（インタラクティブモードを回避）
		prompt = flag.String("p", "", "run a single prompt and exit (non-interactive)")
		// --quiet: ツール呼び出しのトレース出力を抑制します
		quiet = flag.Bool("quiet", false, "suppress tool-call trace output")
		// --verbose: ツール実行結果もトレースに表示します
		verbose = flag.Bool("verbose", false, "show tool results in the trace")
	)
	flag.Parse()

	if err := run(*prompt, *quiet, *verbose); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// run はアプリケーションのメインロジックを実行します。
// 設定の読み込み → LLMクライアント作成 → ツール登録 → エージェント構築 → 実行 の順で処理します。
func run(prompt string, quiet, verbose bool) error {
	// 環境変数（または .env ファイル）から設定を読み込みます
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	// langchaingo をラップした LLM クライアントを構築します
	// このクライアントが OpenAI Chat Completions API との通信を担当します
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

	// ナレッジベースを読み込みます（ディレクトリが無い・空の場合は何も登録しません）。
	// 読み込めた場合は knowledge_search ツールとして検索を提供します。
	store, err := knowledge.Load(cfg.KnowledgeDir)
	if err != nil {
		return err
	}

	// 組み込みツールを登録します
	registry := buildToolRegistry(store)

	// エージェントのオプションを構築します
	opts := []agent.Option{
		agent.WithMaxSteps(cfg.MaxSteps),
		agent.WithTemperature(cfg.Temperature),
	}
	if cfg.SystemPrompt != "" {
		opts = append(opts, agent.WithSystemPrompt(cfg.SystemPrompt))
	}
	// LLM クライアントとツールレジストリを組み合わせてエージェントを作成します
	ag := agent.New(client, registry, opts...)

	// Ctrl-C (SIGINT) や SIGTERM で進行中の処理をクリーンにキャンセルします。
	// context.Context を通じてキャンセル信号がHTTPリクエストまで伝播します。
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// ツールアクティビティを表示する Observer を作成します
	obs := newObserver(quiet, verbose)

	// ワンショットモードの処理:
	// -p フラグ、または位置引数としてプロンプトが提供された場合
	oneShot := strings.TrimSpace(prompt)
	if oneShot == "" && flag.NArg() > 0 {
		// 位置引数をスペースで結合してプロンプトとして使用します
		oneShot = strings.Join(flag.Args(), " ")
	}
	if oneShot != "" {
		// 新しい会話を開始してワンショット実行し、結果を出力します
		conv := ag.NewConversation()
		answer, err := ag.Run(ctx, conv, oneShot, obs)
		if err != nil {
			return err
		}
		fmt.Println(answer)
		return nil
	}

	// インタラクティブ REPL モードを開始します
	return repl(ctx, ag, obs, cfg, registry, store)
}

// buildToolRegistry は組み込みツールを登録します。
// 現在利用可能なツール:
//   - Calculator: 算術式評価（正確な数値計算）
//   - Clock: 現在の日時取得（タイムゾーン対応）
//   - HTTPGet: HTTPリクエスト実行（読み取り専用・サイズ制限あり）
//   - KnowledgeSearch: ナレッジベース検索（store にナレッジがある場合のみ登録）
//
// store にナレッジが1件も無い場合は knowledge_search を登録しないため、
// ナレッジを用意していない利用者の動作は従来どおりです。
func buildToolRegistry(store *knowledge.Store) *tools.Registry {
	registry := tools.NewRegistry()
	registry.MustRegister(tools.Calculator{})
	registry.MustRegister(tools.Clock{})
	registry.MustRegister(tools.HTTPGet{})
	if store != nil && store.Len() > 0 {
		registry.MustRegister(tools.KnowledgeSearch{Store: store})
	}
	return registry
}

// repl はインタラクティブループを実行します。
// 会話履歴はターンをまたいで保持されます。
// コマンド: /reset（会話リセット）、/exit または /quit（終了）
func repl(ctx context.Context, ag *agent.Agent, obs agent.Observer, cfg *config.Config, registry *tools.Registry, store *knowledge.Store) error {
	// 起動情報: プロバイダー・モデル・利用可能なツールを表示します
	fmt.Printf("AI Agent (provider=%s model=%s, tools: %s)\n",
		cfg.Provider, cfg.Model, strings.Join(registry.Names(), ", "))
	// ナレッジを読み込めた場合は、件数とソース一覧を表示します
	if store != nil && store.Len() > 0 {
		fmt.Printf("Knowledge: %d chunk(s) from %s\n",
			store.Len(), strings.Join(store.Sources(), ", "))
	}
	fmt.Println("Type your message and press Enter. Commands: /reset, /exit")

	// 会話履歴を初期化します（システムプロンプトが自動的に先頭に追加されます）
	conv := ag.NewConversation()
	// 標準入力を行単位で読み込むスキャナーを設定します
	// バッファサイズ: 最大 1MiB（長いプロンプトに対応）
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			fmt.Println()
			// EOF (Ctrl-D) またはスキャナーエラーで終了します
			return scanner.Err()
		}
		line := strings.TrimSpace(scanner.Text())
		switch {
		case line == "":
			// 空入力はスキップします
			continue
		case line == "/exit" || line == "/quit":
			// 終了コマンド
			return nil
		case line == "/reset":
			// 会話をリセットします（新しいシステムプロンプトから再開）
			conv = ag.NewConversation()
			fmt.Println("(conversation reset)")
			continue
		}

		// エージェントの Reason-Act ループを実行します
		answer, err := ag.Run(ctx, conv, line, obs)
		if err != nil {
			if ctx.Err() != nil {
				// Ctrl-C による割り込み: エラーなしで正常終了します
				return nil
			}
			// エラーを表示して次の入力を待ちます（REPL を終了しません）
			fmt.Fprintln(os.Stderr, "error:", err)
			continue
		}
		fmt.Printf("\n%s\n", answer)
	}
}

// newObserver はツールアクティビティの人間が読めるトレースを表示する Observer を返します。
// quiet フラグが true の場合は nil（何も表示しない）を返します。
// verbose フラグが true の場合はツール実行結果も表示します。
func newObserver(quiet, verbose bool) agent.Observer {
	if quiet {
		// quiet モード: ツールトレースを表示しません
		return nil
	}
	return func(e agent.Event) {
		switch e.Type {
		case agent.EventToolCall:
			// ツール呼び出しを暗い色で表示します: "↳ ツール名(引数)"
			fmt.Printf("  \033[2m↳ %s(%s)\033[0m\n", e.ToolName, e.ToolArgs)
		case agent.EventToolResult:
			if verbose {
				// verbose モード: ツール結果も表示します（200文字に切り詰め）
				fmt.Printf("  \033[2m  = %s\033[0m\n", oneLine(e.ToolResult))
			}
		case agent.EventToolError:
			// ツールエラーを赤色で表示します
			fmt.Printf("  \033[31m  ! %s: %v\033[0m\n", e.ToolName, e.Err)
		}
	}
}

// oneLine は文字列から改行を除去し、200文字を超える場合は切り詰めます。
// Observer のトレース表示で長い結果を簡潔に表示するために使用します。
func oneLine(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > 200 {
		return s[:200] + "…"
	}
	return s
}

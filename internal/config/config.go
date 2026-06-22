// Package config は環境変数からエージェントの設定を読み込みます。
// オプションの .env ファイルにも対応しているため、ローカル開発が摩擦なく行えます。
package config

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/llm"
)

// Config は完全に解決された実行時設定を保持します。
// Load() によって生成され、LLMクライアントやエージェントの構築に使用されます。
type Config struct {
	// Provider は使用するLLMプロバイダーを指定します (openai / openrouter)。
	Provider llm.Provider
	// BaseURL はプロバイダーのデフォルトエンドポイントを上書きします。
	// 自己ホスト型モデルや互換ゲートウェイを使用する際に設定します。
	BaseURL string
	// APIKey はLLM APIへの認証に使用するAPIキーです。
	APIKey string
	// Model は使用するモデル名です（例: "gpt-4o-mini"）。
	Model string
	// Temperature はサンプリング温度です（デフォルト: 0.7）。
	Temperature float64
	// MaxSteps は1ターンあたりのモデル/ツール往復の最大回数です（デフォルト: 10）。
	MaxSteps int
	// SystemPrompt はカスタムシステムプロンプトです。空の場合はエージェントのデフォルトが使用されます。
	SystemPrompt string
	// KnowledgeDir はナレッジ（Markdown 等）を読み込むディレクトリです（デフォルト: "knowledge"）。
	// このディレクトリ配下のファイルが knowledge_search ツールの検索対象になります。
	KnowledgeDir string
	// ExtraHeaders はすべてのLLMリクエストに付加される追加ヘッダーです。
	// OpenRouter のランキングヘッダー等に使用します。
	ExtraHeaders map[string]string
}

// Load は環境変数から設定を読み込みます。
// まず .env ファイルの読み込みを試みてプロセス環境変数として設定し、
// 次にプロバイダー固有のデフォルト値を適用します。
//
// 認識される環境変数:
//
//	AGENT_PROVIDER      "openai" (デフォルト) または "openrouter"
//	AGENT_BASE_URL      OpenAI互換ゲートウェイのエンドポイントを上書き
//	AGENT_API_KEY       APIキー (OPENAI_API_KEY / OPENROUTER_API_KEY にフォールバック)
//	AGENT_MODEL         モデル名 (未設定時はプロバイダー適切なデフォルトが適用)
//	AGENT_TEMPERATURE   サンプリング温度 (デフォルト: 0.7)
//	AGENT_MAX_STEPS     1ターンあたりの最大ステップ数 (デフォルト: 10)
//	AGENT_SYSTEM_PROMPT カスタムシステムプロンプト
//	AGENT_KNOWLEDGE_DIR ナレッジを読み込むディレクトリ (デフォルト: "knowledge")
func Load() (*Config, error) {
	// .env ファイルを読み込みます（存在しない場合は無視）。
	// .env の値はすでに設定されている環境変数を上書きしません。
	_ = loadDotEnv(".env") // best-effort; absence is not an error

	// プロバイダー名を小文字に正規化して取得します（デフォルト: "openai"）
	provider := llm.Provider(strings.ToLower(getenv("AGENT_PROVIDER", string(llm.ProviderOpenAI))))

	cfg := &Config{
		Provider:     provider,
		BaseURL:      os.Getenv("AGENT_BASE_URL"),
		Model:        os.Getenv("AGENT_MODEL"),
		SystemPrompt: os.Getenv("AGENT_SYSTEM_PROMPT"),
		KnowledgeDir: getenv("AGENT_KNOWLEDGE_DIR", "knowledge"),
		Temperature:  0.7, // デフォルト温度: バランスの取れた創造性と一貫性
		MaxSteps:     10,  // デフォルト最大ステップ: ツール連鎖のループを防ぐ安全弁
		ExtraHeaders: map[string]string{},
	}

	// APIキーを解決します: AGENT_API_KEY を優先し、プロバイダー固有変数にフォールバック
	cfg.APIKey = resolveAPIKey(provider)
	// モデル名が未設定の場合はプロバイダー固有のデフォルトを適用します
	if cfg.Model == "" {
		cfg.Model = defaultModel(provider)
	}

	// サンプリング温度の解析（設定されている場合）
	if v := os.Getenv("AGENT_TEMPERATURE"); v != "" {
		t, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return nil, fmt.Errorf("config: invalid AGENT_TEMPERATURE %q: %w", v, err)
		}
		cfg.Temperature = t
	}
	// 最大ステップ数の解析（設定されている場合）
	if v := os.Getenv("AGENT_MAX_STEPS"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return nil, fmt.Errorf("config: invalid AGENT_MAX_STEPS %q: %w", v, err)
		}
		cfg.MaxSteps = n
	}

	// OpenRouter はアプリ識別のためのオプションヘッダーを推奨しています。
	// これらのヘッダーはランキングやモデル分析に使用されます。
	if provider == llm.ProviderOpenRouter {
		if v := os.Getenv("OPENROUTER_REFERER"); v != "" {
			cfg.ExtraHeaders["HTTP-Referer"] = v
		}
		if v := os.Getenv("OPENROUTER_TITLE"); v != "" {
			cfg.ExtraHeaders["X-Title"] = v
		}
	}

	// APIキーが解決できない場合は早期エラーを返します（起動時に失敗させる）
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("config: no API key found; set AGENT_API_KEY (or %s)", apiKeyEnvName(provider))
	}
	return cfg, nil
}

// resolveAPIKey は汎用的な AGENT_API_KEY を優先し、
// 未設定の場合はプロバイダー慣習的な変数名にフォールバックします。
// これにより、プロバイダーを切り替えても1つの変数名で管理できます。
func resolveAPIKey(p llm.Provider) string {
	if v := os.Getenv("AGENT_API_KEY"); v != "" {
		return v
	}
	return os.Getenv(apiKeyEnvName(p))
}

// apiKeyEnvName はプロバイダーに対応するAPIキー環境変数名を返します。
func apiKeyEnvName(p llm.Provider) string {
	switch p {
	case llm.ProviderOpenRouter:
		return "OPENROUTER_API_KEY"
	default:
		// OpenAI および未知のプロバイダー
		return "OPENAI_API_KEY"
	}
}

// defaultModel はプロバイダーに適したデフォルトモデル名を返します。
// モデル名はプロバイダーによって形式が異なります（OpenRouter はベンダープレフィックスが必要）。
func defaultModel(p llm.Provider) string {
	switch p {
	case llm.ProviderOpenRouter:
		// OpenRouter ではモデル名に "ベンダー/モデル名" の形式が必要です
		return "openai/gpt-4o-mini"
	default:
		return "gpt-4o-mini"
	}
}

// getenv は環境変数を取得し、未設定の場合は fallback を返します。
func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// loadDotEnv はシンプルな KEY=VALUE 形式の .env ファイルを解析し、
// 環境にまだ存在しない変数を設定します。ファイルが存在しない場合は黙って無視します。
//
// サポートされる構文:
//   - KEY=VALUE
//   - KEY="VALUE" または KEY='VALUE'（引用符は除去されます）
//   - export KEY=VALUE（export プレフィックスは無視されます）
//   - # で始まる行はコメントとして無視されます
//   - 空行は無視されます
func loadDotEnv(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		// 空行とコメント行をスキップ
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// "export " プレフィックスを除去（bash スクリプトスタイルの .env に対応）
		line = strings.TrimPrefix(line, "export ")
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		// 引用符（" または '）を除去します
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		if key == "" {
			continue
		}
		// 実際の環境変数を上書きしません（環境変数が優先されます）
		if _, exists := os.LookupEnv(key); !exists {
			_ = os.Setenv(key, value)
		}
	}
	return scanner.Err()
}

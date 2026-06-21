// Package config loads agent configuration from environment variables, with
// support for an optional .env file so local development is friction-free.
package config

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/llm"
)

// Config is the fully-resolved runtime configuration.
type Config struct {
	Provider     llm.Provider
	BaseURL      string
	APIKey       string
	Model        string
	Temperature  float64
	MaxSteps     int
	SystemPrompt string
	// ExtraHeaders are attached to every LLM request (e.g. OpenRouter ranking
	// headers).
	ExtraHeaders map[string]string
}

// Load reads configuration from the environment. It first attempts to load a
// .env file (if present) so values defined there populate the process
// environment, then resolves provider-specific defaults.
//
// Recognised variables:
//
//	AGENT_PROVIDER   "openai" (default) or "openrouter"
//	AGENT_BASE_URL   override the endpoint for any OpenAI-compatible gateway
//	AGENT_API_KEY    API key (falls back to OPENAI_API_KEY / OPENROUTER_API_KEY)
//	AGENT_MODEL      model name (provider-appropriate default applied if unset)
//	AGENT_TEMPERATURE sampling temperature (default 0.7)
//	AGENT_MAX_STEPS  max tool/model round-trips per turn (default 10)
//	AGENT_SYSTEM_PROMPT custom system prompt
func Load() (*Config, error) {
	_ = loadDotEnv(".env") // best-effort; absence is not an error

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

	if v := os.Getenv("AGENT_TEMPERATURE"); v != "" {
		t, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return nil, fmt.Errorf("config: invalid AGENT_TEMPERATURE %q: %w", v, err)
		}
		cfg.Temperature = t
	}
	if v := os.Getenv("AGENT_MAX_STEPS"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return nil, fmt.Errorf("config: invalid AGENT_MAX_STEPS %q: %w", v, err)
		}
		cfg.MaxSteps = n
	}

	// OpenRouter recommends identifying your app via these optional headers.
	if provider == llm.ProviderOpenRouter {
		if v := os.Getenv("OPENROUTER_REFERER"); v != "" {
			cfg.ExtraHeaders["HTTP-Referer"] = v
		}
		if v := os.Getenv("OPENROUTER_TITLE"); v != "" {
			cfg.ExtraHeaders["X-Title"] = v
		}
	}

	if cfg.APIKey == "" {
		return nil, fmt.Errorf("config: no API key found; set AGENT_API_KEY (or %s)", apiKeyEnvName(provider))
	}
	return cfg, nil
}

// resolveAPIKey prefers the generic AGENT_API_KEY and falls back to the
// provider-conventional variable.
func resolveAPIKey(p llm.Provider) string {
	if v := os.Getenv("AGENT_API_KEY"); v != "" {
		return v
	}
	return os.Getenv(apiKeyEnvName(p))
}

func apiKeyEnvName(p llm.Provider) string {
	switch p {
	case llm.ProviderOpenRouter:
		return "OPENROUTER_API_KEY"
	default:
		return "OPENAI_API_KEY"
	}
}

func defaultModel(p llm.Provider) string {
	switch p {
	case llm.ProviderOpenRouter:
		return "openai/gpt-4o-mini"
	default:
		return "gpt-4o-mini"
	}
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// loadDotEnv parses a simple KEY=VALUE .env file and sets any variables not
// already present in the environment. It silently ignores a missing file.
func loadDotEnv(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		if key == "" {
			continue
		}
		// Do not clobber variables already set in the real environment.
		if _, exists := os.LookupEnv(key); !exists {
			_ = os.Setenv(key, value)
		}
	}
	return scanner.Err()
}

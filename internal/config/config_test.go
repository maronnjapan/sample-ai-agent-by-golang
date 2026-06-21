package config

import (
	"testing"

	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/llm"
)

func TestLoadDefaultsToOpenAI(t *testing.T) {
	clearAgentEnv(t)
	t.Setenv("OPENAI_API_KEY", "sk-test")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.Provider != llm.ProviderOpenAI {
		t.Errorf("provider = %q", cfg.Provider)
	}
	if cfg.Model != "gpt-4o-mini" {
		t.Errorf("model = %q", cfg.Model)
	}
	if cfg.APIKey != "sk-test" {
		t.Errorf("api key = %q", cfg.APIKey)
	}
}

func TestLoadOpenRouter(t *testing.T) {
	clearAgentEnv(t)
	t.Setenv("AGENT_PROVIDER", "openrouter")
	t.Setenv("OPENROUTER_API_KEY", "or-test")
	t.Setenv("OPENROUTER_TITLE", "MyApp")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.Provider != llm.ProviderOpenRouter {
		t.Errorf("provider = %q", cfg.Provider)
	}
	if cfg.Model != "openai/gpt-4o-mini" {
		t.Errorf("model = %q", cfg.Model)
	}
	if cfg.ExtraHeaders["X-Title"] != "MyApp" {
		t.Errorf("X-Title header = %q", cfg.ExtraHeaders["X-Title"])
	}
}

func TestLoadGenericKeyAndOverrides(t *testing.T) {
	clearAgentEnv(t)
	t.Setenv("AGENT_API_KEY", "generic")
	t.Setenv("AGENT_MODEL", "custom-model")
	t.Setenv("AGENT_TEMPERATURE", "0.1")
	t.Setenv("AGENT_MAX_STEPS", "3")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.APIKey != "generic" || cfg.Model != "custom-model" {
		t.Errorf("overrides not applied: %+v", cfg)
	}
	if cfg.Temperature != 0.1 || cfg.MaxSteps != 3 {
		t.Errorf("numeric overrides not applied: %+v", cfg)
	}
}

func TestLoadMissingKey(t *testing.T) {
	clearAgentEnv(t)
	if _, err := Load(); err == nil {
		t.Error("expected error when no API key is set")
	}
}

func TestLoadInvalidTemperature(t *testing.T) {
	clearAgentEnv(t)
	t.Setenv("AGENT_API_KEY", "k")
	t.Setenv("AGENT_TEMPERATURE", "not-a-number")
	if _, err := Load(); err == nil {
		t.Error("expected error for invalid temperature")
	}
}

// clearAgentEnv unsets every variable Load consults so each test starts clean.
// t.Setenv registers automatic restoration after the test.
func clearAgentEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		"AGENT_PROVIDER", "AGENT_BASE_URL", "AGENT_API_KEY", "AGENT_MODEL",
		"AGENT_TEMPERATURE", "AGENT_MAX_STEPS", "AGENT_SYSTEM_PROMPT",
		"OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_REFERER", "OPENROUTER_TITLE",
	} {
		t.Setenv(k, "")
	}
}

package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestDefaultBaseURL(t *testing.T) {
	if got := DefaultBaseURL(ProviderOpenAI); got != "https://api.openai.com/v1" {
		t.Errorf("openai base url = %q", got)
	}
	if got := DefaultBaseURL(ProviderOpenRouter); got != "https://openrouter.ai/api/v1" {
		t.Errorf("openrouter base url = %q", got)
	}
	if got := DefaultBaseURL("unknown"); got != "" {
		t.Errorf("unknown base url = %q, want empty", got)
	}
}

func TestNewClientValidation(t *testing.T) {
	if _, err := NewClient(Config{Provider: ProviderOpenAI}); err == nil {
		t.Error("expected error for missing API key")
	}
	if _, err := NewClient(Config{APIKey: "k", Provider: "bogus"}); err == nil {
		t.Error("expected error for unknown provider and no base URL")
	}
	if _, err := NewClient(Config{APIKey: "k", Provider: ProviderOpenAI}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestChatSendsExpectedRequest(t *testing.T) {
	var gotAuth, gotHeader, gotPath string
	var gotBody ChatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotHeader = r.Header.Get("X-Title")
		gotPath = r.URL.Path
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		_ = json.NewEncoder(w).Encode(ChatResponse{
			Choices: []Choice{{Message: Message{Role: "assistant", Content: "ok"}}},
		})
	}))
	defer srv.Close()

	c, err := NewClient(Config{
		BaseURL: srv.URL,
		APIKey:  "secret",
		Model:   "default-model",
		Headers: map[string]string{"X-Title": "myapp"},
	})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}

	resp, err := c.Chat(context.Background(), ChatRequest{
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if resp.Choices[0].Message.Content != "ok" {
		t.Errorf("content = %q", resp.Choices[0].Message.Content)
	}
	if gotAuth != "Bearer secret" {
		t.Errorf("auth header = %q", gotAuth)
	}
	if gotHeader != "myapp" {
		t.Errorf("custom header = %q", gotHeader)
	}
	if gotPath != "/chat/completions" {
		t.Errorf("path = %q", gotPath)
	}
	if gotBody.Model != "default-model" {
		t.Errorf("model defaulting failed: %q", gotBody.Model)
	}
}

func TestChatProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(w).Encode(ChatResponse{
			Error: &APIError{Message: "bad key", Type: "auth_error"},
		})
	}))
	defer srv.Close()

	c, _ := NewClient(Config{BaseURL: srv.URL, APIKey: "x", Model: "m"})
	if _, err := c.Chat(context.Background(), ChatRequest{Messages: []Message{{Role: "user", Content: "hi"}}}); err == nil {
		t.Error("expected provider error")
	}
}

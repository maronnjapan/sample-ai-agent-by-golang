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

// openAIRequest mirrors the subset of the OpenAI chat-completions request we
// want to assert on in tests.
type openAIRequest struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content any    `json:"content"`
	} `json:"messages"`
	Tools []struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	} `json:"tools"`
}

// TestChatSendsExpectedRequest drives the langchaingo-backed client against a
// mock OpenAI-compatible server and verifies the wire request carries the
// auth header, custom headers, model, tools, and messages we configured.
func TestChatSendsExpectedRequest(t *testing.T) {
	var gotAuth, gotHeader, gotPath string
	var gotReq openAIRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotHeader = r.Header.Get("X-Title")
		gotPath = r.URL.Path
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotReq)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": "ok"}},
			},
		})
	}))
	defer srv.Close()

	c, err := NewClient(Config{
		BaseURL: srv.URL + "/v1",
		APIKey:  "secret",
		Model:   "default-model",
		Headers: map[string]string{"X-Title": "myapp"},
	})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}

	resp, err := c.Chat(context.Background(), ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "hi"}},
		Tools: []Tool{{
			Type: "function",
			Function: FunctionDefinition{
				Name:        "calc",
				Description: "do math",
				Parameters:  json.RawMessage(`{"type":"object"}`),
			},
		}},
		ToolChoice: "auto",
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
	if gotPath != "/v1/chat/completions" {
		t.Errorf("path = %q", gotPath)
	}
	if gotReq.Model != "default-model" {
		t.Errorf("model defaulting failed: %q", gotReq.Model)
	}
	if len(gotReq.Tools) != 1 || gotReq.Tools[0].Function.Name != "calc" {
		t.Errorf("tools not forwarded: %+v", gotReq.Tools)
	}
}

// TestChatDecodesToolCalls verifies the client maps an OpenAI tool_calls
// response back into the agent's ToolCall shape.
func TestChatDecodesToolCalls(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]any{
					"role": "assistant",
					"tool_calls": []map[string]any{{
						"id":   "call_1",
						"type": "function",
						"function": map[string]any{
							"name":      "calculator",
							"arguments": `{"expression":"2+2"}`,
						},
					}},
				},
				"finish_reason": "tool_calls",
			}},
		})
	}))
	defer srv.Close()

	c, _ := NewClient(Config{BaseURL: srv.URL + "/v1", APIKey: "x", Model: "m"})
	resp, err := c.Chat(context.Background(), ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "what is 2+2?"}},
	})
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	calls := resp.Choices[0].Message.ToolCalls
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].ID != "call_1" || calls[0].Function.Name != "calculator" {
		t.Errorf("unexpected tool call: %+v", calls[0])
	}
	if calls[0].Function.Arguments != `{"expression":"2+2"}` {
		t.Errorf("unexpected arguments: %q", calls[0].Function.Arguments)
	}
}

// TestChatProviderError ensures a non-2xx provider response surfaces as an error.
func TestChatProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{"message": "bad key", "type": "auth_error"},
		})
	}))
	defer srv.Close()

	c, _ := NewClient(Config{BaseURL: srv.URL + "/v1", APIKey: "x", Model: "m"})
	if _, err := c.Chat(context.Background(), ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}); err == nil {
		t.Error("expected provider error")
	}
}

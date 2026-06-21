package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/llm"
	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/tools"
)

// mockServer returns an httptest server that replays a fixed sequence of
// chat-completion responses, one per request, so the agent loop can be driven
// deterministically without a real provider.
func mockServer(t *testing.T, responses []llm.ChatResponse) *httptest.Server {
	t.Helper()
	var calls int
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls >= len(responses) {
			t.Errorf("unexpected extra request #%d", calls)
			http.Error(w, "no more responses", http.StatusInternalServerError)
			return
		}
		resp := responses[calls]
		calls++
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
}

func newTestAgent(t *testing.T, baseURL string) *Agent {
	t.Helper()
	client, err := llm.NewClient(llm.Config{BaseURL: baseURL, APIKey: "test", Model: "test-model"})
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	reg := tools.NewRegistry()
	reg.MustRegister(tools.Calculator{})
	return New(client, reg, WithMaxSteps(5))
}

// TestRunDirectAnswer covers the simplest path: the model replies with text and
// no tool calls.
func TestRunDirectAnswer(t *testing.T) {
	srv := mockServer(t, []llm.ChatResponse{
		{Choices: []llm.Choice{{Message: llm.Message{Role: "assistant", Content: "Hello!"}}}},
	})
	defer srv.Close()

	ag := newTestAgent(t, srv.URL)
	conv := ag.NewConversation()
	got, err := ag.Run(context.Background(), conv, "hi", nil)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if got != "Hello!" {
		t.Errorf("got %q, want %q", got, "Hello!")
	}
}

// TestRunWithToolCall covers the full reason-act loop: the model requests a
// tool, the agent runs it, and the model uses the result to answer.
func TestRunWithToolCall(t *testing.T) {
	toolCall := llm.ToolCall{
		ID:       "call_1",
		Type:     "function",
		Function: llm.FunctionCall{Name: "calculator", Arguments: `{"expression":"2+2"}`},
	}
	srv := mockServer(t, []llm.ChatResponse{
		{Choices: []llm.Choice{{Message: llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{toolCall}}}}},
		{Choices: []llm.Choice{{Message: llm.Message{Role: "assistant", Content: "The answer is 4."}}}},
	})
	defer srv.Close()

	ag := newTestAgent(t, srv.URL)
	conv := ag.NewConversation()

	var sawToolCall, sawToolResult bool
	obs := func(e Event) {
		switch e.Type {
		case EventToolCall:
			if e.ToolName == "calculator" {
				sawToolCall = true
			}
		case EventToolResult:
			if e.ToolResult == "4" {
				sawToolResult = true
			}
		}
	}

	got, err := ag.Run(context.Background(), conv, "what is 2+2?", obs)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if got != "The answer is 4." {
		t.Errorf("got %q, want %q", got, "The answer is 4.")
	}
	if !sawToolCall || !sawToolResult {
		t.Errorf("expected tool call+result events; got call=%v result=%v", sawToolCall, sawToolResult)
	}
	// History should contain the tool result message wired back to the call ID.
	var foundToolMsg bool
	for _, m := range conv.Messages {
		if m.Role == llm.RoleTool && m.ToolCallID == "call_1" && m.Content == "4" {
			foundToolMsg = true
		}
	}
	if !foundToolMsg {
		t.Error("expected a tool-result message in conversation history")
	}
}

// TestRunUnknownTool verifies a model hallucinating a tool name does not crash
// the loop; the error is fed back as text and the model can recover.
func TestRunUnknownTool(t *testing.T) {
	bad := llm.ToolCall{ID: "c1", Type: "function", Function: llm.FunctionCall{Name: "nope", Arguments: "{}"}}
	srv := mockServer(t, []llm.ChatResponse{
		{Choices: []llm.Choice{{Message: llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{bad}}}}},
		{Choices: []llm.Choice{{Message: llm.Message{Role: "assistant", Content: "Sorry, recovered."}}}},
	})
	defer srv.Close()

	ag := newTestAgent(t, srv.URL)
	conv := ag.NewConversation()
	got, err := ag.Run(context.Background(), conv, "do something", nil)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if got != "Sorry, recovered." {
		t.Errorf("got %q", got)
	}
}

// TestMaxStepsGuard ensures a model that loops forever on tool calls is stopped.
func TestMaxStepsGuard(t *testing.T) {
	loop := llm.ChatResponse{Choices: []llm.Choice{{Message: llm.Message{
		Role:      "assistant",
		ToolCalls: []llm.ToolCall{{ID: "c", Type: "function", Function: llm.FunctionCall{Name: "calculator", Arguments: `{"expression":"1+1"}`}}},
	}}}}
	srv := mockServer(t, []llm.ChatResponse{loop, loop, loop, loop, loop})
	defer srv.Close()

	ag := newTestAgent(t, srv.URL)
	conv := ag.NewConversation()
	if _, err := ag.Run(context.Background(), conv, "loop", nil); err == nil {
		t.Error("expected max-steps error")
	}
}

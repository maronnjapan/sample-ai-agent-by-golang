// Package llm provides a provider-agnostic client for chat-completion style
// large language model APIs. The actual transport and protocol handling is
// delegated to the langchaingo library (github.com/tmc/langchaingo), so the
// same client targets OpenAI, OpenRouter, or any other OpenAI-compatible
// gateway. This package adapts langchaingo's generic LLM interface to the small
// request/response shape the agent loop works with.
package llm

import "encoding/json"

// Role identifies who produced a message in a conversation.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Message is a single entry in a chat conversation.
type Message struct {
	Role string `json:"role"`
	// Content is the text body. For assistant messages that only request tool
	// calls it may be empty.
	Content string `json:"content"`
	// Name is the optional author name (used for tool results to echo the tool).
	Name string `json:"name,omitempty"`
	// ToolCalls is populated on assistant messages that ask to invoke tools.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID links a tool-result message back to the call that produced it.
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// ToolCall is a request from the model to invoke a named tool with JSON args.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall holds the tool name and its raw JSON-encoded arguments.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool describes a callable function the model is allowed to request.
type Tool struct {
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`
}

// FunctionDefinition is the schema advertised to the model for one tool.
type FunctionDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ChatRequest is a single chat-completion request.
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Tools       []Tool    `json:"tools,omitempty"`
	ToolChoice  string    `json:"tool_choice,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

// ChatResponse is the decoded response from a chat-completion request.
type ChatResponse struct {
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Choice is a single completion candidate.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage reports token accounting for a request, when the provider supplies it.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

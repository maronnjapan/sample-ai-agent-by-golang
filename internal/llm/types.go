// Package llm provides a provider-agnostic client for chat-completion style
// large language model APIs. It speaks the OpenAI "chat completions" wire
// format, which is also implemented by OpenRouter and many other gateways, so a
// single client can target multiple providers.
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

// ChatRequest is the payload sent to the chat-completions endpoint.
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Tools       []Tool    `json:"tools,omitempty"`
	ToolChoice  string    `json:"tool_choice,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

// ChatResponse is the decoded response from the chat-completions endpoint.
type ChatResponse struct {
	ID      string   `json:"id"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
	// Error is populated when the provider returns an error envelope.
	Error *APIError `json:"error,omitempty"`
}

// Choice is a single completion candidate.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage reports token accounting for a request.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// APIError is the error envelope returned by OpenAI-compatible providers.
type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    any    `json:"code"`
}

func (e *APIError) Error() string {
	if e == nil {
		return ""
	}
	if e.Type != "" {
		return e.Type + ": " + e.Message
	}
	return e.Message
}

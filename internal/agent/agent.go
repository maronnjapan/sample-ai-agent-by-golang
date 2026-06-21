// Package agent implements the core reason-act loop: it sends the conversation
// to the model, executes any tools the model requests, feeds the results back,
// and repeats until the model produces a final answer.
package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/llm"
	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/tools"
)

// DefaultSystemPrompt steers the agent toward using its tools rather than
// guessing.
const DefaultSystemPrompt = "You are a helpful AI assistant with access to tools. " +
	"When a tool can produce a more accurate answer than your own reasoning " +
	"(for example arithmetic, the current time, or fetching a web page), prefer " +
	"calling the tool. Think step by step and give concise, correct answers."

// Event describes something that happened during a Run, surfaced via a callback
// so callers (CLIs, tests, UIs) can render progress without the agent knowing
// about any particular frontend.
type Event struct {
	Type EventType
	// Text holds assistant content for EventAssistantMessage / EventFinalAnswer.
	Text string
	// ToolName / ToolArgs / ToolResult describe tool activity.
	ToolName   string
	ToolArgs   string
	ToolResult string
	// Err is set for EventToolError.
	Err error
}

// EventType enumerates the kinds of Event the agent emits.
type EventType int

const (
	// EventToolCall fires just before a tool is invoked.
	EventToolCall EventType = iota
	// EventToolResult fires after a tool returns successfully.
	EventToolResult
	// EventToolError fires when a tool call fails.
	EventToolError
	// EventFinalAnswer fires once with the model's final textual answer.
	EventFinalAnswer
)

// Observer receives Events during a Run. It may be nil.
type Observer func(Event)

// Agent ties together an LLM client and a tool registry.
type Agent struct {
	client   *llm.Client
	registry *tools.Registry

	systemPrompt string
	maxSteps     int
	temperature  *float64
}

// Option customises an Agent.
type Option func(*Agent)

// WithSystemPrompt overrides the default system prompt.
func WithSystemPrompt(p string) Option {
	return func(a *Agent) {
		if p != "" {
			a.systemPrompt = p
		}
	}
}

// WithMaxSteps caps how many model/tool round-trips a single Run may take,
// preventing runaway tool loops. Values < 1 are ignored.
func WithMaxSteps(n int) Option {
	return func(a *Agent) {
		if n > 0 {
			a.maxSteps = n
		}
	}
}

// WithTemperature sets the sampling temperature passed to the model.
func WithTemperature(t float64) Option {
	return func(a *Agent) { a.temperature = &t }
}

// New constructs an Agent.
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

// Conversation holds the running message history for a multi-turn session.
type Conversation struct {
	Messages []llm.Message
}

// NewConversation starts a conversation seeded with the agent's system prompt.
func (a *Agent) NewConversation() *Conversation {
	return &Conversation{
		Messages: []llm.Message{{Role: llm.RoleSystem, Content: a.systemPrompt}},
	}
}

// Run advances the conversation by one user turn: it appends the user input,
// drives the reason-act loop until the model yields a final answer (or the step
// budget is exhausted), and returns that answer. The conversation is mutated in
// place so it can be reused across turns.
func (a *Agent) Run(ctx context.Context, conv *Conversation, userInput string, obs Observer) (string, error) {
	if conv == nil {
		return "", fmt.Errorf("agent: conversation is nil")
	}
	conv.Messages = append(conv.Messages, llm.Message{Role: llm.RoleUser, Content: userInput})

	toolDefs := a.registry.Definitions()

	for step := 0; step < a.maxSteps; step++ {
		req := llm.ChatRequest{
			Messages:    conv.Messages,
			Temperature: a.temperature,
		}
		if len(toolDefs) > 0 {
			req.Tools = toolDefs
			req.ToolChoice = "auto"
		}

		resp, err := a.client.Chat(ctx, req)
		if err != nil {
			return "", fmt.Errorf("agent: chat request: %w", err)
		}

		msg := resp.Choices[0].Message
		// Record the assistant turn (content and/or tool calls) verbatim so the
		// provider sees a well-formed history on the next round-trip.
		conv.Messages = append(conv.Messages, msg)

		if len(msg.ToolCalls) == 0 {
			emit(obs, Event{Type: EventFinalAnswer, Text: msg.Content})
			return msg.Content, nil
		}

		// Execute every requested tool call and append its result.
		for _, tc := range msg.ToolCalls {
			result := a.invokeTool(ctx, tc, obs)
			conv.Messages = append(conv.Messages, llm.Message{
				Role:       llm.RoleTool,
				Name:       tc.Function.Name,
				ToolCallID: tc.ID,
				Content:    result,
			})
		}
	}

	return "", fmt.Errorf("agent: exceeded max steps (%d) without a final answer", a.maxSteps)
}

// invokeTool runs a single tool call and returns the string to feed back to the
// model. Tool errors are returned as text (not Go errors) so the model can see
// what went wrong and recover, rather than aborting the whole run.
func (a *Agent) invokeTool(ctx context.Context, tc llm.ToolCall, obs Observer) string {
	name := tc.Function.Name
	rawArgs := json.RawMessage(tc.Function.Arguments)
	emit(obs, Event{Type: EventToolCall, ToolName: name, ToolArgs: tc.Function.Arguments})

	tool, ok := a.registry.Get(name)
	if !ok {
		err := fmt.Errorf("unknown tool %q", name)
		emit(obs, Event{Type: EventToolError, ToolName: name, Err: err})
		return "Error: " + err.Error()
	}

	result, err := tool.Call(ctx, rawArgs)
	if err != nil {
		emit(obs, Event{Type: EventToolError, ToolName: name, Err: err})
		return "Error: " + err.Error()
	}

	emit(obs, Event{Type: EventToolResult, ToolName: name, ToolResult: result})
	return result
}

func emit(obs Observer, e Event) {
	if obs != nil {
		obs(e)
	}
}

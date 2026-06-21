package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

// Provider identifies a known LLM gateway with sensible default endpoints.
type Provider string

const (
	// ProviderOpenAI targets the official OpenAI API.
	ProviderOpenAI Provider = "openai"
	// ProviderOpenRouter targets OpenRouter, an OpenAI-compatible gateway that
	// fronts many model vendors behind one API.
	ProviderOpenRouter Provider = "openrouter"
)

// defaultBaseURLs maps a known provider to its chat-completions base URL.
var defaultBaseURLs = map[Provider]string{
	ProviderOpenAI:     "https://api.openai.com/v1",
	ProviderOpenRouter: "https://openrouter.ai/api/v1",
}

// DefaultBaseURL returns the canonical base URL for a known provider, or an
// empty string when the provider is unrecognised.
func DefaultBaseURL(p Provider) string {
	return defaultBaseURLs[Provider(strings.ToLower(string(p)))]
}

// Config configures a Client. Only APIKey and one of Provider/BaseURL are
// required; everything else has a usable default.
type Config struct {
	// Provider selects a built-in endpoint. Ignored when BaseURL is set.
	Provider Provider
	// BaseURL overrides the endpoint entirely (handy for self-hosted or
	// proxy gateways that also speak the OpenAI format).
	BaseURL string
	// APIKey authenticates requests via the Authorization: Bearer header.
	APIKey string
	// Model is the default model name used when a request omits one.
	Model string
	// Timeout bounds a single HTTP request. Defaults to 120s.
	Timeout time.Duration
	// HTTPClient lets callers inject a custom transport (e.g. for tests).
	HTTPClient *http.Client
	// Headers are extra headers attached to every request. OpenRouter, for
	// example, recommends HTTP-Referer and X-Title.
	Headers map[string]string
}

// Client wraps a langchaingo LLM and adapts it to the agent's request/response
// shape. The library handles the OpenAI wire protocol, auth, and transport.
type Client struct {
	llm   *openai.LLM
	model string
}

// NewClient builds a Client from cfg, applying provider defaults and
// validating that the essentials are present.
func NewClient(cfg Config) (*Client, error) {
	baseURL := strings.TrimRight(cfg.BaseURL, "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL(cfg.Provider)
	}
	if baseURL == "" {
		return nil, fmt.Errorf("llm: no base URL: set BaseURL or a known Provider (got %q)", cfg.Provider)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("llm: APIKey is required")
	}

	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: timeout}
	}
	// Wrap the transport so any extra headers (e.g. OpenRouter ranking headers)
	// are attached to every request langchaingo issues.
	if len(cfg.Headers) > 0 {
		base := httpClient.Transport
		if base == nil {
			base = http.DefaultTransport
		}
		clone := *httpClient
		clone.Transport = &headerTransport{base: base, headers: cfg.Headers}
		httpClient = &clone
	}

	opts := []openai.Option{
		openai.WithToken(cfg.APIKey),
		openai.WithBaseURL(baseURL),
		openai.WithHTTPClient(httpClient),
	}
	if cfg.Model != "" {
		opts = append(opts, openai.WithModel(cfg.Model))
	}

	model, err := openai.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("llm: init langchaingo client: %w", err)
	}

	return &Client{llm: model, model: cfg.Model}, nil
}

// Model returns the client's default model name.
func (c *Client) Model() string { return c.model }

// Chat performs a single chat-completion request and returns the decoded
// response. When req.Model is empty the client's default model is used.
func (c *Client) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	model := req.Model
	if model == "" {
		model = c.model
	}
	if model == "" {
		return nil, fmt.Errorf("llm: no model specified")
	}

	messages, err := toLangChainMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	callOpts := []llms.CallOption{llms.WithModel(model)}
	if req.Temperature != nil {
		callOpts = append(callOpts, llms.WithTemperature(*req.Temperature))
	}
	if req.MaxTokens > 0 {
		callOpts = append(callOpts, llms.WithMaxTokens(req.MaxTokens))
	}
	if len(req.Tools) > 0 {
		tools, err := toLangChainTools(req.Tools)
		if err != nil {
			return nil, err
		}
		callOpts = append(callOpts, llms.WithTools(tools))
		if req.ToolChoice != "" {
			callOpts = append(callOpts, llms.WithToolChoice(req.ToolChoice))
		}
	}

	resp, err := c.llm.GenerateContent(ctx, messages, callOpts...)
	if err != nil {
		return nil, fmt.Errorf("llm: request failed: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("llm: response contained no choices")
	}

	return fromLangChainResponse(resp), nil
}

// toLangChainMessages converts the agent's conversation history into the
// langchaingo message representation, preserving tool calls and tool results so
// the model sees a well-formed multi-turn history.
func toLangChainMessages(in []Message) ([]llms.MessageContent, error) {
	out := make([]llms.MessageContent, 0, len(in))
	for _, m := range in {
		switch m.Role {
		case RoleSystem:
			out = append(out, llms.TextParts(llms.ChatMessageTypeSystem, m.Content))
		case RoleUser:
			out = append(out, llms.TextParts(llms.ChatMessageTypeHuman, m.Content))
		case RoleAssistant:
			parts := []llms.ContentPart{}
			if m.Content != "" {
				parts = append(parts, llms.TextContent{Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				parts = append(parts, llms.ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					FunctionCall: &llms.FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				})
			}
			out = append(out, llms.MessageContent{Role: llms.ChatMessageTypeAI, Parts: parts})
		case RoleTool:
			out = append(out, llms.MessageContent{
				Role: llms.ChatMessageTypeTool,
				Parts: []llms.ContentPart{llms.ToolCallResponse{
					ToolCallID: m.ToolCallID,
					Name:       m.Name,
					Content:    m.Content,
				}},
			})
		default:
			return nil, fmt.Errorf("llm: unknown message role %q", m.Role)
		}
	}
	return out, nil
}

// toLangChainTools renders the agent's tool definitions into langchaingo's tool
// schema. The JSON-Schema parameters are decoded into a generic structure
// because langchaingo expects an `any`, not raw bytes.
func toLangChainTools(in []Tool) ([]llms.Tool, error) {
	out := make([]llms.Tool, 0, len(in))
	for _, t := range in {
		var params any
		if len(t.Function.Parameters) > 0 {
			if err := json.Unmarshal(t.Function.Parameters, &params); err != nil {
				return nil, fmt.Errorf("llm: tool %q has invalid parameters schema: %w", t.Function.Name, err)
			}
		}
		out = append(out, llms.Tool{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  params,
			},
		})
	}
	return out, nil
}

// fromLangChainResponse maps langchaingo's first content choice back into the
// agent's ChatResponse shape, reconstructing any requested tool calls.
func fromLangChainResponse(resp *llms.ContentResponse) *ChatResponse {
	choice := resp.Choices[0]

	msg := Message{Role: RoleAssistant, Content: choice.Content}
	for _, tc := range choice.ToolCalls {
		call := ToolCall{ID: tc.ID, Type: tc.Type}
		if call.Type == "" {
			call.Type = "function"
		}
		if tc.FunctionCall != nil {
			call.Function = FunctionCall{
				Name:      tc.FunctionCall.Name,
				Arguments: tc.FunctionCall.Arguments,
			}
		}
		msg.ToolCalls = append(msg.ToolCalls, call)
	}

	out := &ChatResponse{
		Choices: []Choice{{Message: msg, FinishReason: choice.StopReason}},
	}
	out.Usage = usageFromGenerationInfo(choice.GenerationInfo)
	return out
}

// usageFromGenerationInfo extracts token accounting when the provider reports it
// in the choice's generation info. Missing fields default to zero.
func usageFromGenerationInfo(info map[string]any) Usage {
	var u Usage
	if info == nil {
		return u
	}
	asInt := func(v any) int {
		switch n := v.(type) {
		case int:
			return n
		case int64:
			return int(n)
		case float64:
			return int(n)
		}
		return 0
	}
	u.PromptTokens = asInt(info["PromptTokens"])
	u.CompletionTokens = asInt(info["CompletionTokens"])
	u.TotalTokens = asInt(info["TotalTokens"])
	return u
}

// headerTransport injects a fixed set of headers onto every outgoing request.
type headerTransport struct {
	base    http.RoundTripper
	headers map[string]string
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Clone to avoid mutating a request the caller may reuse.
	clone := req.Clone(req.Context())
	for k, v := range t.headers {
		clone.Header.Set(k, v)
	}
	return t.base.RoundTrip(clone)
}

package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
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

// Client is a minimal, dependency-free chat-completions client.
type Client struct {
	baseURL string
	apiKey  string
	model   string
	headers map[string]string
	http    *http.Client
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

	return &Client{
		baseURL: baseURL,
		apiKey:  cfg.APIKey,
		model:   cfg.Model,
		headers: cfg.Headers,
		http:    httpClient,
	}, nil
}

// Model returns the client's default model name.
func (c *Client) Model() string { return c.model }

// Chat performs a single chat-completion request and returns the decoded
// response. When req.Model is empty the client's default model is used.
func (c *Client) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	if req.Model == "" {
		req.Model = c.model
	}
	if req.Model == "" {
		return nil, fmt.Errorf("llm: no model specified")
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal request: %w", err)
	}

	url := c.baseURL + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("llm: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	for k, v := range c.headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llm: request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("llm: read response: %w", err)
	}

	var out ChatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		// Fall back to surfacing the raw body so non-JSON errors (e.g. an HTML
		// gateway page) are not swallowed.
		return nil, fmt.Errorf("llm: decode response (status %d): %w: %s", resp.StatusCode, err, truncate(string(raw), 500))
	}
	if out.Error != nil {
		return nil, fmt.Errorf("llm: provider error (status %d): %w", resp.StatusCode, out.Error)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("llm: unexpected status %d: %s", resp.StatusCode, truncate(string(raw), 500))
	}
	if len(out.Choices) == 0 {
		return nil, fmt.Errorf("llm: response contained no choices")
	}
	return &out, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

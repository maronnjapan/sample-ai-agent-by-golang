package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// HTTPGet fetches the body of a URL over HTTP(S). It is deliberately read-only
// (GET requests against http/https only) and caps the response size so the
// agent can pull in reference material without becoming an open proxy.
type HTTPGet struct {
	// Client is injectable for tests; defaults to a 30s-timeout client.
	Client *http.Client
	// MaxBytes caps how much of the response body is returned. Defaults to 16KiB.
	MaxBytes int64
}

// Name implements Tool.
func (HTTPGet) Name() string { return "http_get" }

// Description implements Tool.
func (HTTPGet) Description() string {
	return "Perform an HTTP GET request against a public http or https URL and " +
		"return the response body as text (truncated). Useful for fetching web " +
		"pages, JSON APIs, or documentation referenced by the user."
}

// Parameters implements Tool.
func (HTTPGet) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"url": {
				"type": "string",
				"description": "The absolute http(s) URL to fetch."
			}
		},
		"required": ["url"]
	}`)
}

type httpGetArgs struct {
	URL string `json:"url"`
}

// Call implements Tool.
func (h HTTPGet) Call(ctx context.Context, args json.RawMessage) (string, error) {
	var a httpGetArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return "", fmt.Errorf("http_get: invalid arguments: %w", err)
	}
	url := strings.TrimSpace(a.URL)
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		return "", fmt.Errorf("http_get: url must start with http:// or https://")
	}

	client := h.Client
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	maxBytes := h.MaxBytes
	if maxBytes == 0 {
		maxBytes = 16 * 1024
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("http_get: build request: %w", err)
	}
	req.Header.Set("User-Agent", "sample-ai-agent-by-golang/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("http_get: request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBytes))
	if err != nil {
		return "", fmt.Errorf("http_get: read body: %w", err)
	}

	result := fmt.Sprintf("HTTP %d %s\n\n%s", resp.StatusCode, resp.Status, string(body))
	if int64(len(body)) >= maxBytes {
		result += "\n\n[truncated]"
	}
	return result, nil
}

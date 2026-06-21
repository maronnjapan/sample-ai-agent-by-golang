# sample-ai-agent-by-golang

A small, dependency-free **AI agent** written in pure Go. It implements a
classic *reason–act* loop: the agent sends the conversation to a large language
model, the model can request **tools** (function calling), the agent runs those
tools and feeds the results back, and the loop repeats until the model produces
a final answer.

The LLM client speaks the OpenAI **chat completions** wire format, so the same
binary works against **OpenAI**, **OpenRouter**, or any other OpenAI-compatible
gateway (Azure OpenAI, Groq, Together, a local `llama.cpp` server, …) just by
changing configuration — no code changes required.

## Highlights

- **Pure Go, zero third-party dependencies** — only the standard library.
- **Provider-agnostic** — switch between OpenAI and OpenRouter via env vars.
- **Tool / function calling** with a clean `Tool` interface and registry.
- **Built-in tools**: `calculator` (exact arithmetic), `current_time`
  (timezone-aware clock), `http_get` (read-only web fetch).
- **Interactive REPL** and **one-shot** modes.
- **Well tested** — agent loop is exercised against a mock provider; tools and
  config have unit tests.

## Architecture

```
cmd/agent           CLI: REPL + one-shot mode, tool trace rendering
internal/config     Env/.env configuration loading & provider defaults
internal/llm        OpenAI-compatible chat-completions client + types
internal/agent      The reason–act loop (model ⇄ tools) and event stream
internal/tools      Tool interface, registry, and built-in tools
```

The agent depends only on two abstractions — an `*llm.Client` and a
`*tools.Registry` — which keeps each piece independently testable.

```
 user input
     │
     ▼
 ┌────────────┐   tools[]    ┌─────────────┐
 │   Agent    │ ───────────► │  LLM (chat) │
 │  Run loop  │ ◄─────────── │  completion │
 └────┬───────┘  tool_calls  └─────────────┘
      │ execute requested tools
      ▼
 ┌────────────┐
 │  Registry  │  calculator / current_time / http_get
 └────────────┘
      │ results fed back as tool messages
      └──────────────► loop until final answer
```

## Getting started

### 1. Configure

```bash
cp .env.example .env
# edit .env and set your API key
```

Minimal OpenAI setup:

```bash
export AGENT_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

OpenRouter setup:

```bash
export AGENT_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-...
export AGENT_MODEL=openai/gpt-4o-mini   # any model OpenRouter exposes
```

Any other OpenAI-compatible endpoint:

```bash
export AGENT_BASE_URL=http://localhost:11434/v1   # e.g. a local server
export AGENT_API_KEY=whatever
export AGENT_MODEL=llama3.1
```

### 2. Run

Interactive REPL:

```bash
go run ./cmd/agent
```

One-shot:

```bash
go run ./cmd/agent -p "What is (12345 * 6789) + 2^10, and what's the time in Tokyo?"
# or simply
go run ./cmd/agent "Summarise https://example.com"
```

REPL commands: `/reset` (clear history), `/exit`.

Flags: `-p <prompt>` one-shot, `-quiet` hide the tool trace, `-verbose` also
show tool results.

## Configuration reference

| Variable              | Default                       | Description                                            |
| --------------------- | ----------------------------- | ------------------------------------------------------ |
| `AGENT_PROVIDER`      | `openai`                      | `openai` or `openrouter`                               |
| `AGENT_BASE_URL`      | provider default              | Override endpoint for any OpenAI-compatible gateway    |
| `AGENT_API_KEY`       | —                             | Generic API key (preferred)                            |
| `OPENAI_API_KEY`      | —                             | Fallback key when provider is `openai`                 |
| `OPENROUTER_API_KEY`  | —                             | Fallback key when provider is `openrouter`             |
| `AGENT_MODEL`         | `gpt-4o-mini` / `openai/...`  | Model name                                             |
| `AGENT_TEMPERATURE`   | `0.7`                         | Sampling temperature                                   |
| `AGENT_MAX_STEPS`     | `10`                          | Max model⇄tool round-trips per turn (loop guard)       |
| `AGENT_SYSTEM_PROMPT` | built-in                      | Override the system prompt                             |
| `OPENROUTER_REFERER`  | —                             | Optional OpenRouter `HTTP-Referer` ranking header      |
| `OPENROUTER_TITLE`    | —                             | Optional OpenRouter `X-Title` ranking header           |

## Adding a tool

Implement the `tools.Tool` interface and register it:

```go
type Tool interface {
    Name() string
    Description() string
    Parameters() json.RawMessage // JSON Schema for the arguments
    Call(ctx context.Context, args json.RawMessage) (string, error)
}
```

```go
registry.MustRegister(MyTool{})
```

The registry renders each tool's schema into the format the model expects, and
the agent loop dispatches calls to it automatically. Tool errors are returned to
the model as text so it can recover, rather than aborting the run.

## Development

```bash
make test    # go test ./...
make vet     # go vet ./...
make build   # -> bin/agent
make fmt     # gofmt -w .
```

CI (GitHub Actions) runs gofmt-check, `go vet`, `go test -race`, and `go build`
on every push and pull request.

## License

MIT — see [LICENSE](LICENSE).

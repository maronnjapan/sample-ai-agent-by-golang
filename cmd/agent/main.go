// Command agent is a CLI front-end for the AI agent. It supports a one-shot
// mode (pass a prompt as arguments or via -p) and an interactive REPL.
//
// Configuration is read from the environment; see the internal/config package
// or .env.example for the recognised variables. The same binary targets OpenAI,
// OpenRouter, or any other OpenAI-compatible gateway by switching AGENT_PROVIDER
// or AGENT_BASE_URL.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/agent"
	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/config"
	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/llm"
	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/tools"
)

func main() {
	var (
		prompt  = flag.String("p", "", "run a single prompt and exit (non-interactive)")
		quiet   = flag.Bool("quiet", false, "suppress tool-call trace output")
		verbose = flag.Bool("verbose", false, "show tool results in the trace")
	)
	flag.Parse()

	if err := run(*prompt, *quiet, *verbose); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(prompt string, quiet, verbose bool) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	client, err := llm.NewClient(llm.Config{
		Provider: cfg.Provider,
		BaseURL:  cfg.BaseURL,
		APIKey:   cfg.APIKey,
		Model:    cfg.Model,
		Headers:  cfg.ExtraHeaders,
	})
	if err != nil {
		return err
	}

	registry := buildToolRegistry()

	opts := []agent.Option{
		agent.WithMaxSteps(cfg.MaxSteps),
		agent.WithTemperature(cfg.Temperature),
	}
	if cfg.SystemPrompt != "" {
		opts = append(opts, agent.WithSystemPrompt(cfg.SystemPrompt))
	}
	ag := agent.New(client, registry, opts...)

	// Cancel in-flight work cleanly on Ctrl-C.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	obs := newObserver(quiet, verbose)

	// One-shot mode: -p flag, or a prompt passed as positional arguments.
	oneShot := strings.TrimSpace(prompt)
	if oneShot == "" && flag.NArg() > 0 {
		oneShot = strings.Join(flag.Args(), " ")
	}
	if oneShot != "" {
		conv := ag.NewConversation()
		answer, err := ag.Run(ctx, conv, oneShot, obs)
		if err != nil {
			return err
		}
		fmt.Println(answer)
		return nil
	}

	return repl(ctx, ag, obs, cfg, registry)
}

// buildToolRegistry wires up the built-in tools.
func buildToolRegistry() *tools.Registry {
	registry := tools.NewRegistry()
	registry.MustRegister(tools.Calculator{})
	registry.MustRegister(tools.Clock{})
	registry.MustRegister(tools.HTTPGet{})
	return registry
}

// repl runs the interactive loop, preserving conversation history across turns.
func repl(ctx context.Context, ag *agent.Agent, obs agent.Observer, cfg *config.Config, registry *tools.Registry) error {
	fmt.Printf("AI Agent (provider=%s model=%s, tools: %s)\n",
		cfg.Provider, cfg.Model, strings.Join(registry.Names(), ", "))
	fmt.Println("Type your message and press Enter. Commands: /reset, /exit")

	conv := ag.NewConversation()
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			fmt.Println()
			return scanner.Err()
		}
		line := strings.TrimSpace(scanner.Text())
		switch {
		case line == "":
			continue
		case line == "/exit" || line == "/quit":
			return nil
		case line == "/reset":
			conv = ag.NewConversation()
			fmt.Println("(conversation reset)")
			continue
		}

		answer, err := ag.Run(ctx, conv, line, obs)
		if err != nil {
			if ctx.Err() != nil {
				return nil // interrupted
			}
			fmt.Fprintln(os.Stderr, "error:", err)
			continue
		}
		fmt.Printf("\n%s\n", answer)
	}
}

// newObserver returns an Observer that prints a human-readable trace of tool
// activity, honouring the quiet/verbose flags.
func newObserver(quiet, verbose bool) agent.Observer {
	if quiet {
		return nil
	}
	return func(e agent.Event) {
		switch e.Type {
		case agent.EventToolCall:
			fmt.Printf("  \033[2m↳ %s(%s)\033[0m\n", e.ToolName, e.ToolArgs)
		case agent.EventToolResult:
			if verbose {
				fmt.Printf("  \033[2m  = %s\033[0m\n", oneLine(e.ToolResult))
			}
		case agent.EventToolError:
			fmt.Printf("  \033[31m  ! %s: %v\033[0m\n", e.ToolName, e.Err)
		}
	}
}

func oneLine(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > 200 {
		return s[:200] + "…"
	}
	return s
}

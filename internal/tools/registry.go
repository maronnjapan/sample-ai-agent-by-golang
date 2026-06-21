// Package tools defines the tool abstraction the agent exposes to the model and
// ships a handful of safe, dependency-free built-in tools.
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/maronnjapan/sample-ai-agent-by-golang/internal/llm"
)

// Tool is a single capability the agent can offer to the model. Implementations
// must be safe to call concurrently.
type Tool interface {
	// Name is the unique identifier advertised to the model.
	Name() string
	// Description tells the model when and how to use the tool.
	Description() string
	// Parameters returns the JSON Schema describing the tool's arguments.
	Parameters() json.RawMessage
	// Call executes the tool with the raw JSON arguments supplied by the model
	// and returns a string result to feed back into the conversation.
	Call(ctx context.Context, args json.RawMessage) (string, error)
}

// Registry holds the set of tools available to an agent.
type Registry struct {
	tools map[string]Tool
}

// NewRegistry creates an empty Registry.
func NewRegistry() *Registry {
	return &Registry{tools: make(map[string]Tool)}
}

// Register adds a tool, returning an error on duplicate names so wiring
// mistakes fail loudly at startup.
func (r *Registry) Register(t Tool) error {
	if t == nil {
		return fmt.Errorf("tools: cannot register nil tool")
	}
	name := t.Name()
	if name == "" {
		return fmt.Errorf("tools: tool has empty name")
	}
	if _, exists := r.tools[name]; exists {
		return fmt.Errorf("tools: duplicate tool name %q", name)
	}
	r.tools[name] = t
	return nil
}

// MustRegister is like Register but panics on error; convenient for static
// wiring of trusted built-in tools.
func (r *Registry) MustRegister(t Tool) {
	if err := r.Register(t); err != nil {
		panic(err)
	}
}

// Get returns a tool by name and whether it was found.
func (r *Registry) Get(name string) (Tool, bool) {
	t, ok := r.tools[name]
	return t, ok
}

// Len reports how many tools are registered.
func (r *Registry) Len() int { return len(r.tools) }

// Names returns the registered tool names in sorted order.
func (r *Registry) Names() []string {
	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Definitions renders every registered tool into the llm.Tool schema the model
// understands, ordered deterministically by name.
func (r *Registry) Definitions() []llm.Tool {
	defs := make([]llm.Tool, 0, len(r.tools))
	for _, name := range r.Names() {
		t := r.tools[name]
		defs = append(defs, llm.Tool{
			Type: "function",
			Function: llm.FunctionDefinition{
				Name:        t.Name(),
				Description: t.Description(),
				Parameters:  t.Parameters(),
			},
		})
	}
	return defs
}

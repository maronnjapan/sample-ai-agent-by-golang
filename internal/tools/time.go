package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// Clock reports the current date and time. It lets the agent answer
// time-sensitive questions, which a static model otherwise cannot.
type Clock struct {
	// Now is injectable for deterministic tests; defaults to time.Now.
	Now func() time.Time
}

// Name implements Tool.
func (Clock) Name() string { return "current_time" }

// Description implements Tool.
func (Clock) Description() string {
	return "Return the current date and time. Use this whenever the user asks " +
		"about the present moment, today's date, or the current time."
}

// Parameters implements Tool.
func (Clock) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"timezone": {
				"type": "string",
				"description": "Optional IANA timezone name (e.g. \"Asia/Tokyo\", \"UTC\"). Defaults to UTC."
			}
		}
	}`)
}

type clockArgs struct {
	Timezone string `json:"timezone"`
}

// Call implements Tool.
func (c Clock) Call(_ context.Context, args json.RawMessage) (string, error) {
	var a clockArgs
	if len(args) > 0 {
		if err := json.Unmarshal(args, &a); err != nil {
			return "", fmt.Errorf("current_time: invalid arguments: %w", err)
		}
	}
	now := time.Now
	if c.Now != nil {
		now = c.Now
	}

	loc := time.UTC
	if tz := strings.TrimSpace(a.Timezone); tz != "" {
		l, err := time.LoadLocation(tz)
		if err != nil {
			return "", fmt.Errorf("current_time: unknown timezone %q", tz)
		}
		loc = l
	}
	return now().In(loc).Format("2006-01-02 15:04:05 MST (Mon)"), nil
}

package tools

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestCalculator(t *testing.T) {
	cases := []struct {
		expr string
		want string
	}{
		{"1 + 2", "3"},
		{"2 * 3 + 4", "10"},
		{"2 * (3 + 4)", "14"},
		{"2 ^ 3 ^ 2", "512"}, // right associative
		{"-5 + 3", "-2"},
		{"10 / 4", "2.5"},
		{"10 % 3", "1"},
		{"2 * pi", "6.283185307179586"},
		{"(1 + 2) * (3 + 4)", "21"},
	}
	c := Calculator{}
	for _, tc := range cases {
		args, _ := json.Marshal(calcArgs{Expression: tc.expr})
		got, err := c.Call(context.Background(), args)
		if err != nil {
			t.Errorf("%q: unexpected error: %v", tc.expr, err)
			continue
		}
		if got != tc.want {
			t.Errorf("%q = %q, want %q", tc.expr, got, tc.want)
		}
	}
}

func TestCalculatorErrors(t *testing.T) {
	c := Calculator{}
	for _, expr := range []string{"1 / 0", "1 +", "(1 + 2", "foo(3)", ""} {
		args, _ := json.Marshal(calcArgs{Expression: expr})
		if _, err := c.Call(context.Background(), args); err == nil {
			t.Errorf("%q: expected error, got nil", expr)
		}
	}
}

func TestClock(t *testing.T) {
	fixed := time.Date(2026, 6, 21, 12, 0, 0, 0, time.UTC)
	c := Clock{Now: func() time.Time { return fixed }}
	got, err := c.Call(context.Background(), json.RawMessage(`{"timezone":"UTC"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "2026-06-21 12:00:00 UTC (Sun)"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
	if _, err := c.Call(context.Background(), json.RawMessage(`{"timezone":"Not/AZone"}`)); err == nil {
		t.Error("expected error for invalid timezone")
	}
}

func TestHTTPGet(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("hello world"))
	}))
	defer srv.Close()

	h := HTTPGet{Client: srv.Client()}
	args, _ := json.Marshal(httpGetArgs{URL: srv.URL})
	got, err := h.Call(context.Background(), args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "hello world"; !contains(got, want) {
		t.Errorf("response %q does not contain %q", got, want)
	}

	if _, err := h.Call(context.Background(), json.RawMessage(`{"url":"ftp://x"}`)); err == nil {
		t.Error("expected error for non-http scheme")
	}
}

func TestRegistry(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(Calculator{}); err != nil {
		t.Fatalf("register: %v", err)
	}
	if err := r.Register(Calculator{}); err == nil {
		t.Error("expected duplicate registration to fail")
	}
	if r.Len() != 1 {
		t.Errorf("Len = %d, want 1", r.Len())
	}
	if _, ok := r.Get("calculator"); !ok {
		t.Error("expected to find calculator")
	}
	defs := r.Definitions()
	if len(defs) != 1 || defs[0].Function.Name != "calculator" {
		t.Errorf("unexpected definitions: %+v", defs)
	}
	// Parameters must be valid JSON.
	var schema map[string]any
	if err := json.Unmarshal(defs[0].Function.Parameters, &schema); err != nil {
		t.Errorf("calculator parameters are not valid JSON: %v", err)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

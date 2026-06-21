package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"unicode"
)

// Calculator evaluates arithmetic expressions. It is a self-contained
// shunting-yard parser so the agent can do exact math without relying on the
// model's notoriously unreliable mental arithmetic.
type Calculator struct{}

// Name implements Tool.
func (Calculator) Name() string { return "calculator" }

// Description implements Tool.
func (Calculator) Description() string {
	return "Evaluate a mathematical expression and return the numeric result. " +
		"Supports + - * / %, parentheses, unary minus, exponentiation with ^, " +
		"and the constants pi and e. Use this for any arithmetic instead of computing it yourself."
}

// Parameters implements Tool.
func (Calculator) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"expression": {
				"type": "string",
				"description": "The arithmetic expression to evaluate, e.g. \"(3 + 4) * 2 ^ 3\"."
			}
		},
		"required": ["expression"]
	}`)
}

type calcArgs struct {
	Expression string `json:"expression"`
}

// Call implements Tool.
func (Calculator) Call(_ context.Context, args json.RawMessage) (string, error) {
	var a calcArgs
	if err := json.Unmarshal(args, &a); err != nil {
		return "", fmt.Errorf("calculator: invalid arguments: %w", err)
	}
	if strings.TrimSpace(a.Expression) == "" {
		return "", fmt.Errorf("calculator: empty expression")
	}
	result, err := evalExpression(a.Expression)
	if err != nil {
		return "", err
	}
	// Render integers without a trailing .0 for readability.
	if result == math.Trunc(result) && !math.IsInf(result, 0) {
		return fmt.Sprintf("%d", int64(result)), nil
	}
	return fmt.Sprintf("%g", result), nil
}

// --- minimal expression evaluator (shunting-yard + RPN) ---

type token struct {
	kind  tokenKind
	value float64
	op    byte
}

type tokenKind int

const (
	tokNumber tokenKind = iota
	tokOperator
	tokLParen
	tokRParen
)

func evalExpression(expr string) (float64, error) {
	tokens, err := tokenize(expr)
	if err != nil {
		return 0, err
	}
	rpn, err := toRPN(tokens)
	if err != nil {
		return 0, err
	}
	return evalRPN(rpn)
}

func tokenize(expr string) ([]token, error) {
	var tokens []token
	runes := []rune(strings.ReplaceAll(expr, " ", ""))
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		switch {
		case unicode.IsDigit(r) || r == '.':
			start := i
			for i < len(runes) && (unicode.IsDigit(runes[i]) || runes[i] == '.') {
				i++
			}
			i--
			var v float64
			if _, err := fmt.Sscanf(string(runes[start:i+1]), "%g", &v); err != nil {
				return nil, fmt.Errorf("calculator: invalid number %q", string(runes[start:i+1]))
			}
			tokens = append(tokens, token{kind: tokNumber, value: v})
		case r == '(':
			tokens = append(tokens, token{kind: tokLParen})
		case r == ')':
			tokens = append(tokens, token{kind: tokRParen})
		case isOperator(byte(r)):
			tokens = append(tokens, token{kind: tokOperator, op: byte(r)})
		case unicode.IsLetter(r):
			start := i
			for i < len(runes) && unicode.IsLetter(runes[i]) {
				i++
			}
			i--
			name := strings.ToLower(string(runes[start : i+1]))
			switch name {
			case "pi":
				tokens = append(tokens, token{kind: tokNumber, value: math.Pi})
			case "e":
				tokens = append(tokens, token{kind: tokNumber, value: math.E})
			default:
				return nil, fmt.Errorf("calculator: unknown identifier %q", name)
			}
		default:
			return nil, fmt.Errorf("calculator: unexpected character %q", string(r))
		}
	}
	return tokens, nil
}

func isOperator(b byte) bool {
	switch b {
	case '+', '-', '*', '/', '%', '^':
		return true
	}
	return false
}

func precedence(op byte) int {
	switch op {
	case '+', '-':
		return 1
	case '*', '/', '%':
		return 2
	case '^':
		return 3
	}
	return 0
}

func rightAssociative(op byte) bool { return op == '^' }

func toRPN(tokens []token) ([]token, error) {
	var output []token
	var ops []token
	for idx, t := range tokens {
		switch t.kind {
		case tokNumber:
			output = append(output, t)
		case tokOperator:
			// Detect unary minus/plus: an operator at the start, or following
			// another operator or an opening paren.
			if (t.op == '-' || t.op == '+') && isUnaryPosition(tokens, idx) {
				// Rewrite unary +/- as 0 +/- x by injecting a zero operand.
				output = append(output, token{kind: tokNumber, value: 0})
			}
			for len(ops) > 0 {
				top := ops[len(ops)-1]
				if top.kind == tokOperator &&
					(precedence(top.op) > precedence(t.op) ||
						(precedence(top.op) == precedence(t.op) && !rightAssociative(t.op))) {
					output = append(output, top)
					ops = ops[:len(ops)-1]
					continue
				}
				break
			}
			ops = append(ops, t)
		case tokLParen:
			ops = append(ops, t)
		case tokRParen:
			found := false
			for len(ops) > 0 {
				top := ops[len(ops)-1]
				ops = ops[:len(ops)-1]
				if top.kind == tokLParen {
					found = true
					break
				}
				output = append(output, top)
			}
			if !found {
				return nil, fmt.Errorf("calculator: mismatched parentheses")
			}
		}
	}
	for len(ops) > 0 {
		top := ops[len(ops)-1]
		ops = ops[:len(ops)-1]
		if top.kind == tokLParen {
			return nil, fmt.Errorf("calculator: mismatched parentheses")
		}
		output = append(output, top)
	}
	return output, nil
}

func isUnaryPosition(tokens []token, idx int) bool {
	if idx == 0 {
		return true
	}
	prev := tokens[idx-1]
	return prev.kind == tokOperator || prev.kind == tokLParen
}

func evalRPN(rpn []token) (float64, error) {
	var stack []float64
	pop := func() (float64, error) {
		if len(stack) == 0 {
			return 0, fmt.Errorf("calculator: malformed expression")
		}
		v := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		return v, nil
	}
	for _, t := range rpn {
		if t.kind == tokNumber {
			stack = append(stack, t.value)
			continue
		}
		b, err := pop()
		if err != nil {
			return 0, err
		}
		a, err := pop()
		if err != nil {
			return 0, err
		}
		var res float64
		switch t.op {
		case '+':
			res = a + b
		case '-':
			res = a - b
		case '*':
			res = a * b
		case '/':
			if b == 0 {
				return 0, fmt.Errorf("calculator: division by zero")
			}
			res = a / b
		case '%':
			if b == 0 {
				return 0, fmt.Errorf("calculator: modulo by zero")
			}
			res = math.Mod(a, b)
		case '^':
			res = math.Pow(a, b)
		default:
			return 0, fmt.Errorf("calculator: unknown operator %q", string(t.op))
		}
		stack = append(stack, res)
	}
	if len(stack) != 1 {
		return 0, fmt.Errorf("calculator: malformed expression")
	}
	return stack[0], nil
}

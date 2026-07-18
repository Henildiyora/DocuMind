// Package cliui holds small terminal output helpers shared across commands:
// coloring (auto-disabled on non-TTY), simple status lines, and an interactive
// multiple-choice picker used by the ambiguity clarifier.
package cliui

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/manifoldco/promptui"
	"github.com/mattn/go-isatty"
)

var colorEnabled = isatty.IsTerminal(os.Stdout.Fd()) || isatty.IsCygwinTerminal(os.Stdout.Fd())

// ANSI color codes, applied only when stdout is a TTY.
const (
	reset  = "\033[0m"
	bold   = "\033[1m"
	dim    = "\033[2m"
	red    = "\033[31m"
	green  = "\033[32m"
	yellow = "\033[33m"
	cyan   = "\033[36m"
)

func wrap(code, s string) string {
	if !colorEnabled {
		return s
	}
	return code + s + reset
}

// Bold, Dim, Green, etc. colorize a string for terminal output.
func Bold(s string) string   { return wrap(bold, s) }
func Dim(s string) string    { return wrap(dim, s) }
func Green(s string) string  { return wrap(green, s) }
func Yellow(s string) string { return wrap(yellow, s) }
func Red(s string) string    { return wrap(red, s) }
func Cyan(s string) string   { return wrap(cyan, s) }

// Info prints an informational line to stdout.
func Info(format string, a ...any) {
	fmt.Fprintln(os.Stdout, fmt.Sprintf(format, a...))
}

// Success prints a green status line.
func Success(format string, a ...any) {
	fmt.Fprintln(os.Stdout, Green(fmt.Sprintf(format, a...)))
}

// Warn prints a yellow warning line to stderr.
func Warn(format string, a ...any) {
	fmt.Fprintln(os.Stderr, Yellow(fmt.Sprintf(format, a...)))
}

// Errorln prints a red error line to stderr.
func Errorln(format string, a ...any) {
	fmt.Fprintln(os.Stderr, Red(fmt.Sprintf(format, a...)))
}

// IsTTY reports whether stdin is interactive.
func IsTTY() bool {
	return isatty.IsTerminal(os.Stdin.Fd())
}

// Confirm asks a yes/no question and returns the answer. On a non-interactive
// terminal it returns def without prompting.
func Confirm(question string, def bool) bool {
	if !IsTTY() {
		return def
	}
	suffix := " [y/N] "
	if def {
		suffix = " [Y/n] "
	}
	fmt.Fprint(os.Stdout, question+suffix)
	line, err := bufio.NewReader(os.Stdin).ReadString('\n')
	if err != nil {
		return def
	}
	line = strings.ToLower(strings.TrimSpace(line))
	if line == "" {
		return def
	}
	return line == "y" || line == "yes"
}

// Pick presents an arrow-key selectable list and returns the chosen option. On a
// non-interactive terminal it prints the numbered options and returns ("", false)
// so the caller can ask the user to re-run with a clearer query.
func Pick(question string, options []string) (string, bool) {
	if !IsTTY() {
		fmt.Fprintln(os.Stderr, Yellow(question))
		for i, opt := range options {
			fmt.Fprintf(os.Stderr, "  %d. %s\n", i+1, opt)
		}
		fmt.Fprintln(os.Stderr, Dim("Re-run with a clearer question, or use an interactive terminal to pick."))
		return "", false
	}
	prompt := promptui.Select{
		Label: question,
		Items: options,
	}
	_, choice, err := prompt.Run()
	if err != nil {
		return "", false
	}
	return choice, true
}

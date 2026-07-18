// Package cli wires the cobra command tree for the documind binary. Each
// subcommand lives in its own file. The root command owns global flags (--debug,
// --version).
package cli

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

// Version is the documind CLI version.
const Version = "0.1.0-go"

// debug is set by the persistent --debug flag; commands print full error chains
// when true, otherwise short user-facing messages.
var debug bool

// NewRootCmd builds the root command with all subcommands attached.
func NewRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:           "documind",
		Short:         "Fast, local hybrid search and Q&A for your codebase.",
		Long:          "DocuMind: pure-local hybrid search (keyword + vectors) with optional local LLM answers via Ollama.",
		SilenceUsage:  true,
		SilenceErrors: true,
		Version:       Version,
	}
	root.PersistentFlags().BoolVarP(&debug, "debug", "v", false, "Show full error details.")

	root.AddCommand(
		newIndexCmd(),
		newResetCmd(),
		newSearchCmd(),
		newAskCmd(),
		newChatCmd(),
		newSetupCmd(),
		newModelsCmd(),
		newDoctorCmd(),
	)
	return root
}

// Execute runs the root command and maps errors to exit codes.
func Execute() int {
	if err := NewRootCmd().Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	return 0
}

// resolveRoot returns an absolute project root from an optional path (default:
// current working directory).
func resolveRoot(path string) (string, error) {
	if path == "" {
		wd, err := os.Getwd()
		if err != nil {
			return "", err
		}
		return wd, nil
	}
	return filepath.Abs(path)
}

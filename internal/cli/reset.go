package cli

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/Henildiyora/DocuMind/internal/cliui"
	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/indexer"
	"github.com/spf13/cobra"
)

func newResetCmd() *cobra.Command {
	var path string
	var yes bool

	cmd := &cobra.Command{
		Use:   "reset",
		Short: "Delete the project's index directory (.documind/).",
		RunE: func(cmd *cobra.Command, args []string) error {
			root, err := resolveRoot(path)
			if err != nil {
				return err
			}
			cfg := config.Load()
			dir := cfg.IndexDirFor(root)
			if _, err := os.Stat(dir); err != nil {
				cliui.Warn("Nothing to delete.")
				return nil
			}
			if !yes {
				fmt.Printf("Delete %s? [y/N] ", dir)
				reader := bufio.NewReader(os.Stdin)
				line, _ := reader.ReadString('\n')
				if strings.ToLower(strings.TrimSpace(line)) != "y" {
					return nil
				}
			}
			if err := indexer.Reset(root, cfg); err != nil {
				return err
			}
			cliui.Success("Removed %s", dir)
			return nil
		},
	}
	cmd.Flags().StringVarP(&path, "path", "p", "", "Project root (default: cwd).")
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Skip confirmation.")
	return cmd
}

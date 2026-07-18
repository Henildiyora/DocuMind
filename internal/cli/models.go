package cli

import (
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/Henildiyora/DocuMind/internal/cliui"
	"github.com/Henildiyora/DocuMind/internal/models"
	"github.com/spf13/cobra"
)

func newModelsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "models",
		Short: "List the free/local Ollama model catalog (sizes, RAM, tradeoffs).",
		RunE: func(cmd *cobra.Command, args []string) error {
			cliui.Info("%s", cliui.Bold("DocuMind model catalog (all free, all local)"))
			w := tabwriter.NewWriter(os.Stdout, 0, 2, 2, ' ', 0)
			fmt.Fprintln(w, "TAG\tSIZE\tRAM\tSPEED/QUALITY\tDESCRIPTION")
			for _, m := range models.Catalog {
				fmt.Fprintf(w, "%s\t%.1f GB\t~%.0f GB\t%s\t%s\n", m.Tag, m.SizeGB, m.RAMGb, m.Tradeoff, m.BestFor)
			}
			w.Flush()
			cliui.Info("%s", cliui.Dim("Run `documind setup` to pick one that fits your machine. Override anytime with --model <tag>."))
			return nil
		},
	}
}

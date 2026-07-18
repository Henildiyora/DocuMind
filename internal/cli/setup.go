package cli

import (
	"context"
	"fmt"
	"time"

	"github.com/Henildiyora/DocuMind/internal/cliui"
	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/models"
	"github.com/Henildiyora/DocuMind/internal/ollama"
	"github.com/Henildiyora/DocuMind/internal/structural"
	"github.com/spf13/cobra"
)

func newSetupCmd() *cobra.Command {
	var path, model, keepAlive string
	var yes bool
	var pull bool

	cmd := &cobra.Command{
		Use:   "setup",
		Short: "Pick a local model for ask/chat (hardware-aware).",
		Long: "Detect your hardware, recommend the best local model that fits, save the\n" +
			"choice to ~/.config/documind/config.toml, and optionally pull it. Search and\n" +
			"index never need a model, so this is optional.",
		RunE: func(cmd *cobra.Command, args []string) error {
			root, err := resolveRoot(path)
			if err != nil {
				return err
			}
			keepAliveSet := cmd.Flags().Changed("keep-alive")
			pullSet := cmd.Flags().Changed("pull")
			return runSetup(root, setupOptions{
				Model:        model,
				KeepAlive:    keepAlive,
				KeepAliveSet: keepAliveSet,
				Yes:          yes,
				Pull:         pull,
				PullSet:      pullSet,
			})
		},
	}
	cmd.Flags().StringVarP(&path, "path", "p", "", "Project to scan for the recommendation.")
	cmd.Flags().StringVarP(&model, "model", "m", "", "Force a specific Ollama model tag.")
	cmd.Flags().StringVar(&keepAlive, "keep-alive", "", "Default keep_alive for ask/chat (e.g. 5m, 0).")
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "Accept the recommendation without prompting.")
	cmd.Flags().BoolVar(&pull, "pull", false, "Pull the chosen model via Ollama.")
	return cmd
}

type setupOptions struct {
	Model        string
	KeepAlive    string
	KeepAliveSet bool
	Yes          bool
	Pull         bool
	PullSet      bool
}

// runSetup runs the interactive/scripted setup flow. Reused by the inline
// post-index offer.
func runSetup(root string, opts setupOptions) error {
	cliui.Info("%s", cliui.Bold("DocuMind setup"))
	cliui.Info("%s", cliui.Dim("100% free, 100% local. No API keys."))

	facts, err := structural.GatherFacts(root, config.Load())
	if err != nil {
		return err
	}
	cliui.Info("Scanned %s: %d indexable files", cliui.Cyan(baseName(root)), facts.FileCount)

	ram := models.DetectSystemRAMGB()
	if ram > 0 {
		cliui.Info("Detected ~%.0f GB system RAM", ram)
	}
	if gpu := models.DetectGPUHint(); gpu != "" {
		cliui.Info("GPU hint: %s", gpu)
	}

	spec := selectSpec(opts, facts.FileCount, ram)
	cliui.Success("Selected model: %s (%s)", spec.Tag, spec.Family)

	// Persist preference.
	updates := map[string]any{"model": spec.Tag, "setup_done": true}
	if opts.KeepAliveSet {
		updates["keep_alive"] = opts.KeepAlive
	}
	cfgPath, err := config.UpdateUser(updates)
	if err != nil {
		return err
	}
	cliui.Info("%s", cliui.Dim("Saved to "+cfgPath))
	cliui.Success("Ready. Search and index already work; this enables `documind ask` / `documind chat`.")

	// Optionally pull the model.
	if !wantPull(opts) {
		cliui.Info("%s", cliui.Dim("Skipping model download. Pull later with: ollama pull "+spec.Tag))
		return nil
	}
	return pullModel(spec.Tag, opts.PullSet && opts.Pull)
}

// selectSpec resolves the model to use from an explicit override, the
// recommendation, or an interactive pick.
func selectSpec(opts setupOptions, fileCount int, ram float64) models.Spec {
	if opts.Model != "" {
		if s, ok := models.FindByTag(opts.Model); ok {
			return s
		}
		return models.Spec{Tier: "custom", Tag: opts.Model, Family: "custom", BestFor: "User-specified model", Tradeoff: "balanced"}
	}
	recommended := models.RecommendForHardware(ram, fileCount, 0)
	candidates := models.FilterForRAM(ram, 0.6)

	if opts.Yes || !cliui.IsTTY() {
		cliui.Info("Using recommended model: %s", recommended.Tag)
		return recommended
	}

	labels := make([]string, len(candidates))
	recIdx := 0
	for i, c := range candidates {
		mark := ""
		if c.Tag == recommended.Tag {
			mark = " (recommended)"
			recIdx = i
		}
		labels[i] = fmt.Sprintf("%s — %.1f GB, ~%.0f GB RAM, %s%s", c.Tag, c.SizeGB, c.RAMGb, c.Tradeoff, mark)
	}
	// Put the recommended option first for convenience.
	if recIdx != 0 {
		labels[0], labels[recIdx] = labels[recIdx], labels[0]
		candidates[0], candidates[recIdx] = candidates[recIdx], candidates[0]
	}
	choice, ok := cliui.Pick("Pick a model", labels)
	if !ok {
		return recommended
	}
	for i, l := range labels {
		if l == choice {
			return candidates[i]
		}
	}
	return recommended
}

func wantPull(opts setupOptions) bool {
	if opts.PullSet {
		return opts.Pull
	}
	if opts.Yes {
		return true
	}
	return cliui.Confirm("Pull the model now? (free, local)", true)
}

// pullModel pulls a model via Ollama, rendering typed errors as short, actionable
// lines. mustSucceed controls whether a failure is returned as an error.
func pullModel(tag string, mustSucceed bool) error {
	if !ollama.Installed() {
		msg := "Ollama is not installed. Install it, then run: ollama pull " + tag
		if mustSucceed {
			return fmt.Errorf("%s", msg)
		}
		cliui.Warn("%s", msg)
		return nil
	}
	client := ollama.New(config.Load())
	if !client.Ping() {
		msg := "Ollama daemon is not running. Start it with: ollama serve"
		if mustSucceed {
			return fmt.Errorf("%s", msg)
		}
		cliui.Warn("%s", msg)
		return nil
	}
	if client.ModelAvailable(tag) {
		cliui.Success("Model already pulled: %s", tag)
		return nil
	}
	cliui.Info("Pulling %s ... (this can take a few minutes)", tag)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	if err := client.Pull(ctx, tag); err != nil {
		if mustSucceed {
			return fmt.Errorf("failed to pull %s: retry with `ollama pull %s`", tag, tag)
		}
		cliui.Errorln("Failed to pull %s: %v", tag, err)
		return nil
	}
	cliui.Success("Pulled %s", tag)
	return nil
}

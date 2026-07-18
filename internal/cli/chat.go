package cli

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Henildiyora/DocuMind/internal/cliui"
	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/ollama"
	"github.com/Henildiyora/DocuMind/internal/prompts"
	"github.com/Henildiyora/DocuMind/internal/router"
	"github.com/Henildiyora/DocuMind/internal/search"
	"github.com/Henildiyora/DocuMind/internal/structural"
	"github.com/Henildiyora/DocuMind/internal/threads"
	"github.com/spf13/cobra"
)

const slashHelp = `Slash commands:
  /help               show this help
  /clear              clear conversation memory for this thread
  /k N                set top-k for retrieval
  /model M            switch LLM model for this session
  /new <name>         create and switch to a new thread
  /switch <name>      switch to an existing thread
  /threads            list threads with last-message preview
  /rename <old> <new> rename a thread
  /delete <name>      delete a thread
  /exit               quit`

func newChatCmd() *cobra.Command {
	var path, model, thread, keepAlive string
	var k int

	cmd := &cobra.Command{
		Use:   "chat",
		Short: "Interactive chat grounded in your project.",
		Long:  "Start a REPL grounded in your project. Threads persist under .documind/chats/.",
		RunE: func(cmd *cobra.Command, args []string) error {
			root, err := resolveRoot(path)
			if err != nil {
				return err
			}
			cfg := config.Load()
			if model != "" {
				cfg.Model = model
			}
			if k > 0 {
				cfg.TopK = k
			}
			if cmd.Flags().Changed("keep-alive") {
				cfg.KeepAlive = keepAlive
			}
			if !indexReady(root, cfg) {
				return fmt.Errorf("no index found for this project. Run: documind index %s", root)
			}
			client, ready := llmReady(cfg)
			if !ready {
				return fmt.Errorf("chat needs a local model. Run: documind setup (or ensure Ollama is running and the model is pulled)")
			}
			return runChat(root, cfg, thread, client)
		},
	}
	cmd.Flags().StringVarP(&path, "path", "p", "", "Project root.")
	cmd.Flags().StringVarP(&model, "model", "m", "", "Ollama model override.")
	cmd.Flags().StringVarP(&thread, "thread", "t", threads.DefaultThread, "Named chat thread.")
	cmd.Flags().StringVar(&keepAlive, "keep-alive", "", "How long Ollama keeps the model in RAM.")
	cmd.Flags().IntVarP(&k, "k", "k", 0, "Snippets per question.")
	return cmd
}

// runChat drives the interactive REPL with persistent named threads and windowed
// conversation memory.
func runChat(root string, cfg config.Config, threadName string, client *ollama.Client) error {
	bundle, err := openRetriever(root, cfg)
	if err != nil {
		return err
	}
	defer bundle.Close()

	current, err := threads.SanitizeName(threadName)
	if err != nil {
		cliui.Warn("%v", err)
		current = threads.DefaultThread
	}
	thread, err := threads.Ensure(root, current, cfg)
	if err != nil {
		return err
	}

	sessionModel := cfg.Model
	sessionK := cfg.TopK
	historyTurns := cfg.ChatHistoryTurns

	cliui.Info("%s", cliui.Bold("DocuMind chat"))
	cliui.Info("model: %s  |  thread: %s  |  project: %s", cliui.Cyan(sessionModel), cliui.Cyan(thread.Name), cliui.Green(baseName(root)))
	cliui.Info("%s", cliui.Dim("Type /help for commands, /exit to quit. keep_alive="+cfg.KeepAlive))

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Fprint(os.Stdout, cliui.Bold("you")+" > ")
		line, readErr := reader.ReadString('\n')
		user := strings.TrimSpace(line)
		if readErr != nil && user == "" {
			fmt.Fprintln(os.Stdout)
			break
		}
		if user == "" {
			continue
		}

		if strings.HasPrefix(user, "/") {
			quit, newThread := handleSlash(root, cfg, user, thread, &sessionModel, &sessionK)
			if quit {
				break
			}
			if newThread != nil {
				thread = newThread
			}
			continue
		}

		turnCfg := cfg
		turnCfg.Model = sessionModel
		turnCfg.TopK = sessionK

		if err := chatTurn(root, turnCfg, bundle, client, thread, user, historyTurns); err != nil {
			cliui.Errorln("%v", err)
		}
	}
	return nil
}

// chatTurn runs one question: route, retrieve, answer with windowed memory, and
// persist the turn.
func chatTurn(root string, cfg config.Config, bundle *retrieverBundle, client *ollama.Client, thread *threads.Thread, user string, historyTurns int) error {
	ctx := context.Background()

	decision := router.Classify(ctx, client, user, router.Options{})

	var answer string
	switch decision.Category {
	case router.Structural:
		facts, err := structural.GatherFacts(root, cfg)
		if err != nil {
			return err
		}
		cliui.Info("%s", cliui.Bold("documind"))
		messages := withHistory(thread, historyTurns, structuralMessages(user, facts.Summary()))
		answer, err = streamAnswer(client, messages, cfg)
		if err != nil {
			return err
		}
	default:
		hits, err := bundle.Retriever.Search(ctx, user, cfg.TopK)
		if err != nil {
			return err
		}
		hits = search.Refine(hits, user, cfg)
		if len(hits) == 0 {
			cliui.Warn("No matches found.")
			return nil
		}
		cliui.Info("%s", cliui.Bold("documind"))
		messages := withHistory(thread, historyTurns, prompts.BuildAnswerMessages(user, search.HitsToContext(hits, 8000)))
		answer, err = streamAnswer(client, messages, cfg)
		if err != nil {
			return err
		}
		printSources(hits)
	}

	// Persist the turn (in-memory + disk), keeping a generous on-disk history.
	ts := nowStamp()
	thread.Messages = append(thread.Messages,
		threads.Message{Role: "user", Content: user, TS: ts},
		threads.Message{Role: "assistant", Content: answer, TS: ts},
	)
	if maxMsgs := historyTurns * 2 * 4; maxMsgs > 0 && len(thread.Messages) > maxMsgs && maxMsgs >= 32 {
		thread.Messages = thread.Messages[len(thread.Messages)-maxMsgs:]
	}
	return threads.Save(root, thread, cfg)
}

// withHistory prepends the thread's windowed memory to answer-synthesis messages.
// Memory is fed ONLY to answer prompts, never to classify/clarify (which stay
// single-purpose).
func withHistory(thread *threads.Thread, turns int, messages []ollama.Message) []ollama.Message {
	history := thread.HistoryWindow(turns)
	if len(history) == 0 {
		return messages
	}
	// Insert history after the system message so grounding rules stay first.
	if len(messages) > 0 && messages[0].Role == "system" {
		out := []ollama.Message{messages[0]}
		out = append(out, history...)
		out = append(out, messages[1:]...)
		return out
	}
	return append(history, messages...)
}

// handleSlash processes a slash command. Returns (quit, newThread). newThread is
// non-nil when the active thread changed.
func handleSlash(root string, cfg config.Config, input string, thread *threads.Thread, sessionModel *string, sessionK *int) (bool, *threads.Thread) {
	parts := strings.Fields(input[1:])
	if len(parts) == 0 {
		return false, nil
	}
	cmd := strings.ToLower(parts[0])
	switch cmd {
	case "exit", "quit", "q":
		return true, nil
	case "help", "h", "?":
		cliui.Info("%s", slashHelp)
	case "clear":
		thread.Messages = nil
		_ = threads.Save(root, thread, cfg)
		cliui.Warn("History cleared.")
	case "k":
		if len(parts) == 2 {
			if n, err := strconv.Atoi(parts[1]); err == nil && n > 0 {
				*sessionK = n
				cliui.Warn("top_k -> %d", n)
			}
		}
	case "model":
		if len(parts) == 2 {
			*sessionModel = parts[1]
			cliui.Warn("model -> %s", parts[1])
		}
	case "new":
		if len(parts) >= 2 {
			t, err := threads.Ensure(root, parts[1], cfg)
			if err != nil {
				cliui.Errorln("%v", err)
				return false, nil
			}
			cliui.Warn("Switched to new thread %s", t.Name)
			return false, t
		}
	case "switch":
		if len(parts) >= 2 {
			t, err := threads.Load(root, parts[1], cfg)
			if err != nil {
				cliui.Errorln("%v", err)
				return false, nil
			}
			cliui.Warn("Switched to %s (%d msgs)", t.Name, len(t.Messages))
			return false, t
		}
	case "threads":
		list, _ := threads.List(root, cfg)
		if len(list) == 0 {
			cliui.Info("%s", cliui.Dim("No threads yet."))
			return false, nil
		}
		for _, t := range list {
			marker := ""
			if t.Name == thread.Name {
				marker = " *"
			}
			cliui.Info("  %s%s  %s  %s", cliui.Bold(t.Name), marker, cliui.Dim(t.UpdatedAt), t.Preview(60))
		}
	case "rename":
		if len(parts) >= 3 {
			t, err := threads.Rename(root, parts[1], parts[2], cfg)
			if err != nil {
				cliui.Errorln("%v", err)
				return false, nil
			}
			cliui.Warn("Renamed %s -> %s", parts[1], parts[2])
			if parts[1] == thread.Name {
				return false, t
			}
		}
	case "delete":
		if len(parts) >= 2 {
			name := parts[1]
			if name == thread.Name {
				cliui.Errorln("Switch away from this thread before deleting it.")
				return false, nil
			}
			if err := threads.Delete(root, name, cfg); err != nil {
				cliui.Errorln("%v", err)
				return false, nil
			}
			cliui.Warn("Deleted thread %s", name)
		}
	default:
		cliui.Errorln("Unknown command. Try /help.")
	}
	return false, nil
}

func baseName(p string) string {
	p = strings.TrimRight(p, "/")
	if i := strings.LastIndex(p, "/"); i >= 0 {
		return p[i+1:]
	}
	return p
}

func nowStamp() string {
	return time.Now().UTC().Truncate(time.Second).Format(time.RFC3339)
}

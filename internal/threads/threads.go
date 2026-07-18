// Package threads persists named chat conversations under
// <project>/.documind/chats/<name>.json. Shape mirrors the Python threads module:
// a thread has a name, timestamps, and an ordered list of user/assistant
// messages. A windowed slice of recent turns feeds the answer prompt (never the
// classify/clarify prompts, which stay single-purpose).
package threads

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/Henildiyora/DocuMind/internal/config"
	"github.com/Henildiyora/DocuMind/internal/ollama"
)

// DefaultThread is the thread name used by one-shot `documind ask`.
const DefaultThread = "default"

var nameRe = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

// Message is one turn in a conversation.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	TS      string `json:"ts,omitempty"`
}

// Thread is a named, persisted conversation.
type Thread struct {
	Name      string    `json:"name"`
	CreatedAt string    `json:"created_at"`
	UpdatedAt string    `json:"updated_at"`
	Messages  []Message `json:"messages"`
}

func nowUTC() string {
	return time.Now().UTC().Truncate(time.Second).Format(time.RFC3339)
}

// SanitizeName validates and returns a safe thread name.
func SanitizeName(name string) (string, error) {
	cleaned := strings.TrimSpace(name)
	if cleaned == "" || !nameRe.MatchString(cleaned) {
		return "", fmt.Errorf("invalid thread name %q. Use letters, digits, _ or - only", name)
	}
	return cleaned, nil
}

// ChatsDir returns <project>/.documind/chats.
func ChatsDir(projectRoot string, cfg config.Config) string {
	return filepath.Join(cfg.IndexDirFor(projectRoot), "chats")
}

func threadPath(projectRoot string, name string, cfg config.Config) (string, error) {
	safe, err := SanitizeName(name)
	if err != nil {
		return "", err
	}
	return filepath.Join(ChatsDir(projectRoot, cfg), safe+".json"), nil
}

// Load reads a thread from disk, or returns a fresh in-memory one if absent.
func Load(projectRoot, name string, cfg config.Config) (*Thread, error) {
	safe, err := SanitizeName(name)
	if err != nil {
		return nil, err
	}
	path, _ := threadPath(projectRoot, safe, cfg)
	data, err := os.ReadFile(path)
	if err != nil {
		return &Thread{Name: safe, CreatedAt: nowUTC(), UpdatedAt: nowUTC()}, nil
	}
	var t Thread
	if err := json.Unmarshal(data, &t); err != nil {
		return &Thread{Name: safe, CreatedAt: nowUTC(), UpdatedAt: nowUTC()}, nil
	}
	t.Name = safe
	if t.CreatedAt == "" {
		t.CreatedAt = nowUTC()
	}
	return &t, nil
}

// Save persists a thread, updating its UpdatedAt timestamp.
func Save(projectRoot string, t *Thread, cfg config.Config) error {
	path, err := threadPath(projectRoot, t.Name, cfg)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	t.UpdatedAt = nowUTC()
	data, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

// AppendTurn appends a user/assistant turn to a thread and saves it.
func AppendTurn(projectRoot, name, user, assistant string, cfg config.Config) error {
	t, err := Load(projectRoot, name, cfg)
	if err != nil {
		return err
	}
	ts := nowUTC()
	t.Messages = append(t.Messages,
		Message{Role: "user", Content: user, TS: ts},
		Message{Role: "assistant", Content: assistant, TS: ts},
	)
	return Save(projectRoot, t, cfg)
}

// List returns all threads sorted by UpdatedAt descending.
func List(projectRoot string, cfg config.Config) ([]*Thread, error) {
	dir := ChatsDir(projectRoot, cfg)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil // no chats dir yet: not an error
	}
	var out []*Thread
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		name := strings.TrimSuffix(e.Name(), ".json")
		t, err := Load(projectRoot, name, cfg)
		if err != nil {
			continue
		}
		out = append(out, t)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].UpdatedAt > out[j].UpdatedAt })
	return out, nil
}

// Ensure loads or creates and persists an empty thread.
func Ensure(projectRoot, name string, cfg config.Config) (*Thread, error) {
	t, err := Load(projectRoot, name, cfg)
	if err != nil {
		return nil, err
	}
	path, _ := threadPath(projectRoot, t.Name, cfg)
	if _, statErr := os.Stat(path); statErr != nil {
		if err := Save(projectRoot, t, cfg); err != nil {
			return nil, err
		}
	}
	return t, nil
}

// Rename renames a thread file and updates its name.
func Rename(projectRoot, oldName, newName string, cfg config.Config) (*Thread, error) {
	oldSafe, err := SanitizeName(oldName)
	if err != nil {
		return nil, err
	}
	newSafe, err := SanitizeName(newName)
	if err != nil {
		return nil, err
	}
	if oldSafe == newSafe {
		return Load(projectRoot, oldSafe, cfg)
	}
	srcPath, _ := threadPath(projectRoot, oldSafe, cfg)
	dstPath, _ := threadPath(projectRoot, newSafe, cfg)
	if _, err := os.Stat(srcPath); err != nil {
		return nil, fmt.Errorf("thread %q does not exist", oldSafe)
	}
	if _, err := os.Stat(dstPath); err == nil {
		return nil, fmt.Errorf("thread %q already exists", newSafe)
	}
	t, err := Load(projectRoot, oldSafe, cfg)
	if err != nil {
		return nil, err
	}
	t.Name = newSafe
	if err := Save(projectRoot, t, cfg); err != nil {
		return nil, err
	}
	_ = os.Remove(srcPath)
	return t, nil
}

// Delete removes a thread file if present.
func Delete(projectRoot, name string, cfg config.Config) error {
	path, err := threadPath(projectRoot, name, cfg)
	if err != nil {
		return err
	}
	if _, statErr := os.Stat(path); statErr == nil {
		return os.Remove(path)
	}
	return nil
}

// Preview returns a short last-message preview for `/threads`.
func (t *Thread) Preview(limit int) string {
	if len(t.Messages) == 0 {
		return "(empty)"
	}
	text := strings.ReplaceAll(strings.TrimSpace(t.Messages[len(t.Messages)-1].Content), "\n", " ")
	if len(text) > limit {
		return text[:limit-1] + "…"
	}
	if text == "" {
		return "(empty)"
	}
	return text
}

// HistoryWindow returns the last `turns` user+assistant pairs as chat messages
// for prompt memory. Truncates to the window; never includes timestamps.
func (t *Thread) HistoryWindow(turns int) []ollama.Message {
	n := turns * 2
	if n <= 0 || len(t.Messages) == 0 {
		return nil
	}
	start := len(t.Messages) - n
	if start < 0 {
		start = 0
	}
	var out []ollama.Message
	for _, m := range t.Messages[start:] {
		out = append(out, ollama.Message{Role: m.Role, Content: m.Content})
	}
	return out
}

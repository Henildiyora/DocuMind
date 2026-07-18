// Package ollama is a small HTTP client for the local Ollama daemon. It is the
// only inference dependency in DocuMind: embeddings and chat/generation both go
// through http://localhost:11434 (configurable). No other ML runtime is linked.
//
// Every capability is single-purpose by design so callers never accidentally
// combine "classify" and "answer" in one request (a core project ground rule).
package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os/exec"
	"strings"
	"time"

	"github.com/Henildiyora/DocuMind/internal/config"
)

// Typed errors surfaced to the CLI as short, actionable messages.
var (
	ErrNotInstalled   = errors.New("ollama is not installed")
	ErrNotRunning     = errors.New("ollama daemon is not running")
	ErrModelNotPulled = errors.New("model is not pulled")
	ErrPullFailed     = errors.New("model pull failed")
)

// Client talks to the Ollama daemon over HTTP.
type Client struct {
	baseURL string
	cfg     config.Config
	http    *http.Client
}

// New builds a client from config.
func New(cfg config.Config) *Client {
	return &Client{
		baseURL: strings.TrimRight(cfg.OllamaBaseURL, "/"),
		cfg:     cfg,
		http:    &http.Client{Timeout: 0}, // per-request contexts control timeouts
	}
}

// Installed reports whether the ollama binary is on PATH. This is a cheap check
// used to decide whether vector embedding is even possible.
func Installed() bool {
	path, err := exec.LookPath("ollama")
	if err != nil || path == "" {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return exec.CommandContext(ctx, "ollama", "--version").Run() == nil
}

// Ping reports whether the daemon responds. Fast, so callers can probe before
// attempting embeddings.
func (c *Client) Ping() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/version", nil)
	if err != nil {
		return false
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

type tagsResponse struct {
	Models []struct {
		Name  string `json:"name"`
		Model string `json:"model"`
	} `json:"models"`
}

// ModelAvailable reports whether the given model (or the configured one when
// empty) is pulled locally. Matches on full tag or the base name before ":".
func (c *Client) ModelAvailable(model string) bool {
	target := strings.TrimSpace(model)
	if target == "" {
		target = c.cfg.Model
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/tags", nil)
	if err != nil {
		return false
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false
	}
	var tags tagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		return false
	}
	base := func(s string) string { return strings.SplitN(s, ":", 2)[0] }
	for _, m := range tags.Models {
		name := m.Model
		if name == "" {
			name = m.Name
		}
		if name == target || base(name) == base(target) {
			return true
		}
	}
	return false
}

type embedRequest struct {
	Model     string   `json:"model"`
	Input     []string `json:"input"`
	KeepAlive string   `json:"keep_alive,omitempty"`
}

type embedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// EmbedBatch returns one dense vector per input text using the configured
// embedding model via /api/embed. Vectors are L2-normalized so cosine
// similarity equals a dot product downstream.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	// Ollama rejects empty strings; substitute a single space.
	clean := make([]string, len(texts))
	for i, t := range texts {
		if strings.TrimSpace(t) == "" {
			clean[i] = " "
		} else {
			clean[i] = t
		}
	}
	body, err := json.Marshal(embedRequest{
		Model:     c.cfg.EmbeddingModel,
		Input:     clean,
		KeepAlive: c.cfg.KeepAlive,
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNotRunning, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embed failed (%d): %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	var out embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	for i := range out.Embeddings {
		normalize(out.Embeddings[i])
	}
	return out.Embeddings, nil
}

// EmbedQuery embeds a single query string.
func (c *Client) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	vecs, err := c.EmbedBatch(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, errors.New("no embedding returned")
	}
	return vecs[0], nil
}

// Message is a single chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatOptions struct {
	Temperature float64 `json:"temperature"`
	NumCtx      int     `json:"num_ctx"`
}

type chatRequest struct {
	Model     string      `json:"model"`
	Messages  []Message   `json:"messages"`
	Stream    bool        `json:"stream"`
	Options   chatOptions `json:"options"`
	KeepAlive string      `json:"keep_alive,omitempty"`
	Format    string      `json:"format,omitempty"`
}

type chatStreamChunk struct {
	Message Message `json:"message"`
	Done    bool    `json:"done"`
}

// ChatStream streams assistant tokens for the given messages, invoking onToken
// for each content fragment. model overrides the configured chat model when
// non-empty. format (e.g. "json") constrains output; keep it empty for prose.
func (c *Client) ChatStream(ctx context.Context, messages []Message, model, format string, onToken func(string)) error {
	target := model
	if target == "" {
		target = c.cfg.Model
	}
	body, err := json.Marshal(chatRequest{
		Model:     target,
		Messages:  messages,
		Stream:    true,
		Options:   chatOptions{Temperature: c.cfg.LLMTemp, NumCtx: c.cfg.LLMNumCtx},
		KeepAlive: c.cfg.KeepAlive,
		Format:    format,
	})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrNotRunning, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("chat failed (%d): %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var chunk chatStreamChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			continue
		}
		if chunk.Message.Content != "" && onToken != nil {
			onToken(chunk.Message.Content)
		}
		if chunk.Done {
			break
		}
	}
	return scanner.Err()
}

// Chat is a non-streaming convenience wrapper returning the full reply.
func (c *Client) Chat(ctx context.Context, messages []Message, model, format string) (string, error) {
	var b strings.Builder
	err := c.ChatStream(ctx, messages, model, format, func(tok string) { b.WriteString(tok) })
	return b.String(), err
}

// Pull downloads a model via /api/pull (blocking until complete).
func (c *Client) Pull(ctx context.Context, model string) error {
	target := model
	if target == "" {
		target = c.cfg.Model
	}
	body, _ := json.Marshal(map[string]any{"model": target, "stream": false})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/pull", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrPullFailed, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%w (%d): %s", ErrPullFailed, resp.StatusCode, strings.TrimSpace(string(data)))
	}
	io.Copy(io.Discard, resp.Body)
	return nil
}

func normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}
